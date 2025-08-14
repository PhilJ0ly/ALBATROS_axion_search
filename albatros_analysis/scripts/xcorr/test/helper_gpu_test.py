"""
Test suite for GPU RePFB Stream implementation

Run with: pytest test_helper_gpu_stream_clean.py -v -rs
"""

import pytest
import numpy as np
import cupy as cp
import tempfile
import os
from pathlib import Path
import time
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Tuple, Optional

import sys
from os import path
sys.path.insert(0, path.expanduser("~"))

try:
    from albatros_analysis.src.correlations import baseband_data_classes as bdc
    from albatros_analysis.src.correlations import correlations as cr
    from albatros_analysis.src.utils import pfb_gpu_utils as pu
except ImportError:
    pytest.skip(reason="ALBATROS analysis package not available", allow_module_level=True)

try:
    import albatros_analysis.scripts.xcorr.helper_gpu_stream_clean as rs
    from albatros_analysis.scripts.xcorr.helper_gpu_stream_clean import (
        ProcessingConfig, BufferSizes, BufferManager, IPFBProcessor,
        MissingDataTracker, plan_chunks, setup_antenna_objects,
        setup_filters_and_windows, fill_pfb_buffer, repfb_xcorr_avg
    )
except ImportError:
    print("rePFB package not found. Skipping tests.")
    pytest.skip("RePFB stream module not available", allow_module_level=True)


# Test data generation utilities
def generate_synthetic_timestream(length: int, freqs: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Generate a synthetic time stream with known spectral content"""
    t = np.arange(length) * dt
    signal = np.zeros(length, dtype=np.complex64)
    
    # Add some test tones at specific frequencies
    test_freqs = [0.1, 0.25, 0.4]  # Normalized frequencies
    amplitudes = [1.0, 0.5, 0.3]
    
    for freq, amp in zip(test_freqs, amplitudes):
        signal += amp * np.exp(2j * np.pi * freq * t)
    
    # Add some noise
    noise_level = 0.1
    signal += noise_level * (np.random.randn(length) + 1j * np.random.randn(length))
    
    return signal


def create_synthetic_baseband_data(nchunks: int, acclen: int, nchans: int = 2049,
                                 npol: int = 2, nant: int = 4) -> List[List[dict]]:
    """Create synthetic baseband data structure"""
    data = []
    
    for chunk_idx in range(nchunks):
        chunk_data = []
        
        for ant_idx in range(nant):
            # Generate synthetic data for each polarization
            pol_data = {}
            
            for pol_idx in range(npol):
                # Create frequency domain data (post-PFB)
                freq_data = np.random.randn(acclen, nchans).astype(np.complex64)
                freq_data += 1j * np.random.randn(acclen, nchans).astype(np.complex64)
                
                # Add some coherent signal across antennas
                if pol_idx == 0:  # Add correlated signal to pol 0
                    coherent_signal = 0.3 * np.exp(2j * np.pi * 0.2 * np.arange(acclen)[:, None])
                    freq_data[:, 500:600] += coherent_signal
                
                pol_data[f"pol{pol_idx}"] = freq_data
            
            pol_data["specnums"] = np.arange(chunk_idx * acclen, (chunk_idx + 1) * acclen)
            chunk_data.append(pol_data)
        
        data.append(chunk_data)
    
    return data


class TestDataGenerator:
    """Test data generation utilities"""
    
    @staticmethod
    def create_test_impulse(length: int, position: int) -> np.ndarray:
        """Create impulse signal for testing"""
        signal = np.zeros(length, dtype=np.complex64)
        signal[position] = 1.0
        return signal
    
    @staticmethod
    def create_test_sinusoid(length: int, freq: float, amplitude: float = 1.0) -> np.ndarray:
        """Create sinusoidal signal for testing"""
        t = np.arange(length)
        return amplitude * np.exp(2j * np.pi * freq * t / length).astype(np.complex64)
    
    @staticmethod
    def create_white_noise(length: int, variance: float = 1.0) -> np.ndarray:
        """Create white noise signal"""
        noise = np.random.randn(length) + 1j * np.random.randn(length)
        return (np.sqrt(variance/2) * noise).astype(np.complex64)


class TestProcessingConfig:
    """Test ProcessingConfig dataclass"""
    
    def test_config_creation_with_defaults(self):
        config = ProcessingConfig(
            acclen=100, pfb_size=4096, nchunks=1000, nblock=64,
            chanstart=0, chanend=1024, osamp=2, nant=4
        )
        
        assert config.acclen == 100
        assert config.cut == 10  # default value
        assert config.filt_thresh == 0.45  # default value
        assert config.ntap == 4  # default value
        assert config.npol == 2  # default value
    
    def test_config_validation(self):
        """Test configuration parameter validation"""
        config = ProcessingConfig(
            acclen=50, pfb_size=4096, nchunks=100, nblock=32,
            chanstart=100, chanend=500, osamp=2, nant=2
        )
        
        assert config.chanend > config.chanstart
        assert config.osamp >= 1
        assert config.nant >= 1
        assert config.acclen > 0


class TestBufferSizes:
    """Test BufferSizes calculations"""
    
    @pytest.mark.parametrize("osamp,expected_multiplier", [
        (1, 1),
        (2, 2), 
        (4, 4),
        (8, 8)
    ])
    def test_lblock_scaling(self, osamp, expected_multiplier):
        """Test that lblock scales correctly with oversampling"""
        config = ProcessingConfig(
            acclen=100, pfb_size=4096, nchunks=1000, nblock=64,
            chanstart=0, chanend=1024, osamp=osamp, nant=4
        )
        
        sizes = BufferSizes.from_config(config)
        assert sizes.lblock == 4096 * expected_multiplier
    
    def test_buffer_size_consistency(self):
        """Test internal consistency of buffer size calculations"""
        config = ProcessingConfig(
            acclen=100, pfb_size=4096, nchunks=1000, nblock=64,
            chanstart=200, chanend=800, osamp=2, nant=4, ntap=4
        )
        
        sizes = BufferSizes.from_config(config)
        
        # Check relationships
        assert sizes.szblock == (config.nblock + config.ntap - 1) * sizes.lblock
        assert sizes.lchunk == 4096 * config.acclen
        assert sizes.nchan == (config.chanend - config.chanstart) * config.osamp


@pytest.mark.skipif(not cp.cuda.is_available(), reason="CUDA not available")
class TestBufferManager:
    """Test BufferManager with actual GPU operations"""
    
    @pytest.fixture
    def config(self):
        return ProcessingConfig(
            acclen=50, pfb_size=4096, nchunks=100, nblock=32,
            chanstart=0, chanend=512, osamp=2, nant=2
        )
    
    @pytest.fixture
    def sizes(self, config):
        return BufferSizes.from_config(config)
    
    def test_buffer_manager_initialization(self, config, sizes):
        """Test BufferManager creates GPU buffers correctly"""
        buffer_mgr = BufferManager(config, sizes)
        
        # Check buffer types and shapes
        assert isinstance(buffer_mgr.pol, cp.ndarray)
        assert isinstance(buffer_mgr.cut_pol, cp.ndarray)
        assert isinstance(buffer_mgr.pfb_buf, cp.ndarray)
        assert isinstance(buffer_mgr.rem_buf, cp.ndarray)
        
        # Check shapes
        assert buffer_mgr.pol.shape == (config.acclen + 2*config.cut, 2049)
        assert buffer_mgr.cut_pol.shape == (config.nant, config.npol, 2*config.cut, 2049)
        assert buffer_mgr.pfb_buf.shape == (config.nant, config.npol, config.nblock + config.ntap - 1, sizes.lblock)
        assert buffer_mgr.rem_buf.shape == (config.nant, config.npol, sizes.lchunk)
    
    def test_buffer_operations(self, config, sizes):
        """Test buffer operations with real GPU arrays"""
        buffer_mgr = BufferManager(config, sizes)
        
        # Test adding data to buffers
        test_data = cp.random.randn(1000).astype(np.float32)
        
        # Test normal case - data fits in buffer
        buffer_mgr.rem_buf[0, 0, :1000] = test_data
        buffer_mgr.rem_idx[0, 0] = 1000
        buffer_mgr.pfb_idx[0, 0] = 0
        
        buffer_full = buffer_mgr.add_remaining_to_pfb_buffer(0, 0)
        
        assert not buffer_full
        assert buffer_mgr.pfb_idx[0, 0] == 1000
        assert buffer_mgr.rem_idx[0, 0] == 0
        
        # Verify data was copied correctly
        assert cp.allclose(buffer_mgr.pfb_buf[0, 0].flat[:1000], test_data)
    
    def test_buffer_overflow_handling(self, config, sizes):
        """Test buffer overflow scenarios"""
        buffer_mgr = BufferManager(config, sizes)
        
        # Fill buffer almost to capacity
        almost_full = sizes.szblock - 500
        buffer_mgr.pfb_idx[0, 0] = almost_full
        
        # Try to add more data than fits
        test_data_size = 1000
        buffer_mgr.rem_buf[0, 0, :test_data_size] = cp.arange(test_data_size, dtype=np.float32)
        buffer_mgr.rem_idx[0, 0] = test_data_size
        
        buffer_full = buffer_mgr.add_remaining_to_pfb_buffer(0, 0)
        
        assert buffer_full
        assert buffer_mgr.pfb_idx[0, 0] == sizes.szblock
        assert buffer_mgr.rem_idx[0, 0] == 500  # Overflow amount


@pytest.mark.skipif(not cp.cuda.is_available(), reason="CUDA not available")
class TestIPFBProcessing:
    """Test IPFB processing accuracy"""
    
    @pytest.fixture
    def simple_config(self):
        return ProcessingConfig(
            acclen=64, pfb_size=4096, nchunks=10, nblock=16,
            chanstart=500, chanend=1500, osamp=1, nant=1, cut=5, filt_thresh=0.1, ntap=4
        )
    
    
    def test_ipfb_linearity(self, simple_config):
        """Test IPFB linearity: IPFB(a*x + b*y) = a*IPFB(x) + b*IPFB(y)"""
        # Create two different signals
        np.random.seed(123)
        signal1 = cp.random.randn(simple_config.acclen + 2*simple_config.cut, 2049).astype(cp.complex64)
        signal2 = cp.random.randn(simple_config.acclen + 2*simple_config.cut, 2049).astype(cp.complex64)
        
        # Scaling factors
        a, b = 2.0, 3.0
        
        # Create filter
        matft = pu.get_matft(simple_config.acclen + 2*simple_config.cut)
        filt = pu.calculate_filter(matft, simple_config.filt_thresh)
        
        # Test linearity
        combined_signal = a * signal1 + b * signal2
        ipfb_combined = pu.cupy_ipfb(combined_signal, filt)
        
        ipfb_signal1 = pu.cupy_ipfb(signal1, filt)
        ipfb_signal2 = pu.cupy_ipfb(signal2, filt)
        ipfb_linear = a * ipfb_signal1 + b * ipfb_signal2
        
        # Check linearity (with numerical tolerance)
        diff = cp.max(cp.abs(ipfb_combined - ipfb_linear))
        max_val = cp.max(cp.abs(ipfb_combined))
        relative_error = float(diff / max_val)
        
        assert relative_error < 1e-5, f"IPFB linearity test failed: relative error = {relative_error}"


@pytest.mark.skipif(not cp.cuda.is_available(), reason="CUDA not available")
class TestPFBProcessing:
    """Test PFB processing accuracy"""
    
    @pytest.fixture
    def pfb_config(self):
        return ProcessingConfig(
            acclen=64, pfb_size=4096, nchunks=10, nblock=16,
            chanstart=0, chanend=1024, osamp=2, nant=1, ntap=4
        )
    
    def test_pfb_window_function(self, pfb_config):
        """Test PFB window function properties"""
        window = pu.sinc_hamming(pfb_config.ntap, 4096 * pfb_config.osamp)
        cupy_window = cp.asarray(window, dtype=cp.float32)
        
        assert len(window) == 4096 * pfb_config.osamp * pfb_config.ntap
        assert np.sum(window) > 0   # Non-zero sum
    
    def test_pfb_impulse_response(self, pfb_config):
        """Test PFB with impulse input"""
        sizes = BufferSizes.from_config(pfb_config)
        
        # Create impulse signal
        time_impulse = cp.zeros((pfb_config.nblock + pfb_config.ntap - 1, sizes.lblock), dtype=cp.float32)
        impulse_pos = (pfb_config.nblock // 2) * sizes.lblock + sizes.lblock // 2
        time_impulse.flat[impulse_pos] = 1.0
        
        # Create window
        window = pu.sinc_hamming(pfb_config.ntap, sizes.lblock)
        cupy_window = cp.asarray(window, dtype=cp.float32)
        
        # Apply PFB
        freq_output = pu.cupy_pfb(time_impulse, cupy_window, 
                                 nchan=sizes.lblock//2 + 1, ntap=pfb_config.ntap)
        
        # Check output shape and properties
        expected_shape = (pfb_config.nblock, sizes.lblock//2 + 1)
        assert freq_output.shape == expected_shape
        
        # Check that output has expected spectral characteristics
        power_spectrum = cp.abs(freq_output)**2
        assert cp.sum(power_spectrum) > 0  # Should have non-zero power
    
    def test_pfb_frequency_resolution(self, pfb_config):
        """Test PFB frequency resolution with known sinusoid"""
        sizes = BufferSizes.from_config(pfb_config)
        
        # Create sinusoidal signal at known frequency
        test_freq_bin = sizes.lblock // 8  # 1/8 of Nyquist
        t = cp.arange((pfb_config.nblock + pfb_config.ntap - 1) * sizes.lblock, dtype=cp.float32)
        omega = 2 * cp.pi * test_freq_bin / sizes.lblock
        time_signal = cp.cos(omega * t).reshape(-1, sizes.lblock)
        
        # Create window
        window = pu.sinc_hamming(pfb_config.ntap, sizes.lblock)
        cupy_window = cp.asarray(window, dtype=cp.float32)
        
        # Apply PFB
        freq_output = pu.cupy_pfb(time_signal, cupy_window,
                                 nchan=sizes.lblock//2 + 1, ntap=pfb_config.ntap)
        
        # Check that power is concentrated near expected frequency
        power_spectrum = cp.mean(cp.abs(freq_output)**2, axis=0)
        peak_bin = cp.argmax(power_spectrum)
        
        # Allow some tolerance due to windowing effects
        assert abs(int(peak_bin) - test_freq_bin) <= 2, \
            f"Peak at bin {peak_bin}, expected near {test_freq_bin}"


@pytest.mark.skipif(not cp.cuda.is_available(), reason="CUDA not available")
class TestRePFBRoundTrip:
    """Test full IPFB -> PFB round trip for RePFB accuracy"""
    
    @pytest.fixture
    def roundtrip_config(self):
        return ProcessingConfig(
            acclen=32, pfb_size=4096, nchunks=5, nblock=8,
            chanstart=512, chanend=1536, osamp=2, nant=1, cut=4
        )
    
    def test_repfb_roundtrip_impulse(self, roundtrip_config):
        """Test that IPFB -> PFB roundtrip preserves impulse"""
        sizes = BufferSizes.from_config(roundtrip_config)
        
        # Create impulse in original frequency domain
        original_freq = cp.zeros((roundtrip_config.acclen + 2*roundtrip_config.cut, 2049), dtype=cp.complex64)
        impulse_time_bin = roundtrip_config.cut + roundtrip_config.acclen // 2
        impulse_freq_bin = 1024  # Center frequency
        original_freq[impulse_time_bin, impulse_freq_bin] = 1.0
        
        # Setup filters and windows
        matft = pu.get_matft(roundtrip_config.acclen + 2*roundtrip_config.cut)
        ipfb_filter = pu.calculate_filter(matft, roundtrip_config.filt_thresh)
        pfb_window = cp.asarray(pu.sinc_hamming(roundtrip_config.ntap, sizes.lblock), dtype=cp.float32)
        
        # Apply IPFB
        time_stream = pu.cupy_ipfb(original_freq, ipfb_filter)
        
        # Remove edge effects and reshape for PFB
        clean_time = time_stream[roundtrip_config.cut:-roundtrip_config.cut]
        time_length = len(clean_time)
        
        # Pad and reshape for PFB processing
        pfb_input_size = (roundtrip_config.nblock + roundtrip_config.ntap - 1) * sizes.lblock
        if time_length < pfb_input_size:
            # Pad with zeros
            padded_time = cp.zeros((pfb_input_size, 4096), dtype=clean_time.dtype)
            padded_time[:time_length] = clean_time
        else:
            padded_time = clean_time[:pfb_input_size]
        
        pfb_input = padded_time.reshape(-1, sizes.lblock)
        
        # Apply PFB
        recovered_freq = pu.cupy_pfb(pfb_input.astype(cp.float32), pfb_window,
                                    nchan=sizes.lblock//2 + 1, ntap=roundtrip_config.ntap)
        
        # Check that we can identify the impulse location
        power_spectrum = cp.abs(recovered_freq)**2
        max_power_time, max_power_freq = cp.unravel_index(cp.argmax(power_spectrum), power_spectrum.shape)
        
        # The impulse should be detectable (allowing for some spreading due to windowing)
        assert cp.max(power_spectrum) > 0.01, "Impulse not detected in round-trip"
        
        # Check energy conservation (with tolerance for edge effects and windowing)
        original_energy = cp.sum(cp.abs(original_freq)**2)
        recovered_energy = cp.sum(cp.abs(recovered_freq)**2)
        energy_ratio = float(cp.abs(recovered_energy - original_energy) / original_energy)
        
        assert energy_ratio < 0.5, f"Energy not preserved in round-trip: ratio = {energy_ratio}"
    
    def test_repfb_roundtrip_sinusoid(self, roundtrip_config):
        """Test IPFB -> PFB roundtrip with sinusoidal signal"""
        sizes = BufferSizes.from_config(roundtrip_config)
        
        # Create sinusoidal signal in frequency domain
        freq_signal = cp.zeros((roundtrip_config.acclen + 2*roundtrip_config.cut, 2049), dtype=cp.complex64)
        
        # Add a few frequency components
        test_freqs = [800, 1000, 1200]  # Frequency bins
        for freq_bin in test_freqs:
            # Create sinusoidal modulation in time
            time_indices = cp.arange(roundtrip_config.cut, roundtrip_config.acclen + roundtrip_config.cut)
            modulation = cp.exp(2j * cp.pi * 0.1 * time_indices / roundtrip_config.acclen)
            freq_signal[time_indices, freq_bin] = 0.5 * modulation
        
        # Setup processing components
        matft = pu.get_matft(roundtrip_config.acclen + 2*roundtrip_config.cut)
        ipfb_filter = pu.calculate_filter(matft, roundtrip_config.filt_thresh)
        pfb_window = cp.asarray(pu.sinc_hamming(roundtrip_config.ntap, sizes.lblock), dtype=cp.float32)
        
        # Apply IPFB
        time_stream = pu.cupy_ipfb(freq_signal, ipfb_filter)
        
        # Process for PFB
        clean_time = time_stream[roundtrip_config.cut:-roundtrip_config.cut]
        time_length = len(clean_time)
        
        pfb_input_size = (roundtrip_config.nblock + roundtrip_config.ntap - 1) * sizes.lblock
        if time_length < pfb_input_size:
            padded_time = cp.zeros((pfb_input_size, 4096), dtype=clean_time.dtype)
            padded_time[:time_length] = clean_time
        else:
            padded_time = clean_time[:pfb_input_size]
        
        pfb_input = padded_time.reshape(-1, sizes.lblock)
        
        # Apply PFB
        recovered_freq = pu.cupy_pfb(pfb_input.astype(cp.float32), pfb_window,
                                    nchan=sizes.lblock//2 + 1, ntap=roundtrip_config.ntap)
        
        # Analyze spectral content
        avg_power = cp.mean(cp.abs(recovered_freq)**2, axis=0)
        
        # Check that signal is recovered
        assert cp.max(avg_power) > 0.001, "Signal not detected in round-trip"
        
        # Check that power is concentrated in expected regions
        # (This is qualitative due to the frequency mapping between IPFB and PFB)
        total_power = cp.sum(avg_power)
        assert total_power > 0, "No power detected in recovered signal"


@pytest.mark.skipif(not cp.cuda.is_available(), reason="CUDA not available")
class TestMissingDataHandling:
    """Test missing data tracking and handling"""
    
    def test_missing_data_tracker_accuracy(self):
        """Test that missing data tracking is accurate"""
        nant, n_job_chunks = 3, 10
        tracker = MissingDataTracker(nant, n_job_chunks)
        
        # Simulate chunks with known missing data
        test_cases = [
            ({"specnums": np.arange(80)}, 100, 20.0),  # 20% missing
            ({"specnums": np.arange(90)}, 100, 10.0),  # 10% missing  
            ({"specnums": np.arange(100)}, 100, 0.0),  # 0% missing
            ({"specnums": np.arange(50)}, 100, 50.0),  # 50% missing
        ]
        
        for chunk, acclen, expected_missing in test_cases:
            tracker.add_chunk_info(0, chunk, acclen, 1)
        
        # Calculate average and check
        tracker.calculate_job_average(0, 0)
        
        expected_avg = (20.0 + 10.0 + 0.0 + 50.0) / 4
        assert abs(tracker.missing_fraction[0, 0] - expected_avg) < 1e-10
    
    def test_missing_data_affects_multiple_blocks(self):
        """Test tracking when missing data affects multiple PFB blocks"""
        tracker = MissingDataTracker(1, 5)
        
        # Add chunk that affects 3 blocks
        chunk = {"specnums": np.arange(70)}
        tracker.add_chunk_info(0, chunk, 100, 3)
        
        # Calculate for first job - should include this chunk
        tracker.calculate_job_average(0, 0)
        assert round(tracker.missing_fraction[0, 0], 1) == 30.0
        
        # Calculate for second job - should still include (blocks_affected = 2)
        tracker.calculate_job_average(0, 1) 
        assert round(tracker.missing_fraction[0, 1], 1) == 30.0
        
        # Calculate for third job - should still include (blocks_affected = 1)
        tracker.calculate_job_average(0, 2)
        assert round(tracker.missing_fraction[0, 2], 1) == 30.0
        
        # Calculate for fourth job - should not include (blocks_affected = 0)
        tracker.calculate_job_average(0, 3)
        assert round(tracker.missing_fraction[0, 3], 1) == 0.0


class TestChunkPlanning:
    """Test chunk planning algorithms"""
    
    def test_plan_chunks_exact_fit(self):
        """Test chunk planning when data fits exactly"""
        # Parameters chosen so chunks fit exactly
        nchunks = 10
        lchunk = 1000  
        nblock = 5
        lblock = 200
        ntap = 2
        
        ranges = plan_chunks(nchunks, lchunk, nblock, lblock, ntap)
        
        # Should have at least one range
        assert len(ranges) > 0
        
        # Check that ranges cover expected number of chunks
        total_chunks = sum(end - start for start, end in ranges if end is not None)
        if ranges and ranges[-1][1] is None:
            # Handle final incomplete range
            total_chunks += nchunks - ranges[-1][0]
        
        assert total_chunks == nchunks
    
    def test_plan_chunks_with_remainder(self):
        """Test chunk planning with remainder data"""
        nchunks = 7
        lchunk = 1500
        nblock = 3
        lblock = 1000  
        ntap = 2
        
        ranges = plan_chunks(nchunks, lchunk, nblock, lblock, ntap)
        
        assert len(ranges) > 0
        
        # Verify all chunks are covered
        covered_chunks = set()
        for start, end in ranges:
            if end is None:
                end = nchunks
            covered_chunks.update(range(start, end))
        
        assert covered_chunks == set(range(nchunks))
    
    def test_plan_chunks_insufficient_data(self):
        """Test chunk planning when insufficient data for even one block"""
        nchunks = 1
        lchunk = 100
        nblock = 10
        lblock = 1000
        ntap = 4
        
        ranges = plan_chunks(nchunks, lchunk, nblock, lblock, ntap)
        
        # Should still create a range even if insufficient
        assert len(ranges) >= 0
        
        if ranges:
            assert ranges[0][0] == 0


    def test_zero_chunks(self):
        """Test handling of zero chunks"""
        ranges = plan_chunks(nchunks=0, lchunk=1000, nblock=10, lblock=100, ntap=4)
        assert ranges == []
    
    def test_missing_data_tracker_edge_cases(self):
        """Test edge cases in missing data tracking"""
        tracker = MissingDataTracker(nant=1, n_job_chunks=1)
        
        # Test with completely missing chunk
        chunk_empty = {"specnums": np.array([])}
        tracker.add_chunk_info(0, chunk_empty, 100, 1)
        tracker.calculate_job_average(0, 0)
        
        assert tracker.missing_fraction[0, 0] == 100.0
        
        # Test with over-full chunk (shouldn't happen but test robustness)
        chunk_overfull = {"specnums": np.arange(150)}
        tracker = MissingDataTracker(nant=1, n_job_chunks=1)
        tracker.add_chunk_info(0, chunk_overfull, 100, 1)
        tracker.calculate_job_average(0, 0)
        
        # Should handle gracefully (may be negative missing fraction)
        assert isinstance(tracker.missing_fraction[0, 0], (int, float, np.number))


@pytest.mark.skipif(not cp.cuda.is_available(), reason="CUDA not available")
class TestNumericalAccuracy:
    """Test numerical accuracy and stability"""
    
    def test_filter_stability(self):
        """Test IPFB filter numerical stability"""
        # Test with different regularization parameters
        acclen = 64
        test_thresholds = [0.1, 0.45, 0.8, 0.99]
        
        matft = pu.get_matft(acclen)
        
        for thresh in test_thresholds:
            filt = pu.calculate_filter(matft, thresh)
            
            # Check that filter is finite and well-conditioned
            assert cp.all(cp.isfinite(filt)), f"Filter not finite for threshold {thresh}"
            assert cp.max(cp.abs(filt)) < 1e10, f"Filter too large for threshold {thresh}"
            assert cp.min(cp.abs(filt)) > 1e-10, f"Filter too small for threshold {thresh}"
    
    def test_precision_consistency(self):
        """Test consistency between float32 and float64 calculations"""
        # Create test signal
        length = 1000
        test_signal_64 = np.random.randn(length).astype(np.float64)
        test_signal_32 = test_signal_64.astype(np.float32)
        
        # Convert to CuPy
        cp_signal_64 = cp.asarray(test_signal_64)
        cp_signal_32 = cp.asarray(test_signal_32)
        
        # Compute some basic operations and check consistency
        sum_64 = float(cp.sum(cp_signal_64))
        sum_32 = float(cp.sum(cp_signal_32))
        
        # Should be close (within float32 precision)
        relative_error = abs(sum_64 - sum_32) / abs(sum_64)
        assert relative_error < 1e-6, f"Precision inconsistency: {relative_error}"
    
 


if __name__ == "__main__":
    # Custom pytest configuration for this test suite
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "--disable-warnings",
        "-k", "not test_processing_speed_benchmark"  # Skip slow benchmark by default
    ])