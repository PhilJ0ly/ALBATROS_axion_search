import struct
import numpy as np
import time
import os
from datetime import datetime, timezone

class BasebandTestFileGenerator:
    def __init__(self, 
                 unix_timestamp,
                 output_folder="test_data",
                 length_channels=120,     # Actual: 120 channels
                 spectra_per_packet=5,    # Actual: 5 spectra per packet
                 bit_mode=4,              # Actual: 4-bit
                 have_trimble=1,
                 channels=None,           # Actual: channels 64-183
                 sample_rate=250e6,       # 250 Msamples/s
                 bandwidth=125e6,         # Full bandwidth 0-125 MHz
                 channel_start=64,        # Actual: starts at channel 64
                 channel_end=183,         # Actual: ends at channel 183
                 file_duration=34):       # 34 seconds per file
        """
        Generate test baseband files matching the actual data format.
        
        The actual data covers channels 64-183 (120 channels total),
        which corresponds to approximately 4-11 MHz frequency range.
        """
        self.unix_timestamp = unix_timestamp
        self.output_folder = output_folder
        self.length_channels = length_channels
        self.spectra_per_packet = spectra_per_packet
        self.bit_mode = bit_mode
        self.have_trimble = have_trimble
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.file_duration = file_duration
        self.channel_start = channel_start
        self.channel_end = channel_end
        
        # Calculate frequency parameters
        self.freq_resolution = bandwidth / 2048  # Full resolution ~61 kHz
        self.freq_start = channel_start * self.freq_resolution  # ~4 MHz
        self.freq_end = (channel_end + 1) * self.freq_resolution  # ~11.5 MHz
        self.actual_bandwidth = self.freq_end - self.freq_start  # ~7.5 MHz
        
        # Generate actual channels array (64 to 183)
        if channels is None:
            self.channels = np.arange(channel_start, channel_end + 1, dtype=np.uint64)
        else:
            self.channels = np.array(channels, dtype=np.uint64)
        
        # Verify we have the right number of channels
        assert len(self.channels) == length_channels, f"Expected {length_channels} channels, got {len(self.channels)}"
        
        # Calculate packet parameters to match actual data
        # Actual: 1204 bytes per packet = 4 bytes (spec_num) + 1200 bytes (data)
        # 1200 bytes = 2400 samples (4-bit, 2 samples per byte)
        # 2400 samples / 120 channels / 5 spectra = 4 samples per channel per spectrum
        # This suggests 2 polarizations × 2 complex components = 4 samples per channel
        self.bytes_per_packet = 1204  # Match actual data exactly
        self.data_bytes_per_packet = 1200  # 1204 - 4 (for spec_num)
        
        # Calculate packets per file based on actual data
        # Actual file has 415282 packets for 34 seconds  
        self.packets_per_second = 415282 / 34  # ~12214 packets/sec
        self.num_packets = int(self.packets_per_second * file_duration)
        
        # Calculate spectrum rate for realistic time evolution
        self.spectrum_rate = self.packets_per_second * self.spectra_per_packet  # ~61k spectra/sec
        
        # GPS coordinates from actual data (Arctic location)
        self.gps_latitude = 79.41560919999999
        self.gps_longitude = -90.7723716
        self.gps_elevation = 179.077
        
        print(f"Frequency range: {self.freq_start/1e6:.2f} - {self.freq_end/1e6:.2f} MHz")
        print(f"Frequency resolution: {self.freq_resolution/1000:.1f} kHz per channel")
        print(f"Actual bandwidth: {self.actual_bandwidth/1e6:.2f} MHz")
        print(f"Channels: {self.channel_start} to {self.channel_end} ({len(self.channels)} channels)")
        print(f"Packets per file: {self.num_packets}")

    def get_frequency_array(self):
        """Get frequency array in Hz for each channel."""
        return self.channels * self.freq_resolution

    def generate_realistic_signals(self, packet_idx, spectrum_idx):
        """
        Generate realistic signal data in the 4-11 MHz range.
        
        Returns data for 120 channels covering ~4-11 MHz.
        Each channel gets complex data (real + imaginary components).
        """
        # Time for this spectrum
        time_idx = packet_idx * self.spectra_per_packet + spectrum_idx
        abs_time = time_idx / self.spectrum_rate  # Absolute time in seconds
        
        # Get frequency array for our channels
        freqs = self.get_frequency_array()
        
        # Define test signals in the 4-11 MHz range
        test_signals = [
            {"freq_mhz": 4.5, "amp": 60, "phase": 0},           # Low end
            {"freq_mhz": 5.2, "amp": 45, "phase": np.pi/4},     # 
            {"freq_mhz": 6.8, "amp": 40, "phase": np.pi/2},     # Middle
            {"freq_mhz": 7.3, "amp": 55, "phase": 3*np.pi/4},   #
            {"freq_mhz": 8.9, "amp": 35, "phase": np.pi},       #
            {"freq_mhz": 10.2, "amp": 50, "phase": 0},          # High end
            {"freq_mhz": 10.8, "amp": 30, "phase": np.pi/6},    # Very high end
        ]
        
        # Add slow time-varying modulation
        modulation_freq = 0.05  # Hz, very slow modulation
        modulation = 1.0 + 0.3 * np.sin(2 * np.pi * modulation_freq * abs_time)
        
        # Add faster modulation for some dynamics
        fast_mod_freq = 2.0  # Hz
        fast_modulation = 1.0 + 0.1 * np.sin(2 * np.pi * fast_mod_freq * abs_time)
        
        # Initialize output arrays for complex data
        pol0_real = np.zeros(self.length_channels, dtype=np.float32)
        pol0_imag = np.zeros(self.length_channels, dtype=np.float32)
        pol1_real = np.zeros(self.length_channels, dtype=np.float32)
        pol1_imag = np.zeros(self.length_channels, dtype=np.float32)
        
        # Add noise floor
        noise_level = 8
        np.random.seed(int((time_idx * 7919) % 2**31))  # Reproducible noise
        noise0_real = np.random.normal(0, noise_level, self.length_channels)
        noise0_imag = np.random.normal(0, noise_level, self.length_channels)
        noise1_real = np.random.normal(0, noise_level, self.length_channels)
        noise1_imag = np.random.normal(0, noise_level, self.length_channels)
        
        # Add each test signal
        for sig in test_signals:
            freq_hz = sig["freq_mhz"] * 1e6
            
            # Find closest frequency bin in our channel range
            channel_idx = int(freq_hz / self.freq_resolution)
            
            # Check if this frequency falls in our channel range
            if self.channel_start <= channel_idx <= self.channel_end:
                # Convert to index in our data array
                data_idx = channel_idx - self.channel_start
                
                if 0 <= data_idx < self.length_channels:
                    # Apply modulations and time-varying phase
                    phase_drift = 0.02 * abs_time  # Slow phase drift
                    total_phase = sig["phase"] + phase_drift
                    amplitude = sig["amp"] * modulation * fast_modulation
                    
                    # Add complex signal: amplitude * e^(i*phase) = amp*cos(phase) + i*amp*sin(phase)
                    pol0_real[data_idx] += amplitude * np.cos(total_phase)
                    pol0_imag[data_idx] += amplitude * np.sin(total_phase)
                    
                    # Add correlated signal to pol1 (with some decorrelation)
                    correlation = 0.75
                    decorr_phase = total_phase + np.pi/3
                    pol1_real[data_idx] += amplitude * correlation * np.cos(decorr_phase)
                    pol1_imag[data_idx] += amplitude * correlation * np.sin(decorr_phase)
                    
                    # Add uncorrelated component
                    uncorr_amp = amplitude * (1 - correlation)
                    uncorr_phase = total_phase + np.pi
                    pol1_real[data_idx] += uncorr_amp * np.cos(uncorr_phase)
                    pol1_imag[data_idx] += uncorr_amp * np.sin(uncorr_phase)
        
        # Add broadband noise component that varies across frequency
        freq_dependent_noise = np.sin(np.arange(self.length_channels) * 0.1) * 5
        pol0_real += freq_dependent_noise
        pol0_imag += freq_dependent_noise * 0.7
        pol1_real += freq_dependent_noise * 0.8
        pol1_imag += freq_dependent_noise * 0.9
        
        # Add random noise
        pol0_real += noise0_real
        pol0_imag += noise0_imag
        pol1_real += noise1_real
        pol1_imag += noise1_imag
        
        # Occasionally add RFI in this band
        if np.random.random() < 0.03:  # 3% chance of RFI
            rfi_freq = np.random.uniform(4.2e6, 10.8e6)  # RFI in our band
            rfi_channel = int(rfi_freq / self.freq_resolution)
            
            if self.channel_start <= rfi_channel <= self.channel_end:
                rfi_idx = rfi_channel - self.channel_start
                rfi_width = np.random.randint(1, 5)  # 1-5 channels wide
                rfi_amp = np.random.uniform(80, 150)  # Strong RFI
                
                start_idx = max(0, rfi_idx - rfi_width//2)
                end_idx = min(self.length_channels, rfi_idx + rfi_width//2 + 1)
                
                pol0_real[start_idx:end_idx] += rfi_amp
                pol0_imag[start_idx:end_idx] += rfi_amp * 0.8
                pol1_real[start_idx:end_idx] += rfi_amp * 0.95
                pol1_imag[start_idx:end_idx] += rfi_amp * 0.9
        
        # Convert to unsigned 8-bit range for processing
        offset = 128
        scale = 1.8
        
        pol0_real_u8 = np.clip(pol0_real * scale + offset, 0, 255).astype(np.uint8)
        pol0_imag_u8 = np.clip(pol0_imag * scale + offset, 0, 255).astype(np.uint8)
        pol1_real_u8 = np.clip(pol1_real * scale + offset, 0, 255).astype(np.uint8)
        pol1_imag_u8 = np.clip(pol1_imag * scale + offset, 0, 255).astype(np.uint8)
        
        return (pol0_real_u8, pol0_imag_u8, pol1_real_u8, pol1_imag_u8)

    def pack_data_4bit_actual_format_complex(self, pol0_real, pol0_imag, pol1_real, pol1_imag):
        """
        Pack complex data to match actual format based on unpacking code analysis.
        
        From the C code unpack_4bit_float:
        - Each channel has 2 bytes: pol0_byte, pol1_byte  
        - Each byte has: upper 4 bits = real, lower 4 bits = imag
        - Data layout: [pol0_ch0][pol1_ch0][pol0_ch1][pol1_ch1]...
        - Values are signed 4-bit (-8 to +7)
        
        For 120 channels × 5 spectra = 600 channel-spectra per packet
        600 × 2 bytes = 1200 bytes per packet ✓
        """
        # Convert float data to signed 4-bit values (-8 to +7)
        # First convert to signed range around 0
        pol0_real_centered = pol0_real.astype(np.int16) - 128  # -128 to +127
        pol0_imag_centered = pol0_imag.astype(np.int16) - 128
        pol1_real_centered = pol1_real.astype(np.int16) - 128
        pol1_imag_centered = pol1_imag.astype(np.int16) - 128
        
        # Scale to 4-bit signed range (-8 to +7)
        pol0_real_4bit = np.clip(pol0_real_centered // 8, -8, 7).astype(np.int8)
        pol0_imag_4bit = np.clip(pol0_imag_centered // 8, -8, 7).astype(np.int8)
        pol1_real_4bit = np.clip(pol1_real_centered // 8, -8, 7).astype(np.int8)
        pol1_imag_4bit = np.clip(pol1_imag_centered // 8, -8, 7).astype(np.int8)
        
        # Convert signed to unsigned 4-bit for packing (0-15 range)
        # This matches what the unpacking code expects: if (value > 8) value -= 16
        pol0_real_u4 = ((pol0_real_4bit + 16) % 16).astype(np.uint8)
        pol0_imag_u4 = ((pol0_imag_4bit + 16) % 16).astype(np.uint8)
        pol1_real_u4 = ((pol1_real_4bit + 16) % 16).astype(np.uint8)
        pol1_imag_u4 = ((pol1_imag_4bit + 16) % 16).astype(np.uint8)
        
        # Pack each polarization: [real nibble][imag nibble] in each byte
        pol0_bytes = (pol0_real_u4 << 4) | pol0_imag_u4  # Upper 4 bits real, lower 4 bits imag
        pol1_bytes = (pol1_real_u4 << 4) | pol1_imag_u4
        
        # Interleave pol0 and pol1 bytes: pol0_ch0, pol1_ch0, pol0_ch1, pol1_ch1, ...
        # This matches the unpacking: data[(i+rowstart)*c1 + 2*k] for pol0, data[...+2*k+1] for pol1
        packed_bytes = np.zeros(self.length_channels * 2, dtype=np.uint8)
        packed_bytes[0::2] = pol0_bytes  # Even indices: pol0
        packed_bytes[1::2] = pol1_bytes  # Odd indices: pol1
        
        return packed_bytes

    def write_header(self, file_obj):
        """Write file header matching actual format."""
        # Header bytes = 2008 total
        # This includes: 8 fields × 8 bytes + channels array + GPS data
        channels_bytes = len(self.channels) * 8  # 120 channels × 8 bytes = 960 bytes
        header_size_field = 2008 - 8  # Subtract the 8 bytes for this field itself
        
        # Write header size
        file_obj.write(struct.pack(">Q", header_size_field))
        
        # Write packet parameters (match actual values)
        file_obj.write(struct.pack(">Q", self.bytes_per_packet))  # 1204
        file_obj.write(struct.pack(">Q", self.length_channels))   # 120
        file_obj.write(struct.pack(">Q", self.spectra_per_packet)) # 5
        file_obj.write(struct.pack(">Q", self.bit_mode))          # 4
        file_obj.write(struct.pack(">Q", self.have_trimble))      # 1
        
        # Write channels array (64 to 183)
        for channel in self.channels:
            file_obj.write(struct.pack(">Q", channel))
        
        # GPS data (actual coordinates from real data)
        file_obj.write(struct.pack(">Q", 0))  # GPS week = 0 (as in actual data)
        file_obj.write(struct.pack(">Q", self.unix_timestamp))  # GPS timestamp
        file_obj.write(struct.pack(">d", self.gps_latitude))    # Arctic latitude
        file_obj.write(struct.pack(">d", self.gps_longitude))   # Arctic longitude
        file_obj.write(struct.pack(">d", self.gps_elevation))   # Arctic elevation

    def generate_file(self, timestamp):
        """Generate a single test file for given timestamp."""
        # Create folder structure (e.g., "17214" from timestamp)
        folder_prefix = str(timestamp)[:5]
        full_folder = os.path.join(self.output_folder, folder_prefix)
        os.makedirs(full_folder, exist_ok=True)
        
        filename = f"{timestamp}.raw"
        filepath = os.path.join(full_folder, filename)
        
        print(f"Generating {filepath}...")
        print(f"  Packets: {self.num_packets}, Spectra per packet: {self.spectra_per_packet}")
        print(f"  Bytes per packet: {self.bytes_per_packet}")
        
        with open(filepath, 'wb') as f:
            # Write header (2008 bytes total)
            self.write_header(f)
            
            # Calculate starting spectrum number
            start_spec_num = int((timestamp % 100000) * 100)
            
            # Write packet data
            for packet_idx in range(self.num_packets):
                spec_num = start_spec_num + packet_idx * self.spectra_per_packet
                
                # Generate data for all spectra in this packet
                packet_data = []
                
                for spectrum_idx in range(self.spectra_per_packet):
                    pol0_real, pol0_imag, pol1_real, pol1_imag = self.generate_realistic_signals(packet_idx, spectrum_idx)
                    packed = self.pack_data_4bit_actual_format_complex(pol0_real, pol0_imag, pol1_real, pol1_imag)
                    packet_data.append(packed)
                
                # Combine all spectra data (should be 1200 bytes total)
                spectra_data = np.concatenate(packet_data)
                
                # Verify size matches expected format
                expected_data_size = self.data_bytes_per_packet
                if len(spectra_data) != expected_data_size:
                    print(f"Warning: data size {len(spectra_data)} != expected {expected_data_size}")
                
                # Write packet: spec_num (4 bytes) + spectra data (1200 bytes)
                f.write(struct.pack(">I", spec_num))
                f.write(spectra_data.tobytes())
                
                if packet_idx % 1000 == 0:
                    print(f"    Generated {packet_idx+1}/{self.num_packets} packets")
        
        print(f"Successfully generated {filepath}")
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")
        return filepath

def generate_actual_format_test_files(start_timestamp=1721400005, 
                                     num_files=3,
                                     output_folder="test_baseband_data"):
    """
    Generate test files matching the actual data format exactly.
    
    Based on real header:
    - 120 channels (64-183)
    - ~4-11 MHz frequency range  
    - 5 spectra per packet
    - 1204 bytes per packet
    - 4-bit mode
    """
    print(f"Generating {num_files} test files matching ACTUAL data format...")
    print("Real specs: 120 channels (64-183), ~4-11 MHz, 5 spectra/packet, 1204 bytes/packet")
    
    generator = BasebandTestFileGenerator(
        start_timestamp, 
        output_folder=output_folder,
        file_duration=34
    )
    
    generated_files = []
    current_timestamp = start_timestamp
    
    for i in range(num_files):
        filepath = generator.generate_file(current_timestamp)
        generated_files.append(filepath)
        
        # Next file is 34 seconds later
        current_timestamp += 34
        print(f"Completed {i+1}/{num_files} files")
    
    print(f"\nGenerated {num_files} test files:")
    for filepath in generated_files:
        print(f"  {filepath}")
    
    return generated_files

if __name__ == "__main__":
    print("Generating test files matching ACTUAL baseband data format...")
    print("Channels 64-183 (120 total), ~4-11 MHz, 5 spectra/packet, 1204 bytes/packet")
    
    # Generate test files with actual format
    files = generate_actual_format_test_files(
        start_timestamp=1721400005,
        num_files=1,
        output_folder="/scratch/philj0ly/test_baseband_data"
    )
    
    print(f"\nTest files ready! Test with your reader:")
    print(f"from your_module import get_header, BasebandFloat")
    print(f"header = get_header('{files[0]}', verbose=True)")
    print(f"bb = BasebandFloat('{files[0]}', readlen=50)")
    print(f"print('Data shapes:', bb.pol0.shape, bb.pol1.shape)")
    
    print(f"\nSignal content:")
    print(f"- Test tones at: 4.5, 5.2, 6.8, 7.3, 8.9, 10.2, 10.8 MHz")
    print(f"- Noise floor + occasional RFI")
    print(f"- Time-varying amplitude modulation")