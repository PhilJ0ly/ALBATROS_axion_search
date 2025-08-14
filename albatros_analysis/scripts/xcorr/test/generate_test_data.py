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
        
        This generates PFB (Polyphase Filter Bank) OUTPUT data - i.e., frequency-domain spectra.
        Each channel represents a frequency bin after the PFB has processed time-domain data.
        
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
        
        # Generate actual channels array (64 to 183) - FIXED: Include all 120 channels
        if channels is None:
            self.channels = np.arange(channel_start, channel_end + 1, dtype=np.uint64)
        else:
            self.channels = np.array(channels, dtype=np.uint64)
        
        # Verify we have the right number of channels
        assert len(self.channels) == length_channels, f"Expected {length_channels} channels, got {len(self.channels)}"
        
        # Calculate packet parameters to match actual data
        self.bytes_per_packet = 1204  # Match actual data exactly
        self.data_bytes_per_packet = 1200  # 1204 - 4 (for spec_num)
        
        # Calculate packets per file based on actual data
        self.packets_per_second = 415282 / 34  # ~12214 packets/sec
        self.num_packets = int(self.packets_per_second * file_duration)
        
        # Calculate spectrum rate for realistic time evolution
        self.spectrum_rate = self.packets_per_second * self.spectra_per_packet  # ~61k spectra/sec
        
        # GPS coordinates from actual data (Arctic location)
        self.gps_latitude = 79.41560919999999
        self.gps_longitude = -90.7723716
        self.gps_elevation = 179.077
        
        # Define known time-domain signals that will appear as PFB spectral features
        self.time_domain_signals = [
            {"freq_mhz": 4.5, "amplitude": 1.0, "phase": 0.0, "signal_type": "cw"},
            {"freq_mhz": 5.2, "amplitude": 0.7, "phase": np.pi/4, "signal_type": "cw"}, 
            {"freq_mhz": 6.8, "amplitude": 0.5, "phase": 0.0, "signal_type": "chirp", "chirp_rate": 1000},  # 1kHz/s chirp
            {"freq_mhz": 7.3, "amplitude": 0.8, "phase": 0.0, "signal_type": "pulsed", "pulse_period": 2.0, "duty_cycle": 0.3},
            {"freq_mhz": 8.9, "amplitude": 0.4, "phase": np.pi/2, "signal_type": "cw"},
            {"freq_mhz": 10.2, "amplitude": 0.9, "phase": 0.0, "signal_type": "am_modulated", "mod_freq": 0.5, "mod_depth": 0.5},
            {"freq_mhz": 10.8, "amplitude": 0.3, "phase": 0.0, "signal_type": "cw"},
        ]
        
        print(f"Frequency range: {self.freq_start/1e6:.2f} - {self.freq_end/1e6:.2f} MHz")
        print(f"Frequency resolution: {self.freq_resolution/1000:.1f} kHz per channel")
        print(f"Actual bandwidth: {self.actual_bandwidth/1e6:.2f} MHz")
        print(f"Channels: {self.channel_start} to {self.channel_end} ({len(self.channels)} channels)")
        print(f"Packets per file: {self.num_packets}")
        print(f"Spectrum rate: {self.spectrum_rate:.0f} spectra/sec")
        print(f"WARNING: Writing doubled channel data to compensate for reader bug in baseband_data_classes.py lines 89-91")
        
        # Print what the time-domain signals would look like
        print(f"\nTime-domain signals that will appear in PFB spectra:")
        for sig in self.time_domain_signals:
            print(f"  {sig['freq_mhz']:.1f} MHz: {sig['signal_type']} (amp={sig['amplitude']:.1f})")

    def get_frequency_array(self):
        """Get frequency array in Hz for each channel."""
        return self.channels * self.freq_resolution

    def generate_time_domain_signal_at_time(self, t_seconds):
        """
        Generate what the time-domain signal would look like at time t.
        This is for reference - we then convert to PFB spectral representation.
        
        Returns: Complex time-domain signal that would produce the PFB spectra we generate
        """
        signal = 0.0 + 0.0j
        
        for sig_def in self.time_domain_signals:
            freq_hz = sig_def["freq_mhz"] * 1e6
            amp = sig_def["amplitude"]
            phase = sig_def["phase"]
            sig_type = sig_def["signal_type"]
            
            if sig_type == "cw":
                # Simple continuous wave
                signal += amp * np.exp(1j * (2 * np.pi * freq_hz * t_seconds + phase))
                
            elif sig_type == "chirp":
                # Frequency chirp
                chirp_rate = sig_def["chirp_rate"]  # Hz/s
                instantaneous_freq = freq_hz + chirp_rate * t_seconds
                signal += amp * np.exp(1j * (2 * np.pi * instantaneous_freq * t_seconds + phase))
                
            elif sig_type == "pulsed":
                # Pulsed signal
                pulse_period = sig_def["pulse_period"]  # seconds
                duty_cycle = sig_def["duty_cycle"]
                pulse_phase = (t_seconds % pulse_period) / pulse_period
                if pulse_phase < duty_cycle:
                    signal += amp * np.exp(1j * (2 * np.pi * freq_hz * t_seconds + phase))
                    
            elif sig_type == "am_modulated":
                # AM modulated signal
                mod_freq = sig_def["mod_freq"]  # Hz
                mod_depth = sig_def["mod_depth"]
                envelope = 1.0 + mod_depth * np.sin(2 * np.pi * mod_freq * t_seconds)
                signal += amp * envelope * np.exp(1j * (2 * np.pi * freq_hz * t_seconds + phase))
        
        return signal

    def generate_pfb_spectrum_from_known_signals(self, packet_idx, spectrum_idx):
        """
        Generate PFB frequency-domain spectra based on known time-domain signals.
        
        This simulates what a PFB would output when processing the time-domain signals
        defined in self.time_domain_signals.
        
        The key insight: each PFB channel represents the complex amplitude of frequency 
        components in that channel's frequency bin. For a signal at frequency f, 
        the PFB channel covering that frequency will show the signal's amplitude and phase.
        """
        # Time for this spectrum (for time-varying effects)
        time_idx = packet_idx * self.spectra_per_packet + spectrum_idx
        abs_time = time_idx / self.spectrum_rate  # Absolute time in seconds
        
        # Get frequency array for our channels  
        freqs = self.get_frequency_array()
        
        # Initialize PFB output arrays (frequency-domain)
        pol0_real = np.zeros(self.length_channels, dtype=np.float32)
        pol0_imag = np.zeros(self.length_channels, dtype=np.float32)
        pol1_real = np.zeros(self.length_channels, dtype=np.float32)
        pol1_imag = np.zeros(self.length_channels, dtype=np.float32)
        
        # Add thermal noise floor in each frequency bin
        noise_power = 8.0  # Power per frequency bin
        np.random.seed(int((time_idx * 7919) % 2**31))  # Reproducible noise
        
        # Each PFB channel gets independent thermal noise
        pol0_real += np.random.normal(0, noise_power, self.length_channels)
        pol0_imag += np.random.normal(0, noise_power, self.length_channels)  
        pol1_real += np.random.normal(0, noise_power, self.length_channels)
        pol1_imag += np.random.normal(0, noise_power, self.length_channels)
        
        # Process each known time-domain signal
        for sig_def in self.time_domain_signals:
            freq_hz = sig_def["freq_mhz"] * 1e6
            
            # Find which PFB channel(s) this signal affects
            channel_exact = freq_hz / self.freq_resolution  # Exact channel (can be fractional)
            channel_center = int(np.round(channel_exact))
            
            # Check if signal is in our frequency range
            if not (self.channel_start <= channel_center <= self.channel_end):
                continue
                
            channel_idx = channel_center - self.channel_start
            
            # Calculate the complex amplitude this signal contributes to the PFB channel
            sig_type = sig_def["signal_type"]
            base_amp = sig_def["amplitude"] * 25  # Scale for visibility
            
            if sig_type == "cw":
                # CW signal: constant complex amplitude
                phase = sig_def["phase"]
                complex_amp = base_amp * np.exp(1j * phase)
                
            elif sig_type == "chirp":
                # Chirp: frequency drifts, so amplitude/phase varies
                chirp_rate = sig_def["chirp_rate"]
                inst_freq = freq_hz + chirp_rate * abs_time
                # Phase accumulation due to changing frequency
                phase_acc = sig_def["phase"] + 2 * np.pi * 0.5 * chirp_rate * abs_time**2
                complex_amp = base_amp * np.exp(1j * phase_acc)
                
            elif sig_type == "pulsed":
                # Pulsed signal: amplitude modulation in time
                pulse_period = sig_def["pulse_period"]
                duty_cycle = sig_def["duty_cycle"]
                pulse_phase = (abs_time % pulse_period) / pulse_period
                if pulse_phase < duty_cycle:
                    phase = sig_def["phase"] + 2 * np.pi * freq_hz * abs_time
                    complex_amp = base_amp * np.exp(1j * phase)
                else:
                    complex_amp = 0.0 + 0.0j
                    
            elif sig_type == "am_modulated":
                # AM modulation: amplitude varies sinusoidally
                mod_freq = sig_def["mod_freq"]
                mod_depth = sig_def["mod_depth"]
                envelope = 1.0 + mod_depth * np.sin(2 * np.pi * mod_freq * abs_time)
                phase = sig_def["phase"] + 2 * np.pi * freq_hz * abs_time
                complex_amp = base_amp * envelope * np.exp(1j * phase)
            
            # Add signal to appropriate PFB channel(s)
            # Handle spectral leakage: signal affects nearby channels too
            for ch_offset in range(-2, 3):  # Affect 5 channels around center
                ch_idx = channel_idx + ch_offset
                if 0 <= ch_idx < self.length_channels:
                    # Calculate leakage factor (sinc-like response)
                    if ch_offset == 0:
                        leakage_factor = 1.0
                    else:
                        # Simplified sinc function for spectral leakage
                        x = np.pi * ch_offset * 0.5
                        leakage_factor = np.abs(np.sin(x) / x) * 0.3
                    
                    # Apply signal to this channel
                    signal_contribution = complex_amp * leakage_factor
                    
                    # Add to pol0
                    pol0_real[ch_idx] += signal_contribution.real
                    pol0_imag[ch_idx] += signal_contribution.imag
                    
                    # Add to pol1 with some correlation/decorrelation
                    correlation = 0.7  # Signals are 70% correlated between pols
                    pol1_contribution = signal_contribution * correlation
                    # Add some decorrelated component
                    decorr_phase = np.random.uniform(0, 2*np.pi)
                    decorr_amp = np.abs(signal_contribution) * (1 - correlation)
                    pol1_contribution += decorr_amp * np.exp(1j * decorr_phase)
                    
                    pol1_real[ch_idx] += pol1_contribution.real
                    pol1_imag[ch_idx] += pol1_contribution.imag
        
        # Add realistic PFB artifacts
        # 1. Bandpass filter response (frequency-dependent gain)
        freq_response = 1.0 + 0.1 * np.sin(np.linspace(0, 4*np.pi, self.length_channels))
        pol0_real *= freq_response
        pol0_imag *= freq_response * 0.98  # Slight I/Q imbalance
        pol1_real *= freq_response * 0.97
        pol1_imag *= freq_response * 0.95
        
        # 2. Occasionally add RFI (appears as strong spectral lines)
        if np.random.random() < 0.02:  # 2% chance per spectrum
            rfi_freq = np.random.uniform(self.freq_start, self.freq_end)
            rfi_channel = int(rfi_freq / self.freq_resolution)
            
            if self.channel_start <= rfi_channel <= self.channel_end:
                rfi_idx = rfi_channel - self.channel_start
                rfi_power = np.random.uniform(80, 150)
                rfi_phase = np.random.uniform(0, 2*np.pi)
                
                # RFI appears as strong coherent signal
                pol0_real[rfi_idx] += rfi_power * np.cos(rfi_phase)
                pol0_imag[rfi_idx] += rfi_power * np.sin(rfi_phase)
                pol1_real[rfi_idx] += rfi_power * 0.9 * np.cos(rfi_phase + 0.2)
                pol1_imag[rfi_idx] += rfi_power * 0.9 * np.sin(rfi_phase + 0.2)
        
        # Convert to unsigned 8-bit range for 4-bit packing
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
        pol0_real_u4 = ((pol0_real_4bit + 16) % 16).astype(np.uint8)
        pol0_imag_u4 = ((pol0_imag_4bit + 16) % 16).astype(np.uint8)
        pol1_real_u4 = ((pol1_real_4bit + 16) % 16).astype(np.uint8)
        pol1_imag_u4 = ((pol1_imag_4bit + 16) % 16).astype(np.uint8)
        
        # Pack each polarization: [real nibble][imag nibble] in each byte
        pol0_bytes = (pol0_real_u4 << 4) | pol0_imag_u4
        pol1_bytes = (pol1_real_u4 << 4) | pol1_imag_u4
        
        # Interleave pol0 and pol1 bytes
        packed_bytes = np.zeros(self.length_channels * 2, dtype=np.uint8)
        packed_bytes[0::2] = pol0_bytes  # Even indices: pol0
        packed_bytes[1::2] = pol1_bytes  # Odd indices: pol1
        
        return packed_bytes

    def write_header(self, file_obj):
        """Write file header matching actual format."""
        # CRITICAL FIX: The reader code has a bug in lines 89-91 of baseband_data_classes.py
        # To compensate, we write double the channels and length_channels
        
        # Create expanded channels array to compensate for reader bug
        expanded_channels = []
        for channel in self.channels:
            expanded_channels.extend([channel, channel])  # Duplicate each channel
        expanded_channels = np.array(expanded_channels, dtype=np.uint64)
        
        channels_bytes = len(expanded_channels) * 8  # 240 channels × 8 bytes
        gps_and_padding_bytes = 5 * 8  # GPS data
        header_fields_bytes = 5 * 8   # Header fields
        
        header_size_field = 2000
        
        # Write header size (first 8 bytes)
        file_obj.write(struct.pack(">Q", header_size_field))
        
        # Write packet parameters - CRITICAL: Write 240 for length_channels
        file_obj.write(struct.pack(">Q", self.bytes_per_packet))
        file_obj.write(struct.pack(">Q", self.length_channels * 2))   # 240 (becomes 120 after reader bug)
        file_obj.write(struct.pack(">Q", self.spectra_per_packet))
        file_obj.write(struct.pack(">Q", self.bit_mode))
        file_obj.write(struct.pack(">Q", self.have_trimble))
        
        # Write expanded channels array
        for channel in expanded_channels:
            file_obj.write(struct.pack(">Q", channel))
        
        # GPS data
        file_obj.write(struct.pack(">Q", 0))  # GPS week
        file_obj.write(struct.pack(">Q", self.unix_timestamp))
        file_obj.write(struct.pack(">d", self.gps_latitude))
        file_obj.write(struct.pack(">d", self.gps_longitude))
        file_obj.write(struct.pack(">d", self.gps_elevation))
        
        # Padding to reach 2008 total bytes
        bytes_written = 8 + header_fields_bytes + channels_bytes + gps_and_padding_bytes
        padding_needed = 2008 - bytes_written
        if padding_needed > 0:
            file_obj.write(b'\x00' * padding_needed)

    def generate_file(self, timestamp):
        """Generate a single test file for given timestamp."""
        # Create folder structure
        folder_prefix = str(timestamp)[:5]
        full_folder = os.path.join(self.output_folder, folder_prefix)
        os.makedirs(full_folder, exist_ok=True)
        
        filename = f"{timestamp}.raw"
        filepath = os.path.join(full_folder, filename)
        
        print(f"Generating {filepath}...")
        print(f"  Packets: {self.num_packets}, Spectra per packet: {self.spectra_per_packet}")
        
        with open(filepath, 'wb') as f:
            # Write header (2008 bytes total)
            self.write_header(f)
            
            # Calculate starting spectrum number
            start_spec_num = int((timestamp % 100000) * 100)
            
            # Write packet data
            for packet_idx in range(self.num_packets):
                spec_num = start_spec_num + packet_idx * self.spectra_per_packet
                
                # Generate PFB spectra for all spectra in this packet
                packet_data = []
                
                for spectrum_idx in range(self.spectra_per_packet):
                    pol0_real, pol0_imag, pol1_real, pol1_imag = self.generate_pfb_spectrum_from_known_signals(packet_idx, spectrum_idx)
                    packed = self.pack_data_4bit_actual_format_complex(pol0_real, pol0_imag, pol1_real, pol1_imag)
                    packet_data.append(packed)
                
                # Combine all spectra data (should be 1200 bytes total)
                spectra_data = np.concatenate(packet_data)
                
                # Write packet: spec_num (4 bytes) + spectra data (1200 bytes)
                f.write(struct.pack(">I", spec_num))
                f.write(spectra_data.tobytes())
                
                if packet_idx % 10000 == 0:
                    print(f"    Generated {packet_idx+1}/{self.num_packets} packets")
        
        print(f"Successfully generated {filepath}")
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")
        return filepath

def generate_actual_format_test_files(start_timestamp=1721400005, 
                                     num_files=3,
                                     output_folder="test_baseband_data"):
    """
    Generate test files with PFB frequency-domain spectra based on known time-domain signals.
    
    The generated data represents what a PFB would output when processing known time-domain
    signals. After IPFB (inverse PFB), you should recover the original time-domain signals.
    """
    print(f"Generating {num_files} test files with PFB spectra from known time-domain signals...")
    print("PFB format: 120 channels (64-183), ~4-11 MHz, frequency-domain complex spectra")
    
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
        current_timestamp += 34
        print(f"Completed {i+1}/{num_files} files")
    
    print(f"\nGenerated {num_files} test files with known PFB spectra:")
    for filepath in generated_files:
        print(f"  {filepath}")
    
    return generated_files

if __name__ == "__main__":
    print("Generating test PFB files with known time-domain signal content...")
    print("Data format: PFB frequency-domain spectra")
    print("Known signals: CW, chirp, pulsed, AM modulated")
    
    # Generate test files
    files = generate_actual_format_test_files(
        start_timestamp=1721400005,
        num_files=1,
        output_folder="/scratch/philj0ly/test_baseband_data2"
    )
    
    print(f"\nTest files ready! The PFB spectra contain these known time-domain signals:")
    print(f"- 4.5 MHz: CW tone")  
    print(f"- 5.2 MHz: CW tone (different phase)")
    print(f"- 6.8 MHz: Chirped signal (1 kHz/s)")
    print(f"- 7.3 MHz: Pulsed signal (2s period, 30% duty cycle)")
    print(f"- 8.9 MHz: CW tone")
    print(f"- 10.2 MHz: AM modulated (0.5 Hz modulation, 50% depth)")
    print(f"- 10.8 MHz: CW tone")
    print(f"\nAfter IPFB, you should recover these time-domain signals!")