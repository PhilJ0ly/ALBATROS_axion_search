import struct
import numpy as np
import time
import os
from datetime import datetime, timezone

class SimpleSignalGenerator:
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
                 file_duration=12,        
                 signal_freq_mhz=7.5,    # Single test signal frequency
                 signal_amplitude=50.0):  # Signal amplitude
        """
        Generate test baseband files with ONE simple, clean signal for testing.
        
        This creates PFB frequency-domain data representing a single CW tone.
        No noise, no RFI, no artifacts - just one pure signal for easy interpretation.
        
        Time-domain signal: A(t) = amplitude * exp(j * 2π * f * t)
        PFB output: Single peak at frequency f
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
        self.signal_freq_mhz = signal_freq_mhz
        self.signal_amplitude = signal_amplitude
        
        # Calculate frequency parameters
        self.freq_resolution = bandwidth / 2048  # ~61 kHz per channel
        self.freq_start = channel_start * self.freq_resolution  # ~4 MHz
        self.freq_end = (channel_end + 1) * self.freq_resolution  # ~11.5 MHz
        
        # Generate channels array
        if channels is None:
            self.channels = np.arange(channel_start, channel_end + 1, dtype=np.uint64)
        else:
            self.channels = np.array(channels, dtype=np.uint64)
        
        # Calculate packet parameters
        self.bytes_per_packet = 1204
        self.data_bytes_per_packet = 1200
        self.packets_per_second = 415282 / 34  # ~12214 packets/sec
        self.num_packets = int(self.packets_per_second * file_duration)
        self.spectrum_rate = self.packets_per_second * self.spectra_per_packet
        
        # GPS coordinates
        self.gps_latitude = 79.41560919999999
        self.gps_longitude = -90.7723716
        self.gps_elevation = 179.077
        
        # Calculate which channel contains our signal
        self.signal_freq_hz = signal_freq_mhz * 1e6
        self.signal_channel = int(np.round(self.signal_freq_hz / self.freq_resolution))
        
        print(f"=== SIMPLE SIGNAL GENERATOR ===")
        print(f"Frequency range: {self.freq_start/1e6:.2f} - {self.freq_end/1e6:.2f} MHz")
        print(f"Channel resolution: {self.freq_resolution/1000:.1f} kHz")
        print(f"Channels: {self.channel_start} to {self.channel_end} ({len(self.channels)} total)")
        print(f"")
        print(f"SIGNAL DETAILS:")
        print(f"  Frequency: {signal_freq_mhz:.2f} MHz")
        print(f"  Amplitude: {signal_amplitude:.1f}")
        print(f"  Channel: {self.signal_channel} (index {self.signal_channel - self.channel_start})")
        print(f"")
        print(f"TIME-DOMAIN EQUIVALENT:")
        print(f"  s(t) = {signal_amplitude:.1f} * exp(j * 2π * {signal_freq_mhz:.2f}e6 * t)")
        print(f"")
        print(f"EXPECTED PFB OUTPUT:")
        print(f"  - All channels = 0 except channel {self.signal_channel}")
        print(f"  - Channel {self.signal_channel} = {signal_amplitude:.1f} + 0j (constant over time)")
        print(f"")
        print(f"Packets per file: {self.num_packets}")
        print(f"Spectrum rate: {self.spectrum_rate:.0f} spectra/sec")

    def generate_simple_pfb_spectrum(self, packet_idx, spectrum_idx):
        """
        Generate one simple PFB spectrum: single tone, no noise, no artifacts.
        
        This represents what a PFB would output for the time-domain signal:
        s(t) = amplitude * exp(j * 2π * f * t)
        
        The PFB output should be:
        - Zero in all channels except the one containing our signal frequency
        - Constant complex amplitude in the signal channel
        """
        # Initialize all channels to zero (no noise, no background)
        pol0_real = np.zeros(self.length_channels, dtype=np.float32)
        pol0_imag = np.zeros(self.length_channels, dtype=np.float32) 
        pol1_real = np.zeros(self.length_channels, dtype=np.float32)
        pol1_imag = np.zeros(self.length_channels, dtype=np.float32)
        
        # Check if our signal frequency is within the observed band
        if self.channel_start <= self.signal_channel <= self.channel_end:
            channel_idx = self.signal_channel - self.channel_start
            
            # Time for this spectrum (for any time-dependent phase evolution)
            time_idx = packet_idx * self.spectra_per_packet + spectrum_idx
            abs_time = time_idx / self.spectrum_rate
            
            signal_real = self.signal_amplitude
            signal_imag = 0.0
            
            pol0_real[channel_idx] = signal_real
            pol0_imag[channel_idx] = signal_imag
            pol1_real[channel_idx] = signal_real  # Same signal in both pols
            pol1_imag[channel_idx] = signal_imag
            
            # Debug output for first few spectra
            if time_idx < 5:
                print(f"  Spectrum {time_idx}: Channel {channel_idx} = {signal_real:.1f} + {signal_imag:.1f}j")
        else:
            print(f"Warning: Signal at {self.signal_freq_mhz:.2f} MHz (channel {self.signal_channel}) is outside observed band!")
        
        # Convert to unsigned 8-bit for packing (with offset for signed 4-bit)
        offset = 128  # Center of 8-bit range
        scale = 1.0   # No scaling - keep values interpretable
        
        pol0_real_u8 = np.clip(pol0_real * scale + offset, 0, 255).astype(np.uint8)
        pol0_imag_u8 = np.clip(pol0_imag * scale + offset, 0, 255).astype(np.uint8)
        pol1_real_u8 = np.clip(pol1_real * scale + offset, 0, 255).astype(np.uint8)
        pol1_imag_u8 = np.clip(pol1_imag * scale + offset, 0, 255).astype(np.uint8)
        
        return (pol0_real_u8, pol0_imag_u8, pol1_real_u8, pol1_imag_u8)

    def pack_data_4bit_actual_format_complex(self, pol0_real, pol0_imag, pol1_real, pol1_imag):
        """Pack complex data to match actual 4-bit format."""
        # Convert to signed range around 0
        pol0_real_centered = pol0_real.astype(np.int16) - 128
        pol0_imag_centered = pol0_imag.astype(np.int16) - 128
        pol1_real_centered = pol1_real.astype(np.int16) - 128
        pol1_imag_centered = pol1_imag.astype(np.int16) - 128
        
        # Scale to 4-bit signed range (-8 to +7)
        pol0_real_4bit = np.clip(pol0_real_centered // 8, -8, 7).astype(np.int8)
        pol0_imag_4bit = np.clip(pol0_imag_centered // 8, -8, 7).astype(np.int8)
        pol1_real_4bit = np.clip(pol1_real_centered // 8, -8, 7).astype(np.int8)
        pol1_imag_4bit = np.clip(pol1_imag_centered // 8, -8, 7).astype(np.int8)
        
        # Convert to unsigned 4-bit for packing
        pol0_real_u4 = ((pol0_real_4bit + 16) % 16).astype(np.uint8)
        pol0_imag_u4 = ((pol0_imag_4bit + 16) % 16).astype(np.uint8)
        pol1_real_u4 = ((pol1_real_4bit + 16) % 16).astype(np.uint8)
        pol1_imag_u4 = ((pol1_imag_4bit + 16) % 16).astype(np.uint8)
        
        # Pack: upper 4 bits = real, lower 4 bits = imag
        pol0_bytes = (pol0_real_u4 << 4) | pol0_imag_u4
        pol1_bytes = (pol1_real_u4 << 4) | pol1_imag_u4
        
        # Interleave: pol0, pol1, pol0, pol1, ...
        packed_bytes = np.zeros(self.length_channels * 2, dtype=np.uint8)
        packed_bytes[0::2] = pol0_bytes
        packed_bytes[1::2] = pol1_bytes
        
        return packed_bytes

    def write_header(self, file_obj):
        """Write file header - compensating for reader bug."""
        # Duplicate channels to compensate for reader bug that does [::2]
        expanded_channels = []
        for channel in self.channels:
            expanded_channels.extend([channel, channel])
        expanded_channels = np.array(expanded_channels, dtype=np.uint64)
        
        # Header structure
        header_size_field = 2000
        
        # Write header
        file_obj.write(struct.pack(">Q", header_size_field))
        file_obj.write(struct.pack(">Q", self.bytes_per_packet))
        file_obj.write(struct.pack(">Q", self.length_channels * 2))  # Double for reader bug
        file_obj.write(struct.pack(">Q", self.spectra_per_packet))
        file_obj.write(struct.pack(">Q", self.bit_mode))
        file_obj.write(struct.pack(">Q", self.have_trimble))
        
        # Write expanded channels
        for channel in expanded_channels:
            file_obj.write(struct.pack(">Q", channel))
        
        # GPS data
        file_obj.write(struct.pack(">Q", 0))
        file_obj.write(struct.pack(">Q", self.unix_timestamp))
        file_obj.write(struct.pack(">d", self.gps_latitude))
        file_obj.write(struct.pack(">d", self.gps_longitude))
        file_obj.write(struct.pack(">d", self.gps_elevation))
        
        # Padding to 2008 bytes total
        bytes_written = 8 + 5*8 + len(expanded_channels)*8 + 5*8
        padding_needed = 2008 - bytes_written
        if padding_needed > 0:
            file_obj.write(b'\x00' * padding_needed)

    def generate_file(self, timestamp):
        """Generate a single test file."""
        folder_prefix = str(timestamp)[:5]
        full_folder = os.path.join(self.output_folder, folder_prefix)
        os.makedirs(full_folder, exist_ok=True)
        
        filename = f"{timestamp}.raw"
        filepath = os.path.join(full_folder, filename)
        
        print(f"\nGenerating {filepath}...")
        
        with open(filepath, 'wb') as f:
            self.write_header(f)
            
            start_spec_num = int((timestamp % 100000) * 100)
            
            for packet_idx in range(self.num_packets):
                spec_num = start_spec_num + packet_idx * self.spectra_per_packet
                
                packet_data = []
                for spectrum_idx in range(self.spectra_per_packet):
                    pol0_real, pol0_imag, pol1_real, pol1_imag = self.generate_simple_pfb_spectrum(packet_idx, spectrum_idx)
                    packed = self.pack_data_4bit_actual_format_complex(pol0_real, pol0_imag, pol1_real, pol1_imag)
                    packet_data.append(packed)
                
                spectra_data = np.concatenate(packet_data)
                
                # Write packet
                f.write(struct.pack(">I", spec_num))
                f.write(spectra_data.tobytes())
                
                if packet_idx % 10000 == 0:
                    print(f"    Generated {packet_idx+1}/{self.num_packets} packets")
        
        print(f"Successfully generated {filepath}")
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")
        return filepath

def generate_simple_test_files(start_timestamp=1721400005,
                              num_files=1,
                              output_folder="simple_test_data",
                              signal_freq_mhz=7.5,
                              signal_amplitude=50.0,
                              file_duration=34):
    """
    Generate simple test files with ONE clean signal for easy testing.
    
    Parameters:
    - signal_freq_mhz: Frequency of the test signal (should be in 4-11 MHz range)
    - signal_amplitude: Amplitude of the signal (larger = stronger signal)
    """
    print("=== GENERATING SIMPLE TEST FILES ===")
    print(f"Signal: {signal_freq_mhz:.2f} MHz CW tone, amplitude {signal_amplitude:.1f}")
    print(f"No noise, no RFI, no artifacts - pure signal for testing")
    
    generator = SimpleSignalGenerator(
        start_timestamp,
        output_folder=output_folder,
        signal_freq_mhz=signal_freq_mhz,
        signal_amplitude=signal_amplitude,
        file_duration=file_duration
    )
    
    generated_files = []
    current_timestamp = start_timestamp
    
    for i in range(num_files):
        filepath = generator.generate_file(current_timestamp)
        generated_files.append(filepath)
        current_timestamp += file_duration
        print(f"Completed {i+1}/{num_files} files")
    
    print(f"\n=== TEST FILES GENERATED ===")
    for filepath in generated_files:
        print(f"  {filepath}")
    
    print(f"\n=== WHAT TO EXPECT ===")
    print(f"PFB Data:")
    print(f"  - Channel {generator.signal_channel}: amplitude {signal_amplitude:.1f} + 0j")
    print(f"  - All other channels: 0 + 0j")
    print(f"")
    print(f"After IPFB, you should recover:")
    print(f"  Time-domain signal: {signal_amplitude:.1f} * cos(2π * {signal_freq_mhz:.2f}e6 * t)")
    
    return generated_files

if __name__ == "__main__":
    # Generate simple test with one clean signal
    files = generate_simple_test_files(
        start_timestamp=1721500006,
        num_files=3,
        output_folder="/scratch/philj0ly/simple_test_data",
        signal_freq_mhz=7.5,    # Middle of the band
        signal_amplitude=50.0,   # Strong, clear signal
        file_duration=12
    )
    
    print(f"\nTest the file with:")
    print(f"from baseband_data_classes import get_header, BasebandFloat")
    print(f"header = get_header('{files[0]}', verbose=True)")
    print(f"bb = BasebandFloat('{files[0]}', readlen=100)")
    print(f"print('Data shapes:', bb.pol0.shape, bb.pol1.shape)")
    print(f"print('Channel values:', bb.pol0[0, :])  # Should be mostly zeros with one peak")