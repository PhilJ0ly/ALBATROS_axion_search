import struct
import numpy as np
import os
from typing import Optional, Union, List

class BasebandTestFileGenerator:
    """
    Generate test files with the exact same format as your real baseband data.
    Simple signal generator using superposition of pure frequencies.
    """
    
    def __init__(self, output_folder: str = "test_baseband_data"):
        # Fixed header values based on your real data
        self.bytes_per_packet = 1204
        self.length_channels = 120  # Number of channels in the data
        self.spectra_per_packet = 5
        self.bit_mode = 4
        self.have_trimble = 1
        self.output_folder = output_folder
        
        # Calculate derived values
        self.data_bytes_per_packet = self.bytes_per_packet - 4  # 1200 bytes (4 for spec_num)
        self.bytes_per_spectrum = 2 * self.length_channels  # 240 bytes per spectrum
        
        # Default channel range (64-183, total of 120 channels)
        self.channels = np.arange(64, 184, dtype='>u8')  # 120 channels
        
        # GPS data (can be customized)
        self.gps_week = 0
        self.gps_latitude = 79.41560919999999
        self.gps_longitude = -90.7723716
        self.gps_elevation = 179.077
        
        # Calculate packets needed for 34-second files
        self.sample_rate = 250e6  # 250 Msps
        self.fft_size = 2048
        self.file_duration = 34.0  # seconds
        
        time_per_spectrum = self.fft_size / self.sample_rate
        total_spectra = int(self.file_duration / time_per_spectrum)
        self.num_packets = int(np.ceil(total_spectra / self.spectra_per_packet))
        
        print(f"Configured for {self.file_duration}s files:")
        print(f"  Time per spectrum: {time_per_spectrum*1000:.3f} ms")
        print(f"  Total spectra per file: {total_spectra}")
        print(f"  Packets per file: {self.num_packets}")
        
        # Signal parameters (frequencies as fractions of sample rate)
        self.frequencies = [0.01, 0.05, 0.1]  # Multiple frequency components
        self.amplitudes = [0.8, 0.6, 0.4]     # Amplitudes for each frequency
        self.phases = [0.0, np.pi/3, np.pi/2]  # Phase offsets
        
    def set_signal_parameters(self, frequencies: List[float], 
                            amplitudes: List[float] = None, 
                            phases: List[float] = None):
        """
        Set signal parameters for frequency superposition.
        
        Parameters:
        -----------
        frequencies : List[float]
            List of frequencies as fractions of sample rate (0 to 0.5)
        amplitudes : List[float], optional
            Amplitudes for each frequency component
        phases : List[float], optional  
            Phase offsets for each frequency component
        """
        self.frequencies = frequencies
        
        if amplitudes is None:
            # Default: decreasing amplitudes
            self.amplitudes = [1.0 / (i + 1) for i in range(len(frequencies))]
        else:
            self.amplitudes = amplitudes
            
        if phases is None:
            # Default: zero phases
            self.phases = [0.0] * len(frequencies)
        else:
            self.phases = phases
            
        print(f"Signal configured with {len(frequencies)} frequency components:")
        for i, (f, a, p) in enumerate(zip(self.frequencies, self.amplitudes, self.phases)):
            print(f"  Component {i+1}: freq={f:.3f}, amp={a:.2f}, phase={p:.2f}")
    
    def generate_realistic_signals(self, packet_idx: int, spectrum_idx: int):
        """
        Generate realistic complex signals using superposition of frequencies.
        
        Returns pol0_real, pol0_imag, pol1_real, pol1_imag arrays for all channels.
        """
        # Global spectrum index across all packets
        global_spec_idx = packet_idx * self.spectra_per_packet + spectrum_idx
        
        # Initialize output arrays
        pol0_real = np.zeros(self.length_channels, dtype=np.float32)
        pol0_imag = np.zeros(self.length_channels, dtype=np.float32)
        pol1_real = np.zeros(self.length_channels, dtype=np.float32) 
        pol1_imag = np.zeros(self.length_channels, dtype=np.float32)
        
        # Generate superposition of frequencies
        for freq, amp, phase in zip(self.frequencies, self.amplitudes, self.phases):
            # Time evolution
            time_phase = 2 * np.pi * freq * global_spec_idx
            
            # Frequency domain response (varies across channels)
            for chan_idx in range(self.length_channels):
                # Channel-dependent phase
                chan_phase = 2 * np.pi * freq * chan_idx / self.length_channels
                total_phase = time_phase + chan_phase + phase
                
                # Add to pol0
                pol0_real[chan_idx] += amp * np.cos(total_phase)
                pol0_imag[chan_idx] += amp * np.sin(total_phase)
                
                # pol1 has slight phase offset and amplitude difference
                pol1_phase = total_phase + np.pi/4  # 45 degree phase difference
                pol1_real[chan_idx] += amp * 0.9 * np.cos(pol1_phase)
                pol1_imag[chan_idx] += amp * 0.9 * np.sin(pol1_phase)
        
        # Add small amount of noise
        noise_level = 0.1
        pol0_real += noise_level * np.random.randn(self.length_channels)
        pol0_imag += noise_level * np.random.randn(self.length_channels)
        pol1_real += noise_level * np.random.randn(self.length_channels)
        pol1_imag += noise_level * np.random.randn(self.length_channels)
        
        return pol0_real, pol0_imag, pol1_real, pol1_imag
    
    def pack_data_4bit_actual_format_complex(self, pol0_real, pol0_imag, pol1_real, pol1_imag):
        """
        Pack complex data into 4-bit format matching your actual file structure.
        
        Returns packed data for one spectrum (240 bytes = 120 channels * 2 bytes/channel).
        """
        packed_data = np.zeros(self.bytes_per_spectrum, dtype=np.uint8)
        
        for chan_idx in range(self.length_channels):
            # Quantize to 4-bit signed values (-8 to +7)
            # Scale and clip the floating point values
            scale_factor = 6.0  # Adjust to use most of the 4-bit range
            
            p0_real_int = np.clip(np.round(pol0_real[chan_idx] * scale_factor), -8, 7).astype(int)
            p0_imag_int = np.clip(np.round(pol0_imag[chan_idx] * scale_factor), -8, 7).astype(int)
            p1_real_int = np.clip(np.round(pol1_real[chan_idx] * scale_factor), -8, 7).astype(int)
            p1_imag_int = np.clip(np.round(pol1_imag[chan_idx] * scale_factor), -8, 7).astype(int)
            
            # Convert signed to unsigned 4-bit (0-15)
            p0_real_u4 = (p0_real_int + 8) & 0xF
            p0_imag_u4 = (p0_imag_int + 8) & 0xF
            p1_real_u4 = (p1_real_int + 8) & 0xF
            p1_imag_u4 = (p1_imag_int + 8) & 0xF
            
            # Pack into bytes: pol0 byte = (real<<4) | imag, pol1 byte = (real<<4) | imag
            pol0_byte = (p0_real_u4 << 4) | p0_imag_u4
            pol1_byte = (p1_real_u4 << 4) | p1_imag_u4
            
            # Store in interleaved format: pol0_byte, pol1_byte for each channel
            packed_data[2 * chan_idx] = pol0_byte
            packed_data[2 * chan_idx + 1] = pol1_byte
        
        return packed_data
    
    def write_header(self, f):
        """Write the file header in the correct format."""
        
        # Calculate header size
        channels_size = len(self.channels) * 8  # Each channel is 8 bytes
        header_data_bytes = 8 * 10 + channels_size  # 10 fixed fields + channels
        
        # Write header size (excluding the 8 bytes for this field)
        f.write(struct.pack('>Q', header_data_bytes))
        
        # Write header fields
        f.write(struct.pack('>Q', self.bytes_per_packet))
        f.write(struct.pack('>Q', self.length_channels))
        f.write(struct.pack('>Q', self.spectra_per_packet))
        f.write(struct.pack('>Q', self.bit_mode))
        f.write(struct.pack('>Q', self.have_trimble))
        
        # Write channels array
        for channel in self.channels:
            f.write(struct.pack('>Q', channel))
            
        # Write GPS data
        f.write(struct.pack('>Q', self.gps_week))
        f.write(struct.pack('>Q', self.gps_timestamp))
        f.write(struct.pack('>d', self.gps_latitude))
        f.write(struct.pack('>d', self.gps_longitude))
        f.write(struct.pack('>d', self.gps_elevation))

    def generate_file(self, timestamp):
        """Generate a single test file for given timestamp."""
        # Create folder structure (e.g., "17214" from timestamp)
        folder_prefix = str(timestamp)[:5]
        full_folder = os.path.join(self.output_folder, folder_prefix)
        os.makedirs(full_folder, exist_ok=True)
        
        filename = f"{timestamp}.raw"
        filepath = os.path.join(full_folder, filename)
        
        # Update GPS timestamp for this file
        self.gps_timestamp = timestamp
        
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
                
                if packet_idx % 100 == 0:
                    print(f"    Generated {packet_idx+1}/{self.num_packets} packets")
        
        print(f"Successfully generated {filepath}")
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")
        return filepath

def generate_original_format_test_files(start_timestamp: int, 
                                      num_files: int = 5,
                                      output_folder: str = "test_baseband_data",
                                      frequencies: List[float] = None,
                                      amplitudes: List[float] = None,
                                      phases: List[float] = None) -> List[str]:
    """
    Generate test files in the original naming/folder structure.
    
    Parameters:
    -----------
    start_timestamp : int
        Unix timestamp for the first file (e.g., 1721400005)
    num_files : int
        Number of files to generate (default 5)
    output_folder : str
        Base output folder path
    frequencies : List[float], optional
        Frequency components as fractions of sample rate (0 to 0.5)
    amplitudes : List[float], optional
        Amplitudes for each frequency component
    phases : List[float], optional
        Phase offsets for each frequency component
        
    Returns:
    --------
    List[str]
        List of full paths to generated files
    """
    
    # Create generator
    generator = BasebandTestFileGenerator(output_folder)
    
    # Set signal parameters if provided
    if frequencies is not None:
        generator.set_signal_parameters(frequencies, amplitudes, phases)
    
    generated_files = []
    
    for i in range(num_files):
        # Calculate timestamp for this file (34 seconds apart)
        current_timestamp = start_timestamp + i * int(generator.file_duration)
        
        # Generate the file
        filepath = generator.generate_file(current_timestamp)
        generated_files.append(filepath)
        
    print(f"\n=== Generated {num_files} test files ===")
    print("File structure:")
    for file_path in generated_files:
        print(f"  {file_path}")
        
    return generated_files

def generate_simple_test_sequence():
    """Generate a simple test sequence with known frequencies."""
    return generate_original_format_test_files(
        start_timestamp=1721400005,
        num_files=5,
        output_folder="test_baseband_data",
        frequencies=[0.02, 0.08, 0.15],  # Three frequency components
        amplitudes=[1.0, 0.7, 0.5],     # Decreasing amplitudes
        phases=[0.0, np.pi/4, np.pi/2]  # Different phases
    )

if __name__ == "__main__":
    print("=== Baseband Test File Generator ===")
    
    # Generate files with custom signal
    print("\nGenerating test files with multi-frequency signal...")
    files = generate_original_format_test_files(
        start_timestamp=1721400005,
        num_files=1,
        output_folder="/scratch/philj0ly/test_baseband_data",
        frequencies=[0.01, 0.05, 0.12],  # Low, medium, and higher frequency
        amplitudes=[0.8, 0.6, 0.4],     # Decreasing amplitudes
        phases=[0.0, np.pi/3, np.pi/2]  # Phase variety
    )
    
    print("\n=== Usage Example ===")
    print("To test with your existing code:")
    print("from your_module import Baseband")
    print("test_obj = Baseband('test_baseband_data/17214/1721400005.raw', readlen=100)")
    print("test_obj.print_header()")
    print()
    print("# For testing correlation across files:")
    print("file_paths = [")
    for fp in files[:3]:  # Show first 3 files
        print(f"    '{fp}',")
    print("]")
    print("iterator = BasebandFileIterator(file_paths, fileidx=0, idxstart=0, acclen=1000)")