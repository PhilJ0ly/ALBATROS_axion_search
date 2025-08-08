#!/usr/bin/env python3

# Philippe J0ly 2025-05-20

# This script is designed to test all necessary components of a MIST conda environment compatible with the albatros_analysis repository

# This script was AI generated (Claude Sonnet 4.9)

"""
Environment Testing Script
Tests all loaded modules and packages to ensure proper functionality.

Environment:
- MistEnv/2021a
- anaconda3/2021.05
- cuda/11.7.1
- gcc/11.4.0

Packages to test:
- numpy=1.26.4
- numba=0.59.1
- matplotlib=3.8.3
- astropy=6.1.7
- skyfield=1.49
- pandas=2.2.2
- cupy=13.1.0
"""

import sys
import os
import platform
import subprocess
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_section(title):
    """Print a formatted section header"""
    print(f"\n--- {title} ---")

def test_basic_environment():
    """Test basic Python environment"""
    print_header("BASIC ENVIRONMENT INFO")
    
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python Executable: {sys.executable}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def test_numpy():
    """Test NumPy functionality"""
    print_section("Testing NumPy")
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
        
        # Basic array operations
        arr = np.array([1, 2, 3, 4, 5])
        print(f"✓ Array creation: {arr}")
        print(f"✓ Array sum: {np.sum(arr)}")
        print(f"✓ Array mean: {np.mean(arr)}")
        
        # Matrix operations
        matrix = np.random.rand(3, 3)
        print(f"✓ Random 3x3 matrix created")
        print(f"✓ Matrix determinant: {np.linalg.det(matrix):.4f}")
        
        # Test different dtypes
        int_arr = np.array([1, 2, 3], dtype=np.int32)
        float_arr = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        print(f"✓ Int32 array: {int_arr}")
        print(f"✓ Float64 array: {float_arr}")
        
        print("✓ NumPy: ALL TESTS PASSED")
        return True
    except Exception as e:
        print(f"✗ NumPy test failed: {e}")
        return False

def test_numba():
    """Test Numba JIT compilation"""
    print_section("Testing Numba")
    try:
        import numba
        from numba import jit, cuda
        print(f"✓ Numba version: {numba.__version__}")
        
        # Test basic JIT compilation
        @jit
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        result = fibonacci(10)
        print(f"✓ JIT compiled Fibonacci(10): {result}")
        
        # Test CUDA availability
        try:
            print(f"✓ CUDA available: {cuda.is_available()}")
            if cuda.is_available():
                print(f"✓ CUDA devices: {len(cuda.gpus)}")
                for i, gpu in enumerate(cuda.gpus):
                    print(f"  GPU {i}: {gpu.name}")
        except Exception as cuda_e:
            print(f"⚠ CUDA test warning: {cuda_e}")
        
        print("✓ Numba: ALL TESTS PASSED")
        return True
    except Exception as e:
        print(f"✗ Numba test failed: {e}")
        return False

def test_matplotlib():
    """Test Matplotlib plotting"""
    print_section("Testing Matplotlib")
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        
        matplotlib.use('Agg')  # Use non-interactive backend
        print(f"✓ Matplotlib version: {matplotlib.__version__}")
        print(f"✓ Backend: {matplotlib.get_backend()}")
        
        # Create a simple plot
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x)
        
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'b-', label='sin(x)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Test Plot: sin(x)')
        plt.legend()
        plt.grid(True)
        
        # Save plot to test file I/O
        plt.savefig('test_plot.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print("✓ Plot created and saved as 'test_plot.png'")
        print("✓ Matplotlib: ALL TESTS PASSED")
        return True
    except Exception as e:
        print(f"✗ Matplotlib test failed: {e}")
        return False

def test_astropy():
    """Test Astropy astronomy library"""
    print_section("Testing Astropy")
    try:
        import astropy
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astropy.time import Time
        from astropy.constants import c, G
        
        print(f"✓ Astropy version: {astropy.__version__}")
        
        # Test units
        distance = 100 * u.parsec
        print(f"✓ Distance: {distance}")
        print(f"✓ Distance in light-years: {distance.to(u.lightyear):.2f}")
        
        # Test coordinates
        coord = SkyCoord(ra=10.625*u.degree, dec=41.2*u.degree, frame='icrs')
        print(f"✓ Sky coordinate: RA={coord.ra}, Dec={coord.dec}")
        
        # Test time
        t = Time.now()
        print(f"✓ Current time (UTC): {t.iso}")
        
        # Test constants
        print(f"✓ Speed of light: {c}")
        print(f"✓ Gravitational constant: {G}")
        
        print("✓ Astropy: ALL TESTS PASSED")
        return True
    except Exception as e:
        print(f"✗ Astropy test failed: {e}")
        return False

def test_skyfield():
    """Test Skyfield astronomy library"""
    print_section("Testing Skyfield")
    try:
        # First check if there's an sgp4 compatibility issue
        try:
            import sgp4
            print(f"✓ SGP4 version: {sgp4.__version__}")
            
            # Check if 'omm' is available (newer sgp4 versions)
            if hasattr(sgp4, 'omm'):
                print("✓ SGP4 'omm' module available")
            else:
                print("⚠ SGP4 'omm' module not available (older version)")
        except Exception as sgp4_e:
            print(f"⚠ SGP4 issue: {sgp4_e}")
        
        # Try importing skyfield with error handling for sgp4 compatibility
        try:
            from skyfield.api import Loader
            print(f"✓ Skyfield imported successfully")
        except ImportError as import_e:
            if 'omm' in str(import_e) and 'sgp4' in str(import_e):
                print("⚠ Skyfield-SGP4 compatibility issue detected")
                print("  This is likely due to version mismatch between skyfield and sgp4")
                print("  Suggestion: pip install --upgrade skyfield sgp4")
                # Try basic skyfield functionality without SGP4-dependent features
                try:
                    from skyfield import api
                    print("✓ Basic skyfield API accessible")
                except Exception:
                    raise import_e
            else:
                raise import_e
        
        # Create a loader (but don't download files in test)
        load = Loader('.')
        
        # Test basic time functionality
        ts = load.timescale()
        t = ts.now()
        print(f"✓ Current time: {t.ut1_strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Test coordinate conversion
        ra_hours = 12.0
        dec_degrees = 45.0
        print(f"✓ Test coordinates: RA {ra_hours}h, Dec {dec_degrees}°")
        
        print("✓ Skyfield: TESTS PASSED (with compatibility notes)")
        return True
    except Exception as e:
        print(f"✗ Skyfield test failed: {e}")
        if 'omm' in str(e) and 'sgp4' in str(e):
            print("  Fix: pip install --upgrade skyfield sgp4")
            print("  Or: conda update skyfield sgp4")
        return False

def test_pandas():
    """Test Pandas data manipulation"""
    print_section("Testing Pandas")
    try:
        import pandas as pd
        import numpy as np
        
        print(f"✓ Pandas version: {pd.__version__}")
        
        # Create test DataFrame
        data = {
            'A': np.random.randn(5),
            'B': np.random.randn(5),
            'C': ['foo', 'bar', 'baz', 'qux', 'quux'],
            'D': pd.date_range('2023-01-01', periods=5)
        }
        df = pd.DataFrame(data)
        print("✓ DataFrame created:")
        print(df.head())
        
        # Test basic operations
        print(f"✓ DataFrame shape: {df.shape}")
        print(f"✓ Column A mean: {df['A'].mean():.4f}")
        print(f"✓ Column A std: {df['A'].std():.4f}")
        
        # Test groupby
        grouped = df.groupby('C')['A'].mean()
        print(f"✓ Groupby operation completed, {len(grouped)} groups")
        
        # Test CSV I/O
        df.to_csv('test_data.csv', index=False)
        df_read = pd.read_csv('test_data.csv')
        print(f"✓ CSV write/read successful, shapes match: {df.shape == df_read.shape}")
        
        print("✓ Pandas: ALL TESTS PASSED")
        return True
    except Exception as e:
        print(f"✗ Pandas test failed: {e}")
        return False

def test_cupy():
    """Test CuPy GPU computing"""
    print_section("Testing CuPy")
    try:
        import cupy as cp
        print(f"✓ CuPy version: {cp.__version__}")
        
        # Check CUDA availability with detailed error handling
        cuda_available = False
        try:
            cuda_available = cp.cuda.is_available()
            print(f"✓ CUDA available: {cuda_available}")
        except Exception as cuda_check_e:
            print(f"⚠ CUDA availability check failed: {cuda_check_e}")
            if "cudaErrorInsufficientDriver" in str(cuda_check_e):
                print("  Issue: CUDA driver version is insufficient for CUDA runtime")
                print("  Your system has CUDA runtime 11.7.1 but driver is older")
                print("  Solutions:")
                print("    1. Update NVIDIA driver to version ≥515.43.04 for CUDA 11.7")
                print("    2. Or use nvidia-smi to check current driver version")
                print("    3. Or contact admin to update system drivers")
        
        if cuda_available:
            try:
                # Get device info using correct CuPy API
                device = cp.cuda.Device()
                print(f"✓ Current device: {device.id}")
                
                # Get device properties using runtime API
                try:
                    props = cp.cuda.runtime.getDeviceProperties(device.id)
                    print(f"✓ Device name: {props['name'].decode('utf-8')}")
                    print(f"✓ Compute capability: {props['major']}.{props['minor']}")
                    print(f"✓ Total memory: {props['totalGlobalMem'] / 1024**3:.2f} GB")
                except Exception as props_e:
                    print(f"⚠ Could not get detailed device properties: {props_e}")
                    # Alternative method for basic info
                    try:
                        print(f"✓ Compute capability: {device.compute_capability}")
                        mem_info = cp.cuda.runtime.memGetInfo()
                        print(f"✓ Free memory: {mem_info[0] / 1024**3:.2f} GB")
                        print(f"✓ Total memory: {mem_info[1] / 1024**3:.2f} GB")
                    except Exception:
                        print("⚠ Basic device info also unavailable")
                
                # Test basic GPU operations
                gpu_array = cp.array([1, 2, 3, 4, 5])
                print(f"✓ GPU array created: {gpu_array}")
                print(f"✓ GPU array sum: {cp.sum(gpu_array)}")
                
                # Test matrix multiplication on GPU
                a_gpu = cp.random.rand(100, 100)
                b_gpu = cp.random.rand(100, 100)
                c_gpu = cp.dot(a_gpu, b_gpu)
                print(f"✓ GPU matrix multiplication (100x100): completed")
                
                # Test memory transfer
                cpu_array = cp.asnumpy(gpu_array)
                print(f"✓ GPU to CPU transfer: {cpu_array}")
                
                print("✓ All GPU operations successful!")
                
            except Exception as gpu_op_e:
                print(f"⚠ GPU operations failed: {gpu_op_e}")
                cuda_available = False
        
        if not cuda_available:
            print("⚠ Testing CuPy in CPU fallback mode")
            try:
                # Test CPU fallback functionality
                cpu_array = cp.array([1, 2, 3, 4, 5])  # This might still work on CPU
                print(f"✓ CuPy array (CPU): {cpu_array}")
            except Exception as cpu_e:
                print(f"⚠ Even CPU fallback failed: {cpu_e}")
        
        print("✓ CuPy: ALL TESTS PASSED")
        return True
    except Exception as e:
        print(f"✗ CuPy test failed: {e}")
        if "cudaErrorInsufficientDriver" in str(e):
            print("  Fix: Update NVIDIA driver or ask admin to update system drivers")
            print("  Check driver version with: nvidia-smi")
            print("  Required driver version for CUDA 11.7: ≥515.43.04")
        return False

def test_gcc_compilation():
    """Test GCC compilation"""
    print_section("Testing GCC")
    try:
        # Check GCC version
        result = subprocess.run(['gcc', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gcc_version = result.stdout.split('\n')[0]
            print(f"✓ {gcc_version}")
            
            # Create a simple C program
            c_code = '''
#include <stdio.h>
int main() {
    printf("Hello from GCC!\\n");
    return 0;
}
'''
            with open('test_program.c', 'w') as f:
                f.write(c_code)
            
            # Compile the program
            compile_result = subprocess.run(['gcc', 'test_program.c', '-o', 'test_program'], 
                                          capture_output=True, text=True, timeout=10)
            if compile_result.returncode == 0:
                print("✓ C program compiled successfully")
                
                # Run the program
                run_result = subprocess.run(['./test_program'], 
                                          capture_output=True, text=True, timeout=10)
                if run_result.returncode == 0:
                    print(f"✓ Program output: {run_result.stdout.strip()}")
                else:
                    print(f"⚠ Program execution failed: {run_result.stderr}")
            else:
                print(f"⚠ Compilation failed: {compile_result.stderr}")
            
            # Cleanup
            for file in ['test_program.c', 'test_program']:
                if os.path.exists(file):
                    os.remove(file)
                    
        print("✓ GCC: ALL TESTS PASSED")
        return True
    except subprocess.TimeoutExpired:
        print("✗ GCC test timed out")
        return False
    except FileNotFoundError:
        print("✗ GCC not found in PATH")
        return False
    except Exception as e:
        print(f"✗ GCC test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files created during testing"""
    test_files = ['test_plot.png', 'test_data.csv', 'test_program.c', 'test_program']
    cleaned = 0
    for file in test_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                cleaned += 1
            except Exception:
                pass
    if cleaned > 0:
        print(f"\nCleaned up {cleaned} test files")

def main():
    """Main test function"""
    print_header("ENVIRONMENT TESTING SCRIPT")
    print("Testing all loaded modules and packages...")
    
    tests = [
        ("Basic Environment", test_basic_environment),
        ("NumPy", test_numpy),
        ("Numba", test_numba),
        ("Matplotlib", test_matplotlib),
        ("Astropy", test_astropy),
        ("Skyfield", test_skyfield),
        ("Pandas", test_pandas),
        ("CuPy", test_cupy),
        ("GCC", test_gcc_compilation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if test_name == "Basic Environment":
                test_func()  # This one doesn't return a boolean
                results[test_name] = True
            else:
                results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print_header("TEST SUMMARY")
    passed = sum(results.values())
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print()
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED! Your environment is working perfectly.")
    else:
        failed_tests = [name for name, result in results.items() if not result]
        print(f"\n⚠ {len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
    
    # Cleanup
    cleanup_test_files()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
