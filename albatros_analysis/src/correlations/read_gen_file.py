
import numpy as np
from baseband_data_classes import BasebandFloat, get_header
from os import path
import matplotlib.pyplot as plt

dir_path = "/scratch/philj0ly/test_baseband_data"
fn = "17214/1721400005.raw"
f_path = path.join(dir_path, fn)

header = get_header(f_path, verbose=True)
bb = BasebandFloat(f_path, readlen=50)

print('Data shapes:', bb.pol0.shape, bb.pol1.shape) 

f_bin = 125*1e6 / 2048
start_chan = 64

spectrum = np.abs(bb.pol0[0])
freqs = np.arange(start_chan, start_chan + len(spectrum)) * f_bin

print(spectrum)
spec = np.zeros(spectrum.shape[0], dtype=np.float32)
for i in range(spectrum.shape[0]):
    spec[i] = spectrum[i]

plt.figure(figsize=(8,6))
plt.plot(freqs, spec, label='Pol0 Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Spectrum of Pol0')
plt.legend()
plt.savefig(f"spectrum_pol0.png")
