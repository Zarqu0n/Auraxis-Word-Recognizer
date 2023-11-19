import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

def fft_plot(filename):
    sampFreq, sound = wavfile.read(filename, 'r')
    sound = sound / 2.0 ** 15
    fft_spectrum = np.fft.rfft(sound)
    freq = np.fft.rfftfreq(sound.size, d=1. / sampFreq)
    print(sound.size)
    fft_spectrum_abs = np.abs(fft_spectrum)
    plt.plot(freq, fft_spectrum_abs)
    plt.xlabel("frequency, Hz")
    plt.ylabel("Amplitude, units")
    plt.show()
fft_plot("assets/merkez.wav")
fft_plot("assets/merhaba.wav")