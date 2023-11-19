import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
def plot_audio(filename):
    audio = wave.open(filename, 'r')
    fs = audio.getframerate()
    n_samples = audio.getnframes()
    t_audio = n_samples/(fs*2)
    signal_wave = audio.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)
    channel = audio.getnchannels()
    print(signal_array)
    times = np.linspace(0, n_samples/(fs*2), num=n_samples)
    plt.figure(figsize=(15, 5))
    plt.plot(times, signal_array)
    plt.title('Audio')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    plt.show()

plot_audio('assets/merhaba.wav')
plot_audio('assets/me.wav')