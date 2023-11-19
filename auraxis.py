#!/usr/bin/python
#Author Hüseyin Ayvacı, Zarqu0n, https://github.com/Zarqu0n

#Some requirement librarys
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import statistics
from math import ceil,cos
from scipy.fftpack import dct
import pyaudio

class Extraction:
    def __init__(self,filename,
    NFFT=512,
    scale=False,
    framig_overlap=0.3,
    filter_kernel_size = 3,
    windowing_type='Hamming',
    pre_emphasis_coefficient = 0.95,
    threshold_percentage = 0.01,
    cut_cycle=0.001,
    number_filt = 40,
    number_ceps = 39) -> None:


        self.NFFT = NFFT
        self.scale = scale
        self.filename = filename
        self.pre_emphasis_coefficient = pre_emphasis_coefficient
        self.framing_overlap = framig_overlap
        self.M = filter_kernel_size
        self.th_p = threshold_percentage
        self.cut_cycle = cut_cycle
        self.windowing_type = windowing_type
        self.nfilt = number_filt
        self.num_ceps = number_ceps
        self.fs : int
        self.n_samples : int
        self.t_audio : int


    def plot(self,signals,names):
        fig =plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(len(signals))
        axs = gs.subplots(sharex=True, sharey=True)
        fig.suptitle('Signals')
        for i in range(0,len(signals)):
            signal=signals[i]
            n_sample = len(signal)
            name=names[i]
            times = np.linspace(0, n_sample/(self.fs*2), num=n_sample)
            axs[i].plot(times,signal)
            axs[i].set_title(name)
        plt.show()


    def read(self):
        audio = wave.open(self.filename, 'rb')
        self.fs = audio.getframerate()
        self.n_sample = audio.getnframes()
        self.t_audio = self.n_sample/(self.fs*2)
        signal_wave = audio.readframes(self.n_sample)
        signal_array = np.frombuffer(signal_wave, dtype=np.int16).astype(np.float32)
        if self.scale:
            scale = 1./float(1 << ((8 * audio.getsampwidth()) - 1))
            signal_array *= scale
        return signal_array

    def median_filter(self,signal):
        N = self.n_sample
        filted_signal = np.zeros([1,N+self.M])
        expanded_sound =np.concatenate((np.zeros(self.M),signal,np.zeros(self.M)),axis=0)
        for i in range(0,N):
            filted_signal[0,i + int((self.M - 1)/2)]=statistics.median(expanded_sound[i:i + self.M - 1])
        filted_signal = filted_signal[0][int((self.M - 1)/2):N+int((self.M - 1)/2)]
        return filted_signal

    def split(self,signal):
        out=[]
        t=0
        th = self.th_p*max(signal)
        cut_cycle = self.cut_cycle*len(signal)
        for i in signal:
            if abs(i)>th and t<cut_cycle:
                out.append(i)
                t+=1
            elif abs(i)>th and t>cut_cycle:
                t=0
        self.n_sample=len(out)
        return out

    def write_audio(self,signal,name):
        data = np.array([signal], dtype=np.int16)
        with wave.open('out/'+name, 'w') as f:
            # Ses dosyasının özelliklerini ayarlayın
            f.setnchannels(1)  # Tek kanal (mono)
            f.setsampwidth(2)  # 16 bit veri türü
            f.setframerate(self.fs)  # 44.1kHz örnekleme hızı
            # Verileri ses dosyasına yazdırın
            f.writeframes(data)

    def framing(self,signal):
        framed_signal = []
        n_list=np.floor(self.NFFT/(1 + 2*self.framing_overlap)-1)  
        n_frame = int(np.ceil(len(signal)/n_list))
        for i in range(0,n_frame):
            out=[]  
            if i==0:
                out = signal[0:ceil(n_list*(self.framing_overlap+i+1))]
            else:
                out = signal[ceil(n_list*(i-self.framing_overlap)):ceil(n_list*(self.framing_overlap+i+1))]
            framed_signal.append(out)
        return framed_signal

    def windowing(self,signal):
        windowed_signal = []
        for framing_signal in signal:
            out = np.zeros(len(framing_signal))
            for i in range(0,len(framing_signal)):
                if self.windowing_type=='Hamming':
                    wn = 0.54-0.46*np.cos(2*np.pi*i/(len(framing_signal)-1))
                elif self.windowing_type=='Hanning':
                    wn = 0.5*(1-np.cos(2*np.pi*i/(len(framing_signal)-1)))
                elif self.windowing_type=='Blackman':
                    wn = 0.42 - 0.5*np.cos(2*np.pi*i/(len(framing_signal)-1))+0.08*np.cos(4*np.pi*i/(len(framing_signal)-1))                               
                out[i] = framing_signal[i]*wn
            windowed_signal.append(out)
        return windowed_signal

    def pow_fft(self,signal):
        fft_signal=[]
        fft_freq=[]
        for windowed_signal in signal:
            mag_fft = np.absolute(np.fft.rfft(windowed_signal,self.NFFT))
            fft_signal.append((1.0 / self.NFFT) * ((mag_fft) ** 2))
            fft_freq.append(np.fft.rfftfreq(len(windowed_signal), d=1. / self.fs))
        return fft_signal,fft_freq
    
    def pre_emphasis(self,signal):
        emphasized_signal = []
        emphasized_signal = np.append(signal[0], signal[1:] - self.pre_emphasis_coefficient * signal[:-1])
        return emphasized_signal

    def filter_bank(self,signal):
        M = self.nfilt
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, M+2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((self.NFFT + 1) * hz_points / self.fs)
        fbank = np.zeros((M, int(np.floor(self.NFFT / 2 + 1))))
        for m in range(1, M + 1):
            f_m_minus = int(bin[m - 1]) # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        ## Her bir filtre,fft ile çarpılacak
        filter_banks = np.dot(signal, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB
        return filter_banks

    def dct(self,signal): 
        return dct(signal, type=2, axis=1, norm='ortho')[:, 1 : (self.num_ceps + 1)]
        
    def lifting(self,signal,cep_lifter = 22):
        (nframes, ncoeff) = signal.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        signal *= lift
        return signal

    def run(self):
        base_signal = self.read()
        splitted_signal = self.split(base_signal) 
        filted_signal = self.median_filter(splitted_signal)
        emphasized_signal = self.pre_emphasis(filted_signal)
        framed_signal = self.framing(emphasized_signal)
        windowed_signal = self.windowing(framed_signal)
        fft_signal,fft_freq = self.pow_fft(windowed_signal)
        filt_banked_signal = self.filter_bank(fft_signal)
        dct_signal = self.dct(filt_banked_signal)
        lifted_signal = self.lifting(dct_signal)
        return(lifted_signal.T)

    def record_audio(self,
            filename="test",
            FRAMES_PER_BUFFER = 3200,
            CHANNELS = 1,
            RATE = 44100,
            seconds=float):

        filename = "test/{}.wav".format(filename)
        FORMAT = pyaudio.paInt16
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=44100,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER
        )

        print("Start recording...")

        frames = []
        for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
            data = stream.read(FRAMES_PER_BUFFER)
            frames.append(data)

        print("recording stopped")
        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
