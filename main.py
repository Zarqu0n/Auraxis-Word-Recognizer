from auraxis import Extraction
import tslearn.metrics
import numpy as np
import os
from termcolor import colored
import pyaudio
import wave



FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

class Model:
    def __init__(self,filename) -> None:
        self.filename = filename

    def run(self):
        model=Extraction(
            NFFT=512,
            filename=self.filename,
            scale=False,
            framig_overlap=0.3,
            pre_emphasis_coefficient=0.95,
            threshold_percentage = 0.01,
            cut_cycle=0.001,
            number_filt = 13,
            number_ceps = 12,
            windowing_type='Hamming'
        )

        mfcc = model.run()
        return mfcc


def compare(data,signals,names):
    dist = {}
    for k in range(0,len(signals)):
        signal = signals[k]
        c=0
        for i in range(0,12):
            c += tslearn.metrics.dtw(data[i], signal[i])
        dist[names[k]] = c
    return dist



def record():
    filename = "test/test.wav"
    FRAMES_PER_BUFFER = 3200
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    # starts recording
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=44100,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    print("start recording...")

    frames = []
    seconds = 3
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


if __name__ in '__main__':
    record()
    data = Model('test/test.wav')
    mfcc_data = data.run()
    """
    # Database güncelleme
    for i in os.listdir('assets'):
        data = Model('assets/'+i)
        mfcc_data = data.run()
        np.savetxt("database/{}.txt".format(i), mfcc_data)"""
        
    database = []
    names = []
    for i in os.listdir('database'):
        database.append(np.loadtxt("database/{}".format(i)))
        names.append(i.split(".wav.txt")[0])
    dict = compare(mfcc_data,database,names)
    items = list(dict.items())
    items.sort(key=lambda x: x[1])
    print(colored('Söylenmiş olabilecek sesler:','green'))
    for i in range(0,5):
        kelime = items[i+1]
        print(colored(kelime,'yellow'))