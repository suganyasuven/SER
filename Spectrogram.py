import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
import librosa
import librosa.display
import numpy as np
import pandas as pd
import random
from scipy.fftpack import fft
from scipy.signal import get_window
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')


def mel_scaled_spectograms(wave, sr):
    stft = np.abs(librosa.stft(wave, n_fft=2048, hop_length=512))
    mel_spec = librosa.feature.melspectrogram(y=wave, sr=sr, S=stft)
    mel_spec_db = librosa.power_to_db(mel_spec)
    print(mel_spec_db.shape)
    return mel_spec_db


def load_audio_file(file_path, input_length=131000):
    data, sr = librosa.load(file_path, duration=6)
    data = data[:96000]
    print(sr)
    print(len(data))
    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length + offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
    print(len(data))
    data = mel_scaled_spectograms(data, sr)
    n = data.shape[0]
    librosa.display.specshow(data, sr=sr, hop_length=512, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig('sp.png')
    plt.show()
    return data


if __name__ == '__main__':
    data = load_audio_file('utterances/1_03b10Wb.wav')
    print(data.shape)
