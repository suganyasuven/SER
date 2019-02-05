import glob

import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
import librosa
import librosa.display
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from constants import *

import cv2
import numpy as np

import warnings

warnings.filterwarnings('ignore')


def mel_scaled_spectograms(wave, sr):
    stft = np.abs(librosa.stft(wave, n_fft=2048, hop_length=512))
    mel_spec = librosa.feature.melspectrogram(y=wave, sr=sr, S=stft)
    mel_spec_db = librosa.power_to_db(mel_spec)
    print(mel_spec_db.shape)
    return mel_spec_db


def get_dataset(input_length=131000):
    labels = np.empty(0)
    for fn in glob.glob(os.path.join(DATA_AUDIO_DIR, "*.wav")):
        label = fn.split('/')[1].split('_')[0]
        labels = np.append(labels, label)
        wave, sr = librosa.load(fn)
        wave = wave[1300:6 * sr]
        if len(wave) > input_length:
            max_offset = len(wave) - input_length
            offset = np.random.randint(max_offset)
            wave = wave[offset:(input_length + offset)]
        else:
            if input_length > len(wave):
                max_offset = input_length - len(wave)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            wave = np.pad(wave, (offset, input_length - len(wave) - offset), "constant")
        mel_spec = mel_scaled_spectograms(wave, sr)
        print(mel_spec.shape)
        librosa.display.specshow(mel_spec, sr=sr, hop_length=512, y_axis='mel', x_axis='time')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('SPECTROGRAM' + '/' + fn.split('/')[1].split('.')[0] + '.png', bbox_inches='tight', pad_inches=0.0)


def remove_background():
    for fn in glob.glob(os.path.join(DATA_DIR, "*.png")):
        img = cv2.imread(fn)
        ## (1) Convert to gray, and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        ## (2) Morph-op to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

        ## (3) Find the max-area contour
        cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv2.contourArea)[-1]

        ## (4) Crop and save it
        x, y, w, h = cv2.boundingRect(cnt)
        dst = img[y:y + h, x:x + w]
        cv2.imwrite('SPEC_IMAGES/' + fn.split('/')[1] + '.png', dst)

if __name__ == '__main__':
    # get_dataset(131000)
    remove_background()
