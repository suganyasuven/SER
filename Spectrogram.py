import glob

import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from gimpfu import *
import librosa
import librosa.display
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from constants import *

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


def remove_background:
    BASE_PATH = 'SPECTROGRAM'
    FILE_EXTENSION = '.png'

    for idx, img in enumerate(gimp.image_list()):
        layer = img.layers[0]
        filename = BASE_PATH + str(idx) + FILE_EXTENSION
        gimp.pdb.gimp_by_color_select(layer, 'white', 12, 0, TRUE, 0, 0, 0)
        gimp.pdb.gimp_edit_clear(layer)
        gimp.pdb.plug_in_autocrop(img, layer)
        gimp.pdb.gimp_file_save(img, layer, filename, filename)

if __name__ == '__main__':
    get_dataset(131000)
