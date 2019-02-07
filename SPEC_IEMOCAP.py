import glob

import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
import librosa
import librosa.display

from constants import *

import numpy as np

import warnings

warnings.filterwarnings('ignore')


def mel_scaled_spectograms(wave, sr):
    stft = np.abs(librosa.stft(wave, n_fft=2048, hop_length=512))
    mel_spec = librosa.feature.melspectrogram(y=wave, sr=sr, S=stft)
    # mel_spec_db = librosa.power_to_db(mel_spec)
    return mel_spec


def get_dataset(input_length=131000):
    D_train = []
    D_test = []
    for fn in glob.glob(os.path.join('IEMOCAP', "*.wav")):
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
        label = fn.split('/')[1].split('_')[0]
        if 'Ses01' in fn:
            D_test.append((mel_spec, label))
        else:
            D_train.append((mel_spec, label))
    return D_train, D_test


train, test = get_dataset(131000)

X_train, y_train = zip(*train)
X_test, y_test = zip(*test)

X_train = np.array([x.reshape((128, 256, 1)) for x in X_train])
X_test = np.array([x.reshape((128, 256, 1)) for x in X_test])

y_train = np.array(keras.utils.to_categorical(y_train, 5))
y_test = np.array(keras.utils.to_categorical(y_test, 5))

model = Sequential()
input_shape = (128, 256, 1)

model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding='valid'))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding='valid'))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x=X_train, y=y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))

score = model.evaluate(x=X_test, y=y_test)

print('Test Loss', score[0])
print('Test Accuracy', score[1])
