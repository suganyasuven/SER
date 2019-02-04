import os
import glob

import librosa
import numpy as np
from keras.utils import to_categorical


def wav2mfcc(file_path, max_len=96):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    start = 8000
    end = 96000
    wave = wave[start:end]
    # spectrogram = librosa.feature.melspectrogram(y=wave, sr=16000,  n_fft=512, win_length=400,
    #                                              hop_length=160, power=2.0, n_mels=128)
    mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=26, hop_length=160, n_fft=512)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if max_len > mfcc.shape[1]:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc


# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_train_test():
    main_dir = 'IEMOCAP'
    # Init mfcc vectors
    X_train = []
    X_test = []
    Y_train = np.empty(0)
    Y_test = np.empty(0)
    for fn in glob.glob(os.path.join(main_dir, "*.wav")):
        try:
            mfcc = wav2mfcc(fn, 96)
            label = fn.split('/')[1].split('_')[0]
            if 'Ses01' in fn:
                X_test.append(mfcc)
                Y_test = np.append(Y_test, label)
            else:
                X_train.append(mfcc)
                Y_train = np.append(Y_train, label)
        except Exception as e:
            print("Error encountered while parsing file: ", fn, e)
            continue

    np.save('X_train', X_train)
    np.save('Y_train', Y_train)
    np.save('X_test', X_test)
    np.save('Y_test', Y_test)

    train_x = np.load('X_train.npy')
    test_x = np.load('X_test.npy')
    train_y = np.load('Y_train.npy')
    train_y = to_categorical(train_y, 5)
    test_y = np.load('Y_test.npy')
    test_y = to_categorical(test_y, 5)

    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    assert train_x.shape[0] == len(train_y)
    return train_x, test_x, train_y, test_y
