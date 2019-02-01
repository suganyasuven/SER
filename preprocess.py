import os
import glob

import librosa
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def wav2mfcc(file_path, max_len=40):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    # wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=26, hop_length=int(0.010 * sr), n_fft=int(0.025 * sr))

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
    sessions = os.listdir(main_dir)
    print(sessions)
    # Init mfcc vectors
    mfcc_vectors = []
    labels = np.empty(0)
    i = 0
    while i < len(sessions):
        for label in (os.listdir(main_dir + '/' + (sessions[i]))):
            for fn in glob.glob(os.path.join(main_dir, sessions[i], label, "*.wav")):
                try:
                    mfcc = wav2mfcc(fn, 40)
                    mfcc_vectors.append(mfcc)
                except Exception as e:
                    print("Error encountered while parsing file: ", fn, e)
                    continue
                labels = np.append(labels, fn.split('/')[3].split('_')[0])
        i+=1
    np.save('features', mfcc_vectors)
    labels = to_categorical(labels, 4)
    np.save('label', labels)

    X = np.load('features.npy')
    y = np.load('label.npy')

    print(X.shape)
    assert X.shape[0] == len(y)
    return train_test_split(X, y, test_size=0.3, random_state=42)
