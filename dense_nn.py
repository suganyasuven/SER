import glob
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

sampleRate = 16000


def extract_feature(file_name):
    x, sample_rate = librosa.load(file_name)
    # x = librosa.effects.trim(y=x, top_db=10)
    stft = np.abs(librosa.stft(x))
    mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sampleRate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampleRate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(x, sr=sampleRate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sampleRate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(x),
                                              sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def parse_audio_files(parent_dir, sub_dirs, file_ext="*.wav"):
    features, labels = np.empty((0, 193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
            except Exception as e:
                print("Error encountered while parsing file: ", fn, e)
                continue
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            features = np.vstack([features, ext_features])
            labels = np.append(labels, fn.split('/')[2].split('_')[0])
    return np.array(features), np.array(labels, dtype=np.int)


# def one_hot_encode(labels):
#     n_labels = len(labels)
#     n_unique_labels = len(np.unique(labels))
#     one_hot_encode = np.zeros((n_labels, n_unique_labels))
#     one_hot_encode[np.arange(n_labels), labels] = 1
#     # one_hot_encode = np.delete(one_hot_encode, 0, axis=1)
#     return one_hot_encode


main_dir = 'dataset'
sub_dir = os.listdir(main_dir)
print("\ncollecting features and labels...")
print("\nthis will take some time...")
features, labels = parse_audio_files(main_dir, sub_dir)
print("done")
np.save('X', features)
# one hot encoding labels
# labels = one_hot_encode(labels)
labels = to_categorical(labels, 4)
np.save('y', labels)

X = np.load('X.npy')
y = np.load('y.npy')
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)

n_dim = train_x.shape[1]
n_classes = train_y.shape[1]
n_hidden_units_1 = n_dim
n_hidden_units_2 = 400  # approx n_dim * 2
n_hidden_units_3 = 200  # half of layer 2
n_hidden_units_4 = 100


def create_model(optimiser='adam', dropout_rate=0.2):
    model = Sequential()
    # layer 1
    model.add(Dense(193, input_dim=193, activation="relu", kernel_initializer="normal"))
    # layer 2
    model.add(Dense(400, activation="relu", kernel_initializer="normal"))
    model.add(Dropout(dropout_rate))
    # layer 3
    model.add(Dense(200, activation="relu", kernel_initializer="normal"))
    model.add(Dropout(dropout_rate))
    # layer4
    model.add(Dense(100, activation="relu", kernel_initializer="normal"))
    model.add(Dropout(dropout_rate))
    # output layer
    model.add(Dense(4, activation="softmax", kernel_initializer="normal"))
    # model compilation
    model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
    model.summary()
    return model

    # create the model


model = create_model()
# train the model
history = model.fit(train_x, train_y, epochs=100, batch_size=4)

# predicting from the model
predict = model.predict(test_x, batch_size=4)

emotions = ['sad', 'angry', 'happy', 'neutral']
# predicted emotions from the test set
y_pred = np.argmax(predict, 1)
# print(y_pred)
# print(test_y.shape[0])

predicted_emo = []
for i in range(0, test_y.shape[0]):
    emo = emotions[y_pred[i]]
    predicted_emo.append(emo)
    # print(predicted_emo)

actual_emo = []
y_true = np.argmax(test_y, 1)
for i in range(0, test_y.shape[0]):
    emo = emotions[y_true[i]]
    actual_emo.append(emo)

    # generate the confusion matrix
cm = confusion_matrix(actual_emo, predicted_emo)
print(cm)
index = ['sad', 'angry', 'happy', 'neutral']
columns = ['sad', 'angry', 'happy', 'neutral']
cm_df = pd.DataFrame(cm, index, columns)
plt.figure(figsize=(10, 6))
sns.heatmap(cm_df, annot=True)
