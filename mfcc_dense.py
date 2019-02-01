import glob
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Dense, Input, Dropout, BatchNormalization, Convolution2D, MaxPooling2D, GlobalMaxPool2D
from keras import optimizers, losses, activations, models
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

sampleRate = 16000


def extract_feature(file_name):
    x, sr = librosa.load(file_name)
    x = x[::3]
    stft = np.abs(librosa.stft(x))
    mfcc = librosa.feature.mfcc(x, sr=16000, S=stft, n_mfcc=26, hop_length=int(0.010 * sr), n_fft=int(0.025 * sr))
    # print(mfcc.shape)
    return mfcc


def parse_audio_files(parent_dir, sub_dirs, file_ext="*.wav"):
    mfcc_vectors = []
    max_len=11
    features, labels = np.empty((0, 26)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                mfccs= extract_feature(fn)
                if max_len > mfccs.shape[1]:
                    pad_width = max_len - mfccs.shape[1]
                    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

                # Else cutoff the remaining parts
                else:
                    mfccs = mfccs[:, :max_len]

            except Exception as e:
                print("Error encountered while parsing file: ", fn, e)
                continue
            mfcc_vectors.append(mfccs)
            labels = np.append(labels, fn.split('/')[2].split('_')[0])
    return np.array(mfcc_vectors), np.array(labels, dtype=np.int)


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
np.save('mfcc_features', features)
# one hot encoding labels
# labels = one_hot_encode(labels)
labels = to_categorical(labels, 4)
np.save('lab', labels)

X = np.load('mfcc_features.npy')
y = np.load('lab.npy')
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)
print(train_x.shape)


train_x = train_x.reshape(train_x.shape[0], 26, 11, 1)
# # print(train_x.shape)
# # # (227, 20, 19 1)


def create_model():
    nclass = 4
    inp = Input(shape=(26, 11, 1))
    norm_inp = BatchNormalization()(inp)
    img_1 = Convolution2D(16, kernel_size=(2, 2), activation=activations.relu)(norm_inp)
    img_1 = Convolution2D(16, kernel_size=(2, 2), activation=activations.relu)(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution2D(32, kernel_size=2, activation=activations.relu)(img_1)
    img_1 = Convolution2D(32, kernel_size=2, activation=activations.relu)(img_1)
    img_1 = GlobalMaxPool2D()(img_1)
    img_1 = Dropout(rate=0.1)(img_1)

    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(img_1))
    dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam()

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    return model
    # create the model


model = create_model()
# train the model
history = model.fit(train_x, train_y, epochs=100, batch_size=50)

# predicting from the model
predict = model.predict(test_x, batch_size=50)

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
