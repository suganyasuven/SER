import keras
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.metrics import confusion_matrix

from preprocess import *

# Second dimension of the feature is dim2
feature_dim_2 = 40

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()
print(X_train.shape)
#(227, 20, 19)
print(X_test.shape)

# # Feature dimension
feature_dim_1 = 26
channel = 1
num_classes = 5

# Reshaping to perform 2D convolution
X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
print(X_train.shape)
# (227, 20, 19 1)
X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)
print(X_test.shape)


def create_model():
    model = Sequential()
    model.add(Conv2D(512, kernel_size=(2, 2), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.20))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


# def predict(filepath, model):
#     sample = wav2mfcc(filepath)
#     sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
#     return get_labels()[0][
#             np.argmax(model.predict(sample_reshaped))
#     ]


model = create_model()
model.fit(X_train, y_train, batch_size=32, epochs=200, validation_data=(X_test, y_test))


# predicting from the model

predict = model.predict(X_test, batch_size=50)
print(predict)

emotions = ['neutral', 'happy', 'sad', 'anger']
# predicted emotions from the test set
y_predicted = np.argmax(predict, 1)
print(y_predicted)
# print(test_y.shape[0])

predicted_emo = []
for i in range(0, y_test.shape[0]):
    emo = emotions[y_predicted[i]]
    predicted_emo.append(emo)
    # print(predicted_emo)

actual_emo = []
y_actual = np.argmax(y_test, 1)
for i in range(0, y_test.shape[0]):
    emo = emotions[y_actual[i]]
    actual_emo.append(emo)

    # generate the confusion matrix
cm = confusion_matrix(actual_emo, predicted_emo)
print(cm)
index = ['neutral', 'happy', 'sad', 'anger']
columns = ['neutral', 'happy', 'sad', 'anger']
cm_df = pd.DataFrame(cm, index, columns)
plt.figure(figsize=(10, 6))
sns.heatmap(cm_df, annot=True)
