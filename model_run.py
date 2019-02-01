import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix

from model_data import DataReader
from models import *


class MetricsHistory(Callback):
    def on_epoch_end(self, epoch, logs={}):
        file_logger.write([str(epoch),
                           str(logs['loss']),
                           str(logs['val_loss']),
                           str(logs['acc']),
                           str(logs['val_acc'])])


if __name__ == '__main__':
    model_name = 'm18'
    args = sys.argv
    if len(args) == 2:
        model_name = args[1].lower()
    print('Model selected:', model_name)

    model = None
    num_classes = 5
    if model_name == 'm3':
        model = m3(num_classes=num_classes)
    elif model_name == 'm5':
        model = m5(num_classes=num_classes)
    elif model_name == 'm11':
        model = m11(num_classes=num_classes)
    elif model_name == 'm18':
        model = m18(num_classes=num_classes)
    # elif model_name == 'm34':
    #     model = resnet_34(num_classes=num_classes)

    if model is None:
        exit('Please choose a valid model: [m3, m5, m11, m18, m34]')

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    data_reader = DataReader()
    x_tr, y_tr = data_reader.get_all_training_data()
    y_tr = to_categorical(y_tr, num_classes=num_classes)
    x_te, y_te = data_reader.get_all_testing_data()
    y_te = to_categorical(y_te, num_classes=num_classes)

    print('x_tr.shape =', x_tr.shape)
    print('y_tr.shape =', y_tr.shape)
    print('x_te.shape =', x_te.shape)
    print('y_te.shape =', y_te.shape)

    # if the accuracy does not increase over 10 epochs, we reduce the learning rate by half.
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001, verbose=1)
    metrics_history = MetricsHistory()
    batch_size = 3
    model.fit(x=x_tr,
              y=y_tr,
              batch_size=batch_size,
              epochs=100,
              verbose=1,
              shuffle=True,
              validation_data=(x_te, y_te))
              # callbacks=[metrics_history, reduce_lr])

    predict = model.predict(x_te, batch_size=3)
    print(predict)

    emotions = ['neutral', 'happy', 'sad', 'anger', 'frustration']
    # predicted emotions from the test set
    y_predicted = np.argmax(predict, 1)
    print(y_predicted)
    # print(test_y.shape[0])

    predicted_emo = []
    for i in range(0, y_te.shape[0]):
        emo = emotions[y_predicted[i]]
        predicted_emo.append(emo)
        # print(predicted_emo)

    actual_emo = []
    y_actual = np.argmax(y_te, 1)
    for i in range(0, y_te.shape[0]):
        emo = emotions[y_actual[i]]
        actual_emo.append(emo)

        # generate the confusion matrix
    cm = confusion_matrix(actual_emo, predicted_emo)
    print(cm)
    index = emotions
    columns = emotions
    cm_df = pd.DataFrame(cm, index, columns)
    plt.figure(figsize=
               (10, 6))
    sns.heatmap(cm_df, annot=True)
