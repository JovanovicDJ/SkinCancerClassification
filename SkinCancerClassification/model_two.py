import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow as tf
import os
import warnings

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.utils import array_to_img
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def visualisePlots(X, Y, model, rows, columns):
    class_dicts = {
        0: 'nv',
        1: 'mel',
        2: 'bkl',
        3: 'bcc',
        4: 'akiec',
        5: 'vasc',
        6: 'df',
    }

    data = []
    target = []

    Y_pred = model.predict(X)
    Y_pred = np.array(list(map(lambda x: np.argmax(x), Y_pred)))

    for i in range(rows * columns):
        data.append(X[i])
        target.append(Y[i])

    width = 10
    height = 10
    fig = plt.figure(figsize=(10, 10))
    for i in range(columns * rows):
        temp_img = array_to_img(data[i])
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(temp_img)
        plt.xticks([])
        plt.yticks([])
        plt.title(str(class_dicts[target[i][0]]) + " || " + str(class_dicts[Y_pred[i]]))
    plt.show()


if __name__ == '__main__':
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=UserWarning)
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Reading already prepared data, images are loaded as 28x28x3 RGB arrays and they are labeled
    data = pd.read_csv('HAM10000/hmnist_28_28_RGB.csv')

    # Balancing dataset for more precision in results
    print('Preparing data...')
    data = data.sort_values('label')
    data = data.reset_index()

    index0 = data[data['label'] == 0].index.values
    index1 = data[data['label'] == 1].index.values
    index2 = data[data['label'] == 2].index.values
    index3 = data[data['label'] == 3].index.values
    index5 = data[data['label'] == 5].index.values
    index6 = data[data['label'] == 6].index.values

    df_index0 = data.iloc[int(min(index0)):int(max(index0) + 1)]
    df_index1 = data.iloc[int(min(index1)):int(max(index1) + 1)]
    df_index2 = data.iloc[int(min(index2)):int(max(index2) + 1)]
    df_index3 = data.iloc[int(min(index3)):int(max(index3) + 1)]
    df_index5 = data.iloc[int(min(index5)):int(max(index5) + 1)]
    df_index6 = data.iloc[int(min(index6)):int(max(index6) + 1)]

    df_index0 = df_index0.append([df_index0] * 18, ignore_index=True)
    df_index1 = df_index1.append([df_index1] * 12, ignore_index=True)
    df_index2 = df_index2.append([df_index2] * 5, ignore_index=True)
    df_index3 = df_index3.append([df_index3] * 52, ignore_index=True)
    df_index5 = df_index5.append([df_index5] * 43, ignore_index=True)
    df_index6 = df_index6.append([df_index6] * 5, ignore_index=True)

    # Merging balanced dataframes
    frames = [data, df_index0, df_index1, df_index2, df_index3, df_index5, df_index6]
    final_data = pd.concat(frames)
    final_data.drop('index', inplace=True, axis=1)
    final_data = final_data.sample(frac=1)
    new_data = final_data.iloc[:, :-1]
    labels = final_data.iloc[:, -1:]

    type_cancer = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
    counts = list(labels.value_counts())
    plt.figure(figsize=(8, 6))
    sns.barplot(x=type_cancer, y=counts)
    plt.show()

    # Preparing data for neural network (data split for training and testing)
    X = np.array(new_data)
    Y = np.array(labels)
    X = X.reshape(-1, 28, 28, 3)
    X = (X - np.mean(X)) / np.std(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=10,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode='nearest')
    train_datagen.fit(X_train)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen.fit(X_test)

    train_data = train_datagen.flow(X_train, Y_train, batch_size=64)
    test_data = test_datagen.flow(X_test, Y_test, batch_size=64)

    # Model
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), input_shape=(28, 28, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(7, activation='softmax'))

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=3,
                                                verbose=1)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print(model.summary())

    # Training
    history = model.fit(X_train,
                        Y_train,
                        validation_split=0.2,
                        batch_size=32,
                        epochs=10,
                        callbacks=[learning_rate_reduction])

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.figure(figsize=(8, 6))
    plt.title("Training and Validation accuracy of the model")
    plt.plot(acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend()

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 6))
    plt.title("Training and Validation loss of the model")
    plt.plot(loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.show()

    Y_true = np.array(Y_test)

    Y_pred = model.predict(X_test)
    Y_pred = np.array(list(map(lambda x: np.argmax(x), Y_pred)))

    cm1 = confusion_matrix(Y_true, Y_pred)
    plt.figure(figsize=(12, 6))
    plt.title('The confusion matrix of model on test set')
    sns.heatmap(cm1, annot=True, fmt='g', vmin=0, cmap='viridis')
    plt.show()

    visualisePlots(X_test, Y_test, model, 3, 3)

    label_mapping = {
        0: 'nv',
        1: 'mel',
        2: 'bkl',
        3: 'bcc',
        4: 'akiec',
        5: 'vasc',
        6: 'df'
    }

    classification_report_model = classification_report(Y_true, Y_pred, target_names=label_mapping.values())
    print(classification_report_model)

    model_acc_test = model.evaluate(X_test, Y_test, verbose=0)[1]
    print("Test Accuracy of model: {:.3f}%".format(model_acc_test * 100))
