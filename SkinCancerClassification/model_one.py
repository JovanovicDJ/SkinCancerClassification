import numpy as np
import pandas as pd
import tensorflow as tf

import os
from glob import glob
from PIL import Image

import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

SIZE = 32
CLASSES = 7

if __name__ == '__main__':

    # Reading metadata
    df = pd.read_csv('HAM10000\\HAM10000_metadata.csv')

    # Adding labels
    le = LabelEncoder()
    le.fit(df['dx'])
    LabelEncoder()
    df['label'] = le.transform(df["dx"])
    print(df.sample(20))

    # Creating RGB arrays from images and adding them to dataframe
    image_path = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join('HAM10000\\', '*', '*.jpg'))}
    df['path'] = df['image_id'].map(image_path.get)
    df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE, SIZE))))

    print(df['dx'].value_counts())

    # Preparing data for neural network (normalization and data split for training and testing)
    X = np.asarray(df['image'].tolist())
    X = X / 255.
    Y = df['label']
    Y_cat = to_categorical(Y, num_classes=CLASSES)
    x_train, x_test, y_train, y_test = train_test_split(X, Y_cat, test_size=0.3, random_state=42)

    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation="relu", input_shape=(SIZE, SIZE, 3)))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())

    model.add(Dense(16))
    model.add(Dense(7, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

    batch_size = 16
    epochs = 25

    history = model.fit(
        x_train, y_train,  # data, labels
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=2)

    score = model.evaluate(x_test, y_test)
    print('Test accuracy:', score[1])

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(12, 6))
    plt.title('The confusion matrix of model on test set')
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='viridis')
    plt.show()
