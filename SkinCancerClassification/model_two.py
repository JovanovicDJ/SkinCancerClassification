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


if __name__ == '__main__':
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Reading already prepared data, images are loaded as 28x28x3 RGB arrays and they are labeled
    data = pd.read_csv('HAM10000/hmnist_28_28_RGB.csv')

    # Balancing dataset for more precision in results
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
