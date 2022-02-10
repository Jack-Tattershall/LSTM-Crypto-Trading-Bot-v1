import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, LSTM, BatchNormalization, Dense
import numpy as np

def model(x_len):

    model = Sequential()

    model.add(LSTM(128, input_shape=(x_len), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))

    model.summary()

    loss = 'sparse_categorical_crossentropy'
    optimiser = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    metrics = ['accuracy']
    
    model.complile(loss=loss,
                   optimizer=optimiser,
                   metrics=metrics)

    return model

