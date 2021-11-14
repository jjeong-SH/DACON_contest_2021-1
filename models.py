import pandas as pd 
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import utils
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Flatten, MaxPooling1D, Input, concatenate, Conv1D, BatchNormalization, GlobalAveragePooling1D, Dropout, Dense, Reshape, LSTM, SpatialDropout1D, AveragePooling1D
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model

'''
model7 was excluded due to poor performance
total 8 models trained for ensemble
'''


def build_model1(filter, l_rate, opti):
    ip = Input(shape=(num_time, num_features))

    y = Conv1D(64, 11, activation='relu')(ip)
    y = BatchNormalization()(y)
    y = MaxPooling1D()(y)

    y = Conv1D(64, 5, activation='relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling1D()(y)
    y = SpatialDropout1D(0.3)(y)

    y = GlobalAveragePooling1D()(y)
    out = Dense(num_classes, activation='softmax')(y)

    model1 = Model(ip, out)

    if opti == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)
    if opti == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=l_rate)

    model1.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model1.summary()

    return model1
  
    
def build_model2(l_rate, opti):
    ip = Input(shape=(num_time, num_features))

    y = Conv1D(64, 8, padding='same', activation="relu")(ip)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = Conv1D(64, 5, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)
    
    y = Conv1D(64, 3, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.4)(y)

    y = GlobalAveragePooling1D()(y)
    y = Flatten()(y)
    out = Dense(num_classes, activation='softmax')(y)

    model2 = Model(ip, out)
    model2.summary()

    # add load model code here to fine-tune

    if opti == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)
    if opti == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=l_rate)

    model2.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model2
  
    
def build_model3(filter, l_rate, opti):
    ip = Input(shape=(num_time, num_features))

    y = Conv1D(filter, 11, padding='same', activation='relu')(ip)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.3)(y)

    y = Conv1D(filter, 5, padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.3)(y)

    y = Conv1D(filter, 3, padding='same', activation='relu')(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.3)(y)
    
    y = GlobalAveragePooling1D()(y)
    y = Flatten()(y)
    out = Dense(num_classes, activation='softmax')(y)

    model3 = Model(ip, out)

    if opti == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)
    if opti == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=l_rate)

    model3.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model3.summary()

    return model3
  
    
def build_model4(l_rate, opti):
    ip = Input(shape=(num_time, num_features))

    y = Conv1D(64, 8, padding='same', activation="relu")(ip)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = Conv1D(64, 5, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = Conv1D(64, 3, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = GlobalAveragePooling1D()(y)
    y = Flatten()(y)
    out = Dense(num_classes, activation='softmax')(y)

    model4 = Model(ip, out)
    model4.summary()

    if opti == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)
    if opti == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=l_rate)

    model4.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model4
  
    
def build_model5(l_rate, opti):
    ip = Input(shape=(num_time, num_features))

    y = Conv1D(64, 11, padding='same', activation="relu")(ip)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)
    
    y = Conv1D(64, 5, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = Conv1D(64, 3, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = Conv1D(128, 11, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = Conv1D(128, 5, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = Conv1D(128, 3, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = GlobalAveragePooling1D()(y)
    out = Dense(num_classes, activation='softmax')(y)

    model4 = Model(ip, out)
    model4.summary()

    if opti == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)
    if opti == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=l_rate)

    model4.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model5
  
    
def build_model6(l_rate, opti):
    ip = Input(shape=(num_time, num_features))

    y = Conv1D(64, 8, padding='same', activation="relu")(ip)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = Conv1D(64, 5, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = Conv1D(64, 3, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = Conv1D(128, 8, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = Conv1D(128, 5, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = Conv1D(128, 3, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = Conv1D(128, 8, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = Conv1D(128, 5, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)
    
    y = Conv1D(128, 3, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = GlobalAveragePooling1D()(y)
    out = Dense(num_classes, activation='softmax')(y)

    model4 = Model(ip, out)
    model4.summary()

    if opti == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)
    if opti == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=l_rate)

    model4.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model6
  
    
def build_model8(l_rate, opti):
    ip = Input(shape=(num_time, num_features))

    y = Conv1D(64, 9, padding='same', activation="relu")(ip)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.3)(y)

    y = Conv1D(64, 5, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.4)(y)

    y = Conv1D(64, 3, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.5)(y)

    y = GlobalAveragePooling1D()(y)
    y = Flatten()(y)
    out = Dense(num_classes, activation='softmax')(y)

    model4 = Model(ip, out)
    model4.summary()

    if opti == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)
    if opti == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=l_rate)

    model4.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model8
  
    
def build_model9(l_rate, opti):
    ip = Input(shape=(num_time, num_features))

    y = Conv1D(32, 9, padding='same', activation="relu")(ip)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = Conv1D(32, 5, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = Conv1D(32, 3, padding='same', activation="relu")(y)
    y = BatchNormalization()(y)
    y = SpatialDropout1D(0.2)(y)

    y = GlobalAveragePooling1D()(y)
    y = Flatten()(y)
    out = Dense(num_classes, activation='softmax')(y)

    model4 = Model(ip, out)
    model4.summary()

    if opti == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)
    if opti == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=l_rate)

    model4.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model9
