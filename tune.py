# -*- coding: utf-8 -*-
"""
>> Hyperparameter tuning using GridSearch
>> n_jobs set to -1 to run in parallel 

"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout, BatchNormalization
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.constraints import MaxNorm
from keras.optimizers import RMSprop, Adam, SGD
import random
import winsound
random.seed(8)


def load_data():
    tensor_data = np.load('tensors_v5.npy')
    labels = np.load('labels_v5.npy')
    return tensor_data, labels

def create_model(neurons = 64, dropout_rate=0.1, weight_constraint=4.0):
    model = tf.keras.Sequential()
    
    #model.add(SimpleRNN(neurons, input_shape=(10,52), kernel_constraint=MaxNorm(weight_constraint)))
    #model.add(LSTM(neurons, input_shape=(10,52), kernel_constraint=MaxNorm(weight_constraint)))
    model.add(GRU(neurons, input_shape=(10,52), kernel_constraint=MaxNorm(weight_constraint)))

    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(5, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

opt = SGD(learning_rate=0.001, momentum=0.0)
model = KerasClassifier(model=create_model, verbose=0,
                        epochs=60, batch_size=64,
                        optimizer=opt)

param_grid = {
    'model__neurons': [8, 16, 32, 64, 124],
    'model__dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'model__weight_constraint': [1.0, 2.0, 3.0, 0.4],
    'optimizer__momentum' : [0.0, 0.2, 0.4],
    'optimizer': ['SGD', 'RMSprop', 'Adam'],
    'optimizer__learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'batch_size': [8, 16, 32, 64],
    'epochs': [1, 5, 10, 15, 20, 30, 40, 50, 60]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)

X, Y = load_data()

grid_result = grid.fit(X, Y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for mean, stdev, params in zip(grid_result.cv_results_['mean_test_score'],
                               grid_result.cv_results_['std_test_score'],
                               grid_result.cv_results_['params']):
    print(f"{mean} ({stdev}) with: {params}")

###########################################################################


winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)