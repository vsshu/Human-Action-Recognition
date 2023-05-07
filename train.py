# -*- coding: utf-8 -*-
"""
>> Trains and tests tuned RNN, LSTM and GRU models by Kfold
>> Prints loss, accuracy, precision, recall, F1, misclassifications
>> Plots confusion matrix & loss and accuracy curve per fold
"""

import numpy as np
import tensorflow as tf
import random

from keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
from scikeras.wrappers import KerasClassifier

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(8)


def load_data(dropout_rate=0.1, weight_constraint=2.0):
    # Load the tensor data and labels from the .npy file
    tensor_data = np.load('tensors_k_tt5w9.npy')
    print(tensor_data.shape)
    labels = np.load('labels_tt5.npy')
    return tensor_data, labels


def rnn(dropout_rate=0.1, weight_constraint=4.0):
    print('RNN')
    # Define the RNN model architecture
    model = tf.keras.Sequential()
    model.add(SimpleRNN(124, input_shape=(5,52), kernel_constraint=MaxNorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(5, activation='softmax'))

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001) #momentum=0.4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model

def lstm(dropout_rate=0.2, weight_constraint=1.0):
    print('LSTM')
    # Define the LSTM model architecture
    model = tf.keras.Sequential()
    model.add(LSTM(64, input_shape=(5,52), kernel_constraint=MaxNorm(weight_constraint), kernel_regularizer=l2(0.01)))
    model.add(Dropout(dropout_rate)) 
    model.add(BatchNormalization())
    model.add(Dense(5, activation='softmax'))

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model

def gru(dropout_rate=0.1, weight_constraint=1.0):
    print('GRU')
    # Define the GRU model architecture
    model = tf.keras.Sequential()
    model.add(GRU(64, input_shape=(5,52), kernel_constraint=MaxNorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(5, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model

def train_64(model, X_train, X_val, y_train, y_val, fold):
    # Configurations for training RNN + GRU
    batch_size = 64
    early_stop = EarlyStopping(monitor='val_loss', patience=3)

    # Train the model on the training set for this fold
    history = model.fit(X_train, y_train, epochs=60, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stop])

    return model, history

def train_32(model, X_train, X_val, y_train, y_val, fold):
    # Configurations for training LSTM 
    batch_size = 32
    early_stop = EarlyStopping(monitor='val_loss', patience=3)

    # Train the model on the training set for this fold
    history = model.fit(X_train, y_train, epochs= 50, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stop])

    return model, history

def evaluate_model(model, X_val, y_val, history, fold, video_ids):
    batch_size = 16
    # Evaluate the model on the test set for this fold
    loss, accuracy = model.evaluate(X_val, y_val, batch_size=batch_size)

    # Generate predictions for the test set and the confusion matrix
    y_pred = model.predict(X_val)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    
    # Track down misclassifications
    misclassified = [(video_ids[i], y_true[i], y_pred[i]) for i in range(len(video_ids)) if y_true[i] != y_pred[i]]
    
    # Calculate f1, precision, and recall scores 
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    return loss, accuracy, cm, f1, precision, recall, misclassified


def cross_validate(dl, tensor_data, labels, video_ids):
    # Define the K-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True)

    # Define lists to store evaluation results
    test_loss_list = []
    test_acc_list = []
    f1_list = []
    precision_list = []
    recall_list = []
    overall_cm = np.zeros((5, 5), dtype=int)
    # misclassified_list = []

    # Train and evaluate the model on each fold
    fold = 1
    for train, test in kfold.split(tensor_data, labels):
        X_train, X_val = tensor_data[train], tensor_data[test]
        y_train, y_val = labels[train], labels[test]
        video_ids_val = video_ids[test] # video IDs for the test set

        
        # Define, compile, train, and evaluate the model on this fold
        if dl == 'RNN':
            model = rnn()
            trained_model, history = train_64(model, X_train, X_val, y_train, y_val, fold)
            loss, accuracy, cm, f1, precision, recall, miss = evaluate_model(trained_model, X_val,  y_val, history, fold, video_ids_val)
        elif dl == 'LSTM':
            model = lstm()
            trained_model, history = train_32(model, X_train, X_val, y_train, y_val, fold)
            loss, accuracy, cm, f1, precision, recall, miss = evaluate_model(trained_model, X_val,  y_val, history, fold, video_ids_val)
        elif dl == 'GRU':
            model = gru()
            trained_model, history = train_64(model, X_train, X_val, y_train, y_val, fold)
            loss, accuracy, cm, f1, precision, recall, miss = evaluate_model(trained_model, X_val,  y_val, history, fold, video_ids_val)

        # Append evaluation results to lists
        test_loss_list.append(loss)
        test_acc_list.append(accuracy)
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
        overall_cm += cm
        # misclassified_list.extend(miss)

        # #Print the evaluation results per fold
        # print(f'Fold {fold} Results:')
        # print(f'Test Loss: {loss:.4f}')
        # print(f'Test Accuracy: {accuracy:.4f}')
        # print(f'F1 Score: {f1:.4f}')
        # print(f'Precision: {precision:.4f}')
        # print(f'Recall: {recall:.4f}')
        # print(f'Confusion Matrix:\n{cm}')
        
        # # Plot the learning curves per fold
        # train_loss = history.history['loss']
        # val_loss = history.history['val_loss']
        # train_acc = history.history['accuracy']
        # val_acc = history.history['val_accuracy']
        # epochs = range(1, len(train_loss) + 1)
        
        # # Plot training and validation loss
        # plt.plot(epochs, train_loss, 'bo', color='purple', label='Training loss')
        # plt.plot(epochs, val_loss, 'b', color='red', label='Validation loss')
        # plt.title(f'Fold {fold} Learning Curves')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()
        
        # # Plot training and validation accuracy
        # plt.plot(epochs, train_acc, 'bo', color='purple', label='Training acc')
        # plt.plot(epochs, val_acc, 'b', color='red', label='Validation acc')
        # plt.title(f'Fold {fold} Learning Curves')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend()
        # plt.show()
        

        fold += 1
    
    # Plot the confusion matrix as a heatmap
    avg_cm = overall_cm / (fold-1)
    class_names = ['falling down', 'headache', 'chest pain', 'back pain', 'vomiting']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_cm.astype(int), annot=True, 
                cmap='RdPu', fmt='g', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix {dl}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
    # print(sorted(misclassified_list))
    print(f'Overall Results {dl}:')
    print(f'Average Test Loss: {np.mean(test_loss_list):.4f} ± {np.std(test_loss_list):.4f}')
    print(f'Average Test Accuracy: {np.mean(test_acc_list):.4f} ± {np.std(test_acc_list):.4f}')
    print(f'Average F1 Score: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}')
    print(f'Average Precision: {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}')
    print(f'Average Recall: {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}')


###########################################################################

tensor_data, labels = load_data()
print(labels.shape)
vid_ids = np.arange(1, 3476)
cross_validate('GRU', tensor_data, labels, vid_ids)
#print(tensor_data.shape)

