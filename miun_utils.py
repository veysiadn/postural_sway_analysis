"""
This module contains utility functions for the embedded deployment.

"""
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
import numpy as np
import pathlib
import seaborn as sns
# Using GPU
import os
import scipy.io as scpy


__all__ = ["use_gpu", "fix_seed", "dense_model", "train_dense_model", 
           "get_performance_metrics", "get_sklearn_metrics", "split_data", "plot_history", "plot_confusion_matrix"]


def use_gpu()-> None:
    """
    Use GPU for training.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as error:
            # Memory growth must be set before GPUs have been initialized
            print(error)
    elif cpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            logical_cpus= tf.config.list_logical_devices('CPU')
            print(len(cpus), "Physical CPU,", len(logical_cpus), "Logical CPU")
        except RuntimeError as error:
            # Memory growth must be set before GPUs have been initialized
            print(error)

def fix_seed(seed: int) -> None:
    """
    Fix seed for reproducibility.
    """
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def dense_model(input_shape: tuple, output_shape: int, layer_1 : int, layer_2: int):
    """
    Create a dense model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(layer_1, activation='relu'),
        tf.keras.layers.Dense(layer_2, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
    return model

def train_dense_model(model: tf.keras.Model, x_train: np.array, y_train: np.array, 
                      x_val: np.array, y_val: np.array, epochs: int, 
                      batch_size: int,learning_rate:float) -> tf.keras.callbacks.History:
    """
    Train a dense model.
    """
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
    return history

def get_performance_metrics(model: tf.keras.Model,history: tf.keras.callbacks.History, x_test: np.array, y_test: np.array) -> None:
    """
    Get performance metrics for a dense model.
    """
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    ## Print the performance metrics: training loss, training accuracy, evaluation loss, evaluation accuracy
    print(f"Training accuracy: {history.history['accuracy'][-1]}")
    print(f"Training loss: {history.history['loss'][-1]}")
    print(f"Validation accuracy: {history.history['val_accuracy'][-1]}")
    print(f"Validation loss: {history.history['val_loss'][-1]}")
    print(f"Test accuracy: {test_acc}")
    print(f"Test loss: {test_loss}")

def get_sklearn_metrics(model: tf.keras.Model, x_test: np.array, y_test: np.array) -> None:
    """
    Get sklearn metrics for a dense model.
    """
    from sklearn.metrics import classification_report
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    
def split_data(x: np.array, y: np.array, test_size: float, val_size: float, random_seed: int) -> tuple:
    """
    Split data into training, validation and test sets.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=random_seed)
    return x_train, x_val, x_test, y_train, y_val, y_test

def plot_history(history: tf.keras.callbacks.History) -> None:
    """
    Plot the training and validation accuracy and loss.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def plot_confusion_matrix(model, x_test, y_test, class_names):
    """
    Plot the confusion matrix.
    """
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 16},cmap='Blues')
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Truth', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

def plot_roc_curve(model, x_test, y_test, class_names):
    """
    Plot the ROC curve.
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    y_pred = tf.keras.utils.to_categorical(y_pred)
    y_test = tf.keras.utils.to_categorical(y_test)
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_test[:, i], y_pred[:, i])
        auc = roc_auc_score(y_test[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, label=f'{class_names[i]} (area = {auc:0.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
import torch.nn as nn
import torch
class FCN(nn.Module):
    "Defines a connected network"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.ReLU6
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
    def train(self, x, y, epochs, lr, loss_fn, optimizer):
        "Trains the model"
        self.train()
        optimizer = optimizer(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.forward(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
        return self
    def predict(self, x):
        "Predicts the output"
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(x)
        return y_pred
    def evaluate(self, x, y, loss_fn):
        "Evaluates the model"
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(x)
            loss = loss_fn(y_pred, y)
        return loss.item()
    def plot_training_results(self, x, y, loss_fn):
        "Plots the training results"
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(x)
            loss = loss_fn(y_pred, y)
        plt.figure(figsize=(10, 8))
        plt.plot(x, y, label='Truth')
        plt.plot(x, y_pred, label='Prediction')
        plt.title(f"Loss: {loss.item():.2f}")
        plt.legend()
        plt.show()

# def automl(x_data: np.array, y_data: np.array):
#     from pycaret.classification import *
#     s = setup(x_data, target = y_data)
#     best_model = compare_models(exclude = ['catboost'])