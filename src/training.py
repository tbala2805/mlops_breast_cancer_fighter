import pandas as pd
import os
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report



# ANN Model
# buidling ANN model with input layer and two hidden layer and one output layer and activation function
# is relue and in out put layer is sigmoid
def initializing_model():
    model = Sequential(
        [
            Dense(32, activation='relu', input_dim=30),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ]
    )


    # compileing and add optimizer to a ANN network
    model.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    )

    return model

def fitting_model(model, X, y, model_report_path, target_classes):


    X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, shuffle=True, random_state=42
    )
    # training a model with 60 epochs

    history = model.fit(X_train, y_train, epochs=60, validation_split = 0.2)



    tr_acc = history.history['accuracy']
    tr_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    epochs = [i+1 for i in range(len(tr_acc))]

    plt.figure(figsize=(30, 10))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, tr_loss, 'r', label='Train Loss')
    plt.plot(epochs, val_loss, 'g', label='Valid Loss')
    plt.title('Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, tr_acc, 'r', label='Train Accuracy')
    plt.plot(epochs, val_acc, 'g', label='Valid Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(model_report_path,'accuracy_loss_report.png' ))



    # adding confusion metrics
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True)
    plt.savefig(os.path.join(model_report_path,'confusion_matrix.png' ))


    # report
    name = ['Benign', 'Malignant']
    classification_rep = classification_report(
        y_test, y_pred, target_names=name
    )
    print("\nClassification Report:")
    print(classification_rep)

    return model


def save_model(model, path):
    """
    function to save a model

    :param model:
    :param path:
    :return:
    """
    model.save(os.path.join(path, 'my_model'))
    print("model is saved")




