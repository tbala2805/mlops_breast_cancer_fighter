import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



path = "data/raw/data.csv"
data = pd.read_csv(path)

print(data.head(20))
print(data.info())


## Preprocessing a bit
data.drop(columns=['id', "Unnamed: 32"], inplace=True)

# printing the column to list
print(data.columns.to_list())


# printing the number of Malignant and Benign
data_diagnosis = data['diagnosis'].value_counts().reset_index()
print(data_diagnosis)


# # show the difference between the number of Malignant & benign
# fig = px.pie(data_diagnosis, values='count', names='diagnosis', title='the number of Malignant and Benign')
#
# fig.show()


# training a model,
X = data.drop('diagnosis', axis = 1)
y = data['diagnosis']


# scaling the features to max value 1 and min value of 0
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# encoding the target from categorical to numerical
encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, shuffle=True, random_state=42
)


print(X_train.shape,y_train.shape)
print(X_test.shape, y_test.shape)

print(type(X_train))
print(type(y_train))
#
# # ANN Model
# # buidling ANN model with input layer and two hidden layer and one output layer and activation function
# # is relue and in out put layer is sigmoid
# model = Sequential(
#     [
#         Dense(32, activation='relu', input_dim=30),
#         Dense(16, activation='relu'),
#         Dense(8, activation='relu'),
#         Dense(1, activation='sigmoid')
#     ]
# )
#
#
# # compileing and add optimizer to a ANN network
# model.compile(
#     optimizer = 'adam',
#     loss = 'binary_crossentropy',
#     metrics = ['accuracy']
# )
#
# print(model.summary())
#
#
# # training a model with 60 epochs
#
# history = model.fit(X_train, y_train, epochs=60, validation_split = 0.2)
#
#
#
# tr_acc = history.history['accuracy']
# tr_loss = history.history['loss']
# val_acc = history.history['val_accuracy']
# val_loss = history.history['val_loss']
#
# epochs = [i+1 for i in range(len(tr_acc))]
#
# plt.figure(figsize=(30, 10))
# plt.subplot(1, 2, 1)
# plt.plot(epochs, tr_loss, 'r', label='Train Loss')
# plt.plot(epochs, val_loss, 'g', label='Valid Loss')
# plt.title('Loss')
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs, tr_acc, 'r', label='Train Accuracy')
# plt.plot(epochs, val_acc, 'g', label='Valid Accuracy')
# plt.title('Accuracy')
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
#
# plt.tight_layout()
# plt.show()
#
#
#
# # adding confusion metrics
# y_pred = model.predict(X_test)
# y_pred = (y_pred > 0.5)
#
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True)
# plt.show()
#
# # report
# name = ['Benign', 'Malignant']
# classification_rep = classification_report(
#     y_test, y_pred, target_names=name
# )
# print("\nClassification Report:")
# print(classification_rep)
#
