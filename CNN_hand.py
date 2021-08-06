import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#ignore warning messages 
import warnings
warnings.filterwarnings('ignore') 

sns.set()

dataset = pd.read_csv("img/A_Z Handwritten Data.csv").astype('float32')
dataset.rename(columns={'0':'label'}, inplace=True)

# Splite data the X - Our data , and y - the prdict label
X = dataset.drop('label',axis = 1)
y = dataset['label']

print("shape:",X.shape)
print("columns count:",len(X.iloc[1]))
print("784 = 28X28")

X.head()

from sklearn.utils import shuffle

X_shuffle = shuffle(X)

# plt.figure(figsize = (12,10))
# row, colums = 4, 4
# for i in range(16):  
#     plt.subplot(colums, row, i+1)
#     plt.imshow(X_shuffle.iloc[i].values.reshape(28,28),interpolation='nearest', cmap='Greys')
# plt.show()

# print("Amount of each labels")

# # Change label to alphabets
# alphabets_mapper = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'} 
# dataset_alphabets = dataset.copy()
# dataset['label'] = dataset['label'].map(alphabets_mapper)

# label_size = dataset.groupby('label').size()
# label_size.plot.barh(figsize=(10,10))
# plt.show()

# print("We have very low observations for I and F ")
# print("I count:", label_size['I'])
# print("F count:", label_size['F'])

print(y.unique()[1])
print(X.shape[1])

# splite the data
X_train, X_test, y_train, y_test = train_test_split(X,y)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# scale data
standard_scaler = MinMaxScaler()
standard_scaler.fit(X_train)

X_train = standard_scaler.transform(X_train)
X_test = standard_scaler.transform(X_test)

# train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# valid_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))


cls = Sequential()
#cls.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
# cls.add(MaxPooling2D(pool_size=(2, 2)))
# cls.add(Dropout(0.3))
# cls.add(Flatten())
# cls.add(Dense(128, activation='relu'))
# cls.add(Dense(len(y.unique()), activation='softmax'))

cls.add(Dense(25, input_dim = 784, activation='relu'))
cls.add(Dense(1, activation = 'softmax'))


cls.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#history = cls.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=18, batch_size=100, verbose=2)
history = cls.fit(train_data, epochs=10, validation_data=valid_data, batch_size=100, verbose=2)

scores = cls.evaluate(X_test,y_test, verbose=0)
print("CNN Score:",scores[1])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()