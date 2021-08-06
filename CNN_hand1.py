from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np

# seed for reproducing same results
seed = 785
np.random.seed(seed)

# load dataset
dataset = np.loadtxt("img/A_Z Handwritten Data.csv", delimiter=',')

# split into input and output variables
X = dataset[:,0:784]
Y = dataset[:,0]

# split the data into training (50%) and testing (50%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.50, random_state=seed)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

num_classes = Y_test.shape[1]


# # create model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# # Compile model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)


# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(28, 28, 1), name='conv1'))
# model.add(MaxPooling2D(pool_size=(2, 2,),name='pool1'))
# model.add(Dropout(0.2,name='dropout1'))
# model.add(Conv2D(filters=32, kernel_size=2, strides=1, padding='same', activation='relu',name='conv2'))
# model.add(MaxPooling2D(pool_size=(2, 2),name='pool2'))
# model.add(Flatten(name='flat'))
# model.add(Dropout(0.1,name='dropout2'))
# model.add(Dense(128,activation='relu',name='dense'))
# model.add(Dense(26,activation='softmax',name='res'))

# model.summary()

import tensorflow.keras as ks
checkpoint = ks.callbacks.ModelCheckpoint('model.h5',save_best_only=True)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#training..
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=19, batch_size=1024, callbacks=[checkpoint], verbose=1)

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')


# # Final evaluation of the model
# scores = model.evaluate(X_test,Y_test, verbose=0)
# print("CNN Error: %.2f%%" % (100-scores[1]*100))

# model.save('weights.model')