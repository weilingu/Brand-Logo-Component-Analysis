# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:33:52 2019

@author: emily.gu
"""

from mnist import MNIST

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

'''
This script trains a Convoluted Neural Network Model 
- The model uses emnist data that contains letters
- The model is based on a datacamp CNN tutorial:
    https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python  
- The accuracy rate based on test dataset is about 94%
'''
# create path to the emnist dataset:
emnist_data_path="[path to mnist dataset]"
# create path to save the model architecture
cnn_model_path='[path to save the model]'

emnist_data = MNIST(path=emnist_data_path+"train", return_type='numpy')
emnist_data.select_emnist('letters')
x_train, y_train = emnist_data.load_training()

emnist_data = MNIST(path=emnist_data_path+"test", return_type='numpy')
emnist_data.select_emnist('letters')
x_test, y_test = emnist_data.load_testing()


train_X=x_train.reshape(-1,28,28,1)
test_X=x_test.reshape(-1,28,28,1)

train_X=train_X.astype('float32')
train_X=train_X/255.0
test_X=test_X.astype('float32')
test_X=test_X/255.0

# convert the y labels to categorical data
y_train_softmax=to_categorical(y_train)
y_test_softmax=to_categorical(y_test)

# Split the training dataset into train and validation datasets
train_X,valid_X, train_label, valid_label = train_test_split(train_X,y_train_softmax, test_size=0.2, random_state=13)

# train the CNN
batch_sizes=128 # can change the batch_size to 256
epoch=20
num_classes=27

letter_reg_model=Sequential()
letter_reg_model.add(Conv2D(32,kernel_size=(3,3),activation='linear',input_shape=(28,28,1),padding='same'))
letter_reg_model.add(LeakyReLU(alpha=0.1))
letter_reg_model.add(MaxPooling2D((2,2),padding='same'))
letter_reg_model.add(Dropout(0.25))

letter_reg_model.add(Conv2D(64,kernel_size=(3,3),activation='linear',padding='same'))
letter_reg_model.add(LeakyReLU(alpha=0.1))
letter_reg_model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
letter_reg_model.add(Dropout(0.25))

letter_reg_model.add(Conv2D(128,kernel_size=(3,3),activation='linear',padding='same'))
letter_reg_model.add(LeakyReLU(alpha=0.1))
letter_reg_model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
letter_reg_model.add(Dropout(0.4))

letter_reg_model.add(Flatten())
letter_reg_model.add(Dense(128,activation='linear'))
letter_reg_model.add(LeakyReLU(alpha=0.1))
letter_reg_model.add(Dropout(0.3))
letter_reg_model.add(Dense(num_classes, activation='softmax'))

letter_reg_model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
letter_reg_model.summary()

# Train the model:
letter_reg_model_train=letter_reg_model.fit(train_X,train_label,batch_size=batch_sizes, epochs=epoch,verbose=1,validation_data=(valid_X,valid_label))
#evaluate the model
test_eval = letter_reg_model.evaluate(test_X, y_test_softmax, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

# Save the model architecture
letter_reg_model.save(cnn_model_path+"letter_recog_model.h5py")
letter_reg_model.save_weights(cnn_model_path+"letter_recog_model.h5")

with open(cnn_model_path+"CNN_letter_model.json", 'w') as f:
    f.write(letter_reg_model.to_json())

'''
# letter-number lookup
letter = { 0: 'non', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
                    11: 'k',
                    12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v',
                    23: 'w',
                    24: 'x', 25: 'y', 26: 'z'}
'''


