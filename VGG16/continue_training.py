import os
import numpy as np
import pandas as pd
import random
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout

def training_data_shuffle(x_train, y_train):
    random_index = np.random.permutation(len(y_train))
    x_shuffle = []
    y_shuffle = []
    for i in range(len(y_train)):
        x_shuffle.append(x_train[random_index[i]])
        y_shuffle.append(y_train[random_index[i]])
    x = np.array(x_shuffle)
    y = np.array(y_shuffle)
    
    return x, y    

# load data
x_train = np.load('./x_train.npy')
y_train = np.load('./y_train.npy') 
x_test = np.load('./x_test.npy')
y_test = np.load('./y_test.npy')
# data shuffle
(x_train, y_train) = training_data_shuffle(x_train, y_train) 


# load json and create model
json_file = open('./model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('./cp_weights.hdf5')


# frozen the first 15 layers
for layer in model.layers[:15]:
    layer.trainable = False
for layer in model.layers[15:]:
    layer.trainable = True

# compile the model
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

# set train Generator
datagen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
datagen.fit(x_train)


# trainning process
nb_epoch = 5
batch_size = 32

save_path = './cp_weights_2.hdf5'
checkpointer = ModelCheckpoint(filepath= save_path, verbose=1, monitor='val_acc',save_best_only=False, save_weights_only=True)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch = x_train.shape[0],
                epochs=1,
                validation_data = (x_test, y_test),
                callbacks=[checkpointer])


