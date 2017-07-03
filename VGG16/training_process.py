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
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from PIL import ImageFile

# load data
x_train = np.load('./x_train.npy')
y_train = np.load('./y_train.npy') 
x_test = np.load('./x_test.npy')
y_test = np.load('./y_test.npy')

# print('x_train shape:', x_train.shape)
# print('y_train shape:', y_train.shape)
# print('x_test shape:', x_test.shape)
# print('y_test shape:', y_test.shape)

# build model
img_rows, img_cols, img_channel = 224, 224, 3

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))

x_model = base_model.output
x_model = GlobalAveragePooling2D(name='globalaveragepooling2d')(x_model) # a global spatial average pooling layer

x_model = Dense(1024, activation='relu',name='fc1_Dense')(x_model)
x_model = Dropout(0.5, name='dropout_1')(x_model)

x_model = Dense(256, activation='relu',name='fc2_Dense')(x_model)
x_model = Dropout(0.5, name='dropout_2')(x_model)
predictions = Dense(1, activation='sigmoid',name='output_layer')(x_model)

model = Model(inputs=base_model.input, outputs=predictions) # this is the model we will train
# model.summary()

# serialize model to JSON
model_json = model.to_json()
with open('./model.json', 'w') as json_file:
    json_file.write(model_json)

# # print the model structure
# for i, layer in enumerate(model.layers):
#     print(i, layer.name)
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
checkpointer = ModelCheckpoint(filepath= './cp_weights.hdf5', verbose=1, monitor='val_acc',save_best_only=True, save_weights_only=True)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch = x_train.shape[0],
                    epochs=nb_epoch,
                    validation_data = (x_test, y_test),
                    callbacks=[checkpointer])




