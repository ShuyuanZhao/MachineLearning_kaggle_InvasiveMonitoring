
import os
import numpy as np
import pandas as pd

from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
# from PIL import ImageFile
from keras.applications.resnet50 import ResNet50

# img_rows, img_cols, img_channel = 224, 224, 3
# input_tensor_shape=(img_rows, img_cols, img_channel)
def build_ResNet50(input_tensor_shape):
    '''
    # reference 
        https://keras.io/applications/#vgg16
        https://www.tensorflow.org/api_docs/python/tf/contrib/keras/applications/ResNet50
    # model defination
        https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/contrib/keras/python/keras/applications/resnet50.py
        
    # Arguments
        include_top: whether to include the fully-connected layer at the top of the network.
     
    '''
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape= input_tensor_shape)
    
    x_model = base_model.output
    
    x_model = GlobalAveragePooling2D(name='globalaveragepooling2d')(x_model)
    
    x_model = Dense(1024, activation='relu',name='fc1_Dense')(x_model)
    x_model = Dropout(0.5, name='dropout_1')(x_model)
    
    x_model = Dense(256, activation='relu',name='fc2_Dense')(x_model)
    x_model = Dropout(0.5, name='dropout_2')(x_model)
    predictions = Dense(1, activation='sigmoid',name='output_layer')(x_model)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model
           

# model_save_path = './model.json'
def save_model_to_json(model,model_save_path):
    model_json = model.to_json()
    with open(model_save_path, 'w') as json_file:
        json_file.write(model_json)
    print('model saved')
    

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


# training process
# load data
x_train = np.load('./x_train.npy')
y_train = np.load('./y_train.npy') 
x_test = np.load('./x_test.npy')
y_test = np.load('./y_test.npy')

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# data shuffle
(x_train, y_train) = training_data_shuffle(x_train, y_train) 

# get model
img_rows, img_cols, img_channel = 224, 224, 3
input_tensor_shape=(img_rows, img_cols, img_channel)

model = build_ResNet50(input_tensor_shape)

model_save_path = './model.json'
save_model_to_json(model,model_save_path)


# for i, layer in enumerate(model.layers):
#     if i < 175:
#         print(i, layer.name)

# frozen the first 15 layers
for layer in model.layers[:175]:
    layer.trainable = False
for layer in model.layers[175:]:
    layer.trainable = True


# compile the model
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

# set train Generator
datagen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
datagen.fit(x_train)


# trainning process
nb_epoch = 1
batch_size = 32
checkpointer = ModelCheckpoint(filepath= './ResNet50_weights.hdf5', verbose=1, monitor='val_acc',save_best_only=True, save_weights_only=True)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch = x_train.shape[0],
                    epochs=nb_epoch,
                    validation_data = (x_test, y_test),
                    callbacks=[checkpointer])




