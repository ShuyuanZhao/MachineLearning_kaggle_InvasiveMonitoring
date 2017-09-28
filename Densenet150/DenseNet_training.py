import numpy as np

from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from densenet121 import densenet121_model


model = densenet121_model(img_rows=128, img_cols=128, color_type=3, num_classes=1000)
# model_json = model.to_json()
# with open('./model.json', 'w') as json_file:
#     json_file.write(model_json)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])


# load data
x_train = np.load('./x_train.npy')
y_train = np.load('./y_train.npy') 
x_test = np.load('./x_test.npy')
y_test = np.load('./y_test.npy')
# set train Generator
datagen = ImageDataGenerator(rotation_range=30,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
datagen.fit(x_train)


# trainning process
nb_epoch = 1
batch_size = 32
checkpointer = ModelCheckpoint(filepath= './cp_weights.hdf5', verbose=1, monitor='val_acc',save_best_only=True, save_weights_only=True)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch = x_train.shape[0],
                    epochs=nb_epoch,
                    validation_data = (x_test, y_test),
                    callbacks=[checkpointer])





