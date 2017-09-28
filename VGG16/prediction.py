import numpy as np
import pandas as pd
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

# load json and create model
json_file = open('./model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('./cp_weights.hdf5')

# # build model
# img_rows, img_cols, img_channel = 224, 224, 3

# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))

# x_model = base_model.output
# x_model = GlobalAveragePooling2D(name='globalaveragepooling2d')(x_model) # a global spatial average pooling layer

# x_model = Dense(1024, activation='relu',name='fc1_Dense')(x_model)
# x_model = Dropout(0.5, name='dropout_1')(x_model)

# x_model = Dense(256, activation='relu',name='fc2_Dense')(x_model)
# x_model = Dropout(0.5, name='dropout_2')(x_model)
# predictions = Dense(1, activation='sigmoid',name='output_layer')(x_model)

# model = Model(inputs=base_model.input, outputs=predictions) # this is the model we will train

# # load weights into new model
# model.load_weights('./cp_weights.hdf5')

# get data
test_images = np.load('./test.npy')
test_nos = np.load('./test_img_nos.npy')
print('test_images shape:', test_images.shape)

# make a prediction
predictions = model.predict(test_images)

# write results into csv
sample_submission = pd.read_csv('../../data/sample_submission.csv')

for i, no in enumerate(test_nos):
    sample_submission.loc[sample_submission['name'] == no, 'invasive'] = predictions[i]

sample_submission.to_csv('./submition_with_1_epoch.csv', index=False)




