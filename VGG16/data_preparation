# coding: utf-8
import os
import numpy as np
import pandas as pd
from keras.preprocessing import image
from PIL import ImageFile

# seting
img_size = 224

# get train data
train_label = pd.read_csv("../../data/train_labels.csv")
img_path = "../../data/train/"

file_paths = []
y = []
for i in range(len(train_label)):
    file_paths.append( img_path + str(train_label.iloc[i][0]) +'.jpg' )
    y.append(train_label.iloc[i][1])
y = np.array(y)

x = []
for i, img_path in enumerate(file_paths):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img = image.img_to_array(img)   
    x.append(img)
x = np.array(x)


# get test data
test_no = pd.read_csv("../../data/sample_submission.csv")
test_img_path = "../../data/test/"

test_file_paths = []
test_img_nos = []
for i in range(len(test_no)):
    test_file_paths.append( test_img_path + str(int(test_no.iloc[i][0])) +'.jpg' )
    test_img_nos.append(int(test_no.iloc[i][0]))
test_img_nos = np.array(test_img_nos)

test = []
for i, img_path in enumerate(test_file_paths):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img = image.img_to_array(img)   
    test.append(img)
test = np.array(test)

test = test.astype('float32')
test /= 255


# data shuffle
random_index = np.random.permutation(len(y))
x_shuffle = []
y_shuffle = []
for i in range(len(y)):
    x_shuffle.append(x[random_index[i]])
    y_shuffle.append(y[random_index[i]])
    
x = np.array(x_shuffle) 
y = np.array(y_shuffle)


# data split
val_split_num = int(round(0.2*len(y)))
x_train = x[val_split_num:]
y_train = y[val_split_num:]
x_test = x[:val_split_num]
y_test = y[:val_split_num]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)
print('test shape:', test.shape)


# save data
np.save('./x_train.npy',x_train)
np.save('./y_train.npy',y_train)
np.save('./x_test.npy',x_test)
np.save('./y_test.npy',y_test)
np.save('./test_img_nos',test_img_nos)
np.save('./test',test)




