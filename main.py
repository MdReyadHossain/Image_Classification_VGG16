import tensorflow as tf
from keras.layers import Input
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from glob import glob
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

# path of datasets folder

Directory = r"E:\My Files\Reyad\Study\Coding\Python\Img_Classification_VGG16\animals"
ImageSize = 224 # images will resize in 224 x 224
BatchSize = 64 # this amount of images will train/test simultaneously

# datasets will be divided into two part: 1. for train and 2. for validation

# data augmentation start......
# train datasets preprocessing
TrainDatagent = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255, # rescaling for faster calculation for each image
    zoom_range=0.2, # every image will zoom by 20%
    horizontal_flip=True, # every image will horizontally flip everytime
    validation_split=0.1 # split all images 90% for the train datasets and 10% for the validation datasets
)

# validation datasets preprocessing
ValidationDatagent = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.1
)

TrainGenerator = TrainDatagent.flow_from_directory(
    Directory,
    target_size=(ImageSize, ImageSize),
    batch_size=BatchSize,
    subset='training'
)

ValidationGenerator = ValidationDatagent.flow_from_directory(
    Directory,
    target_size=(ImageSize, ImageSize),
    batch_size=BatchSize,
    subset='validation'
)
# data augmentation done.......

# VGG16...........
ImageSize = [224, 224]
# all weights of vgg16 download, where all classes in imagenet won't be download(include_top=False) for custom classes
vgg = VGG16(input_shape=ImageSize+[3], weights='imagenet', include_top=False)
print(vgg.output)

# every layer of vgg16 will be false
for layers in vgg.layers:
    layers.trainable = False

folder = glob(r"E:\My Files\Reyad\Study\Coding\Python\Img_Classification_VGG16\animals\*")
print("Folder(s):", len(folder)) # how many datasets(folders) in this path

x = Flatten()(vgg.output)
Prediction = Dense(len(folder), activation='softmax')(x) # add all neurons with Dense
ImageModel = Model(inputs=vgg.inputs, outputs=Prediction)
ImageModel.summary()

ImageModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = ImageModel.fit(
    TrainGenerator,
    steps_per_epoch=len(TrainGenerator),
    epochs=15,
    validation_data=ValidationGenerator,
    validation_steps=len(ValidationGenerator)
)

# prediction for a random image....
ImgPredict = tf.keras.utils.load_img(r"E:\My Files\Reyad\Study\Coding\Python\Img_Classification_VGG16\predict\1500566326.jpg", target_size=(128, 128))
ImgPredict = tf.keras.utils.img_to_array(ImgPredict) # Sample Image will convert to an array
ImgPredict = np.expand_dims(ImgPredict, axis=0) # expand the dimension of the array of this image

Result = ImageModel.predict(ImgPredict) # model will predict the sample image

print(Result)
if Result[0][0] > Result[0][1]:
    prediction = "cat"

elif Result[0][0] < Result[0][1]:
    prediction = "dog"

else:
    prediction = "Not recognized"

print("Result :", prediction)