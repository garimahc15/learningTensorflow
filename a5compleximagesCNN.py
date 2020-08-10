#horse_human binary classification
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#model layers definition
#hhere instead of softmax, sigmoid function is used in last layer beacuse we're using 1 neuron in output layer
#for binary classification. i.e. if value~0 => class1, value~1 =>class2
model= keras.Sequential([keras.layers.Conv2D(16, (3,3), activation= 'relu', input_shape=(300,300,3)), keras.layers.MaxPooling2D(2,2), keras.layers.Conv2D(32, (3,3), activation= 'relu'), keras.layers.MaxPooling2D(2,2), keras.layers.Conv2D(64, (3,3), activation= 'relu'), keras.layers.MaxPooling2D(2,2), keras.layers.Flatten(), keras.layers.Dense(512, activation= tf.nn.relu), keras.layers.Dense(1, activation= tf.nn.sigmoid)])

#compile
#since we have binary classification, we'll use binary_crossentropy for better accuracy, categorical_crossentropy can also be used
#we CAN use adam optimizer as earlier but we're using RMSprop to adjust learning rate (lr) to experiment with performance
model.compile(loss= 'binary_crossentropy', optimizer= keras.optimizers.RMSprop(lr=0.001), metrics= ['acc'])

#data preprocessing
train_datagen= ImageDataGenerator(rescale=1./255) #pass rescale to normalize the data
#inside train_dir should be your directories that contain the respective class images
#resize the images AT RUNTIME to (300,300) to make them of uniform size. because neural network has to be given inputs of same sizes
#images will be loaded for training and validating in BATCHES instead of one-by-one
train_dir= "/home/garimachahar/learnTensorflow/horse-or-human"
validation_dir= train_dir= "/home/garimachahar/learnTensorflow/validation-horse-or-human"
train_generator= train_datagen.flow_from_directory(train_dir, target_size= (300,300), batch_size=64, class_mode='binary')
validation_generator= train_datagen.flow_from_directory(validation_dir, target_size= (300,300), batch_size=32, class_mode='binary')


#training
#no. of images in train_data= 1024, we're loading 128 at a time, so need 8 batches (1024=8*128)
# IIly validation data has 256 images so need 8 steps (batches) to load all images where batch size is 32 (32*8-256)
#verbose=2 specifies how much to display while training is going on. with verbose set to 2 we'll see a little less animation hiding the epoch progress
history= model.fit_generator(train_generator, steps_per_epoch=16, epochs=15, verbose=1)


#have to write code to predict