#improving the fashion classifier by using convolutions
#CNN
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist= keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels)= fashion_mnist.load_data()
train_images= train_images.reshape(60000,28,28,1)   #initial shape- 60000,28,28
test_images= test_images.reshape(10000,28,28,1)    #initial shape- 10000,28,28

#covert values b/w 0-1
train_images= train_images/255.0
test_images= test_images/255.0

#define the layers in model
#here (3,3) in conv is filter matrix size, and 64 is total no. of filters so from 1 image 64 new images are generated corresponding to each filter
#here (2,2) maxpooling means (2,2) size matrices are considered and from each max value pixel is chosen
model= keras.Sequential([keras.layers.Conv2D(64, (3,3), activation= 'relu', input_shape=(28,28,1)), keras.layers.MaxPooling2D(2,2), keras.layers.Conv2D(64, (3,3), activation= 'relu'), keras.layers.MaxPooling2D(2,2), keras.layers.Flatten(), keras.layers.Dense(128, activation= tf.nn.relu), keras.layers.Dense(10, activation= tf.nn.softmax)])

#compile the model to define loss and optimizer functions
model.compile(optimizer= tf.compat.v1.train.AdamOptimizer(), loss= 'sparse_categorical_crossentropy')

#train the data
model.fit(train_images, train_labels,epochs=5)

print(model.summary())    #inspection of all layers

#evaluate the data.. returns loss (100-loss%)%= accuracy%
model.evaluate(test_images, test_labels)


#VISULAIZING THE CONVOLUTIONS AND POOLING
(3,4) matrix of images. 1st col gives images 1,2,3 after 1st layer
f, axarr= plt.subplots(3,4)
#images of show at indices 0,23,28
IMAGE1= 0
IMAGE2= 23
IMAGE3= 28
conv_number= 1   #1 of 64  (meaning we did it for 64 filters)

#type and shape of layers' output
layer_outputs= [layer.output for layer in model.layers]
activation_model= keras.models.Model(inputs= model.input, outputs= layer_outputs)  #model.input is shape and type of input to our model

#we wish to see output of image1,2,3 after 1st 4 layers
for x in range(4):
	#f1 is output of image1 after layer x. shape of f1 is (1,26,26,64) after 1st conv
	#when an image of (28,28,1) is passed through 1st conv layer, it produces output of shape (1,26,26,64) meaning 64 images of size (26,26) each 
	#here conv_number=1 refers to image after filter 1 is used
	f1= activation_model.predict(test_images[IMAGE1].reshape(1,28,28,1))[x]
	axarr[0,x].imshow(f1[0, :, :, conv_number], cmap= 'inferno')
	axarr[0,x].grid(False)

	f2= activation_model.predict(test_images[IMAGE2].reshape(1,28,28,1))[x]
	axarr[1,x].imshow(f2[0, :, :, conv_number], cmap= 'inferno')
	axarr[1,x].grid(False)

	f3= activation_model.predict(test_images[IMAGE3].reshape(1,28,28,1))[x]
	axarr[2,x].imshow(f3[0, :, :, conv_number], cmap= 'inferno')
	axarr[2,x].grid(False)
plt.show()