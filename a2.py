#CV for clothing classification

import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

fashion_mnist= keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels)= fashion_mnist.load_data()
train_images= train_images/255.0
test_images= test_images/255.0
model= keras.Sequential([keras.layers.Flatten(input_shape=(28,28)), keras.layers.Dense(128, activation= tf.nn.relu), keras.layers.Dense(10, activation= tf.nn.softmax)])
model.compile(optimizer= tf.compat.v1.train.AdamOptimizer(), loss= 'sparse_categorical_crossentropy') #or optimizer='adam'
model.fit(train_images, train_labels,epochs=5)
model.evaluate(test_images, test_labels)

#if you want training to stop after certain no. of epochs IF
# you're okay with the loss you got after that epoch

#define this class

class myCallback(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('loss')<0.4):
			print("\nLoss is low so cancelling training!")
			self.model.stop_training=True
mcallbacks= myCallback()
model.fit(train_images, train_labels, epochs=5, callbacks=[mcallbacks])
