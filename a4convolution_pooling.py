import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
#import cv2
from scipy import misc

#image to play with
i= misc.ascent()

#draw the image i
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)   #to convert pixel matrix to image
plt.show()

#the image is stored as numpy array so we can copy that image by just copying that array
i_trans= np.copy(i)
size_x= i_trans.shape[0]
size_y= i_trans.shape[1]

#this filter detects edges nicely
#it creates a convolution that only passes through sharp edges and
#straight lines
#experiment with different values for fun effects
filter= [[0,1,0], [1,-4,1], [0,1,0]]

#filter= [[-1,-2,-1], [0,0,0], [1,2,1]]
#filter= [[-1,0,1], [-2,0,2], [-1,0,1]]

#if all the digits in the filter don't add up to 0 or 1,
#you should do a weight to get it to do so
#for eg, if your weights are 1,1,1 1,2,1 1,1,1
#they add up to 10, so you would set a weight of 0.1 if you want to normalize them to 1
weight=1


#now create a convolution. iterate over the image, leaving 1 pixel margin (from left, right, top and bottom)
#and multiply each neighbour of current pixel by the value defined in the filter
#then we'll multiply the result by the weight, and then ensure the result is in the range 0-255

for x in range(1, size_x-1):
	for y in range(1, size_y-1):
		convolution =0
		convolution= convolution + (i[x-1][y-1]*filter[0][0])
		convolution= convolution + (i[x][y-1]*filter[1][0])
		convolution= convolution + (i[x+1][y-1]*filter[2][0])
		convolution= convolution + (i[x-1][y]*filter[0][1])
		convolution= convolution + (i[x][y]*filter[1][1])
		convolution= convolution + (i[x+1][y]*filter[2][1])
		convolution= convolution + (i[x-1][y+1]*filter[0][2])
		convolution= convolution + (i[x][y+1]*filter[1][2])
		convolution= convolution + (i[x+1][y+1]*filter[2][2])
		if convolution<0:
			convolution=0
		if convolution>255:
			convolution=255
		i_trans[x,y]= convolution

#now plot the image, note the size of axis--- they are 512 by 512
plt.gray()
plt.grid(False)
plt.imshow(i_trans)
#plt.axis('off')
plt.show()
		

#POOLING  (2,2)

new_x= int(size_x/2)
new_y= int(size_y/2)
newImage= np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
	for y in range(0, size_y, 2):
		pixels= []
		pixels.append(i_trans[x,y])
		pixels.append(i_trans[x+1,y])
		pixels.append(i_trans[x,y+1])
		pixels.append(i_trans[x+1,y+1])
		newImage[int(x/2), int(y/2)]= max(pixels)

#plot the image. note the size of axes-- now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
#plt.axis('off')
plt.show()

#result--> after pooling the features extracted by convolution are maintained
#but image gets compressed