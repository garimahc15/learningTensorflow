#unzipping zip file to extract directores and plotting images
import os
import zipfile
import matplotlib.pyplot as plt

#contents of .zip are extracted to the directory /home/garimachahar/learnTensorflow/horse-or-human, which in turn
# each contain filtered-horses and filtered-humans subdirectories
local_zip= '/home/garimachahar/learnTensorflow/horse-or-human.zip'
zip_ref= zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/home/garimachahar/learnTensorflow/horse-or-human')
local_zip= '/home/garimachahar/learnTensorflow/validation-horse-or-human.zip'
zip_ref= zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/home/garimachahar/learnTensorflow/validation-horse-or-human')
zip_ref.close()

#directory with training horse pictures
train_horse_dir= os.path.join('/home/garimachahar/learnTensorflow/horse-or-human/horses')

#directory with training human pictures
train_human_dir= os.path.join('/home/garimachahar/learnTensorflow/horse-or-human/humans')

#now let's see what the filenames look like in the horses and humans training directories
train_horse_names= os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names= os.listdir(train_human_dir)
print(train_human_names[:10])

print('total training horse images:', len(train_horse_names))
print('total training human images:', len(train_human_names))


#take a look at pictures to see what they look like
import matplotlib.image as mpimg

#parameteres for our graph; we'll output images in a 4*4 configuration
nrows=4
ncols=4
pic_index=0 #index for iterating over images

#set up matplotlib fig, and size it to fit 4*4 pics
fig= plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index +=8
next_horse_pix = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
	#set up subplot; subplot indices start at 1
	sp= plt.subplot(nrows, ncols, i+1)
	sp.axis('Off') #don't show axes (or gridlines)

	img= mpimg.imread(img_path)  #img is numpy array with pixel value representation of image
	#print(img.shape)  #shape(300,300,4)
	plt.imshow(img)

plt.show()