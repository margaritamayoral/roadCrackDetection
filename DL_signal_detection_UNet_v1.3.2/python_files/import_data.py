from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets

import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob


#import cv2

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
#from PIL import Image

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import config

# Setting connection parameters

#server = '/Volumes'  # /Volumes/bulk_dev/MA_Burlington/011/20190418_MA_Burlington_Undef_1 /
server = config.path['server']
store_loc = config.path['store_loc'] 
client = config.path['client']
vehicle = config.path['vehicle']
survey = config.path['survey']
session = config.path['session']
environment = config.path['environment']


#dir_training = 'Training_Data'
dir_training = config.dir_params['training'] 
dir_test = config.dir_params['test']
#dir_project = 'PCI'
#dir_project = 'test'
dir_project = config.dir_params['project']
#dir_zmap = 'slab_zmap'
#dir_img = 'slab_img'
dir_img = config.dir_params['img']
#dir_mask = 'slab_zmap_mask'  # /Volumes/bulk_dev/Training_Data/PCI/slab_zmap_mask/slab_zmap_mask2957.png
dir_mask = config.dir_params['mask']
training_size = config.dir_params['training_size']

# Setting some  image parameters

im_width = config.image_params['width']
im_height = config.image_params['height']
im_chan = config.image_params['channel']

#path2sess = os.path.join(server,store_loc,client,vehicle,survey,session)
path2sess = os.path.join(server,store_loc,environment)
print(path2sess)

#path2zmap = os.path.join(path2sess,dir_zmap)

path2img = os.path.join(path2sess,dir_project,dir_img)
path2mask = os.path.join(path2sess,dir_project,dir_mask)
print(path2img)
print(path2mask)
#path2img = os.path.join(path2sess,dir_project,dir_img,dir_training)
#path2masks = os.path.join(path2sess,dir_project,dir_mask,dir_training)
#path2imgtest = os.path.join(path2sess,dir_project,dir_img,dir_test)
#path2maskstest = os.path.join(path2sess,dir_project,dir_mask,dir_test)
#path2zmap = os.path.join(path2sess,dir_training,dir_project,dir_zmap)
#path2mask = os.path.join(path2sess, dir_mask)
#path2mask = os.path.join(path2sess,dir_training,dir_project,dir_mask)

#==============================================================================
print(' Getting all the files in the images directory')
img_files = [f for f in glob.iglob(path2img + "/*.png", recursive=True)]
#print(img_files)

mask_files = [f for f in glob.iglob(path2mask + "/*.png", recursive=True)]
#print(mask_files)

#==============================================================================
# use img_files as base to create the other file list, there is a 1-to-1
# correspondance bewteen all these files


random.seed(4)
random.shuffle(img_files)

random.seed(4)
random.shuffle(mask_files)

# =============================================================================
n = len(mask_files)
print("n = ", n)

indx1 = 0
indx2 = int(np.floor(n*training_size))


#training_files_zmap = zmap_files[indx1:indx2]
training_files_img = img_files[indx1:indx2]
training_files_mask = mask_files[indx1:indx2]

#testing_files_zmap = zmap_files[indx2+1:n-1]
testing_files_img = img_files[indx2+1:n-1]
testing_files_mask = mask_files[indx2+1:n-1]
#=============================================================================
#
## Exploring the data

n_train_explor_data = 3
#training_files_zmap_explor_data = training_files_zmap[:n_train_explor_data]
training_files_img_explor_data = training_files_img[:n_train_explor_data]
training_files_mask_explor_data = training_files_mask[:n_train_explor_data]
#n_train_zmap = len(training_files_zmap)
#n_test_zmap = len(testing_files_zmap)
n_train = len(training_files_mask)   ### or training_files_img
n_test = len(testing_files_mask)     ### or training_files_img
#print("these are the img ids with n=3: ", training_files_img_explor_data)
#print("these are the mask ids with n=3: ", training_files_mask_explor_data)


ax1 = plt.figure(1, figsize=(20, 10))
for j, img_name in enumerate(training_files_img_explor_data):
    print(j-1)
    this_img = training_files_img[j-1]
    this_mask = training_files_mask[j-1]
#    this_zmap = training_files_zmap[j-1]
    q = j + 1
    print("this is the image", training_files_img)
    img = load_img(this_img)
#    img_zmap = load_img(this_zmap)
    img_mask = load_img(this_mask)
    plt.subplot(1, 2*(1+len(training_files_img_explor_data)), q*2-1)
    plt.title(img_name)
    plt.imshow(img)
    plt.subplot(1, 2*(1+len(training_files_img_explor_data)), q*2)
#    plt.title('Image Zmap ' + img_name)
#    plt.imshow(img_zmap)
#    plt.subplot(1, 3*(1+len(training_files_img_explor_data)), q*3)
 #   plt.title('Image Mask ' + img_name)
    plt.imshow(img_mask)
plt.savefig('../output/images_masks_512x512_preview.png')


print('Number of train images: ', n_train)
print('Number of test images: ', n_test)
#print(training_files_img)
#print(training_files_mask)


# ##Get and resize train images and masks
#

X_train = np.zeros((n_train, im_height, im_width, im_chan), dtype=np.uint8)
Y_train = np.zeros((n_train, im_height, im_width, 1), dtype=np.bool)
print('Getting and resizing train images and masks')
sys.stdout.flush()

# Loading images and masks into arrays
n = len(mask_files)

for n, id_ in enumerate(training_files_mask):
    this_img = training_files_img[n]
    this_mask = training_files_mask[n]
    img = load_img(this_img)
#    img = load_img(this_img, grayscale=True)
    mask = load_img(this_mask)
#    mask = load_img(this_mask, grayscale=True)
    x = img_to_array(img)
    x = resize(x, (512, 512, im_chan), mode='constant', preserve_range=True,
            anti_aliasing=True)
    X_train[n] = x
    y = img_to_array(mask)
    y = resize(y, (512, 512, 1), mode='constant', preserve_range=True,
            anti_aliasing=True)
    Y_train[n] = y
print('==============  Done!   ======================')

# ### Check if training data looks all right

print("===============  Checking if the new training data is OK ===============")
ix = random.randint(0, len(training_files_mask))
print(ix)
print(training_files_mask[ix])
ax50 = plt.figure()
#plt.imshow(np.dstack((X_train[ix], X_train[ix], X_train[ix])))
plt.imshow(X_train[ix])
plt.title('Image Road ' + training_files_mask[ix])
plt.savefig('../output/resized_images_512x512_preview_x_1.png')
ax51 = plt.figure()
#tmp = np.squeeze(Y_train[ix]).astype(np.float32)
plt.title('Road Signs' + training_files_mask[ix])
#plt.imshow(np.dstack((tmp, tmp, tmp)))
plt.imshow(np.squeeze(Y_train[ix]))
#plt.show()
plt.savefig('../output/resized_images_512x512_preview_y_1.png')

# Visualize any random image along with the mask

print(' ===============  Visualizing any random image along the mask  =======')
in_new = random.randint(0, len(X_train))
has_mask = Y_train[ix].max() > 0  # sign indicator
fig, (ax54, ax55) = plt.subplots(1, 2, figsize=(20, 15))
ax54.imshow(X_train[ix, ..., 0])
#ax54.imshow(X_train[ix, ..., 0], interpolation='bilinear')
#ax54.imshow(X_train[ix, ..., 0], cmap='gray', interpolation='bilinear')

if has_mask: #if sign
    # draw a boundary (contour) in the original image separating signs
    ax54.contour(Y_train[ix].squeeze(), colors='k', linewidths=5, levels=[0.5])
ax54.set_title('Image Road ' + training_files_mask[ix])
ax55.imshow(Y_train[ix].squeeze(), interpolation='bilinear')
#ax55.imshow(Y_train[ix].squeeze(), cmap='gray', interpolation='bilinear')

ax55.set_title('Sign ')
plt.savefig('../output/previewing_random_training_512x512_images_masks_1.png')

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
