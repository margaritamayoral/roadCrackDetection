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

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from import_data import im_chan, im_width, im_height, X_train, Y_train, n_test
from import_data import testing_files_img, testing_files_mask
from IoU_metrics import mean_iou
#from training import X_train


## Get and resize test images

X_test = np.zeros((n_test, im_height, im_width, im_chan), dtype=np.uint8)
Y_test = np.zeros((n_test, im_height, im_width, 1), dtype=np.bool)

sizes_test = []

print('========  Getting and resizing tests images ...  =========')

sys.stdout.flush()
for n, id_ in enumerate(testing_files_mask):
    this_img = testing_files_img[n]
    this_mask = testing_files_mask[n]
#    test_img = load_img(this_img, grayscale=True)
    test_img = load_img(this_img)

#    test_mask = load_img(this_mask, grayscale=True)
    test_mask = load_img(this_mask)

    x = img_to_array(test_img)
    sizes_test.append([x.shape[0], x.shape[1]])
    x = resize(x, (1024, 1024, im_chan), mode='constant', preserve_range=True)
    X_test[n] = x
    y = img_to_array(test_mask)
    y = resize(y, (1024, 1024, 1), mode='constant', preserve_range=True)
    Y_test[n] = y

print('==============   Done resizing test images!! =============')

print('============   Predict on train, validation and test  =======')
# Inference
# Predict on train, val and test
model = load_model('../python_files/model-streetscan-signals-1024x1024-2.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

print('================    Threshold  predictions================')

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)


print('========================    Create list of upsampled test masks ================')
preds_test_upsampled = []
for i in tnrange(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))


preds_test_upsampled[0].shape
print("preds_test[i] = ", preds_test[i])
print("========================")
print("preds_train[i] = ", preds_train[i])
print("========================")
print("preds_test_t = ", preds_test_t)
print("========================")
print("preds_test_upsampled = ", preds_test_upsampled[i])


print('======= Showing train, mask and predicted images =======')
# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
#print(ix)
#print(preds_train[ix])
ax4 = plt.figure()
#plt.imshow(np.dstack((X_train[ix],X_train[ix],X_train[ix])))
plt.imshow(X_train[ix])
plt.title("images to predict ")
plt.savefig('../output/train_images_to_predict_1.png')
plt.show()
tmp = np.squeeze(Y_train[ix]).astype(np.float32)
ax5 = plt.figure()
plt.title("Masks to predict in train images ")
#plt.imshow(np.dstack((tmp,tmp,tmp)))
plt.imshow(tmp)
plt.savefig('../output/train_masks_to_predict_1.png')
plt.show()
tmp_t = np.squeeze(preds_train_t[ix]).astype(np.float32)
#plt.imshow(np.dstack((tmp_t, tmp_t, tmp_t)))
plt.imshow(tmp_t)
plt.title("Predictions in train images")
plt.savefig('../output/preds_train_images_1.png')
plt.show()


def plot_sample(X, y, preds, binary_preds, output_file, ixr=None):
    """This function help us to plot the results"""
    if ixr is None:
        #ixr = random.randint(0, len(y))
        ixr = random.randint(0, len(preds))

    has_mask = y[ixr].max() > 0

    fig, ax40 = plt.subplots(1, 4, figsize=(20, 10))
#    ax40[0].imshow(X[ixr, ..., 0], cmap='gray')
    ax40[0].imshow(X[ixr, ..., 0])
    if has_mask:
        ax40[0].contour(y[ixr].squeeze(), colors='k', levels=[0.5])
    ax40[0].set_title('Images Road')

    ax40[1].imshow(y[ixr].squeeze())
    ax40[1].set_title('Signs')

    ax40[2].imshow(preds[ixr].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax40[2].contour(y[ixr].squeeze(), colors='k', levels=[0.5])
    ax40[2].set_title("Signs predicted")

    ax40[3].imshow(binary_preds[ixr].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax40[3].contour(y[ixr].squeeze(), colors='k', levels=[0.5])
    ax40[3].set_title("Signs predicted binary")
#    plt.savefig('../output/images_results_testing_set_4.png')
    plt.savefig(output_file)

# Check if training data looks all right
#plot_sample(X_train, Y_train, preds_train, preds_train_t)

print('Checking if valid data looks all right')
# Check if valid data looks all right


plot_sample(X_train, Y_train, preds_train, preds_train_t, '../output/visualization_results_predictions_validation_images_1.png')

plot_sample(X_train, Y_train, preds_train, preds_train_t, '../output/visualization_results_predictions_validation_images_2.png')

plot_sample(X_train, Y_train, preds_train, preds_train_t, '../output/visualization_results_predictions_validation_images_3.png')

plot_sample(X_train, Y_train, preds_train, preds_train_t, '../output/visualization_results_predictions_validation_images_4.png')

plot_sample(X_train, Y_train, preds_train, preds_train_t, '../output/visualization_results_predictions_validation_images_5.png')

plot_sample(X_train, Y_train, preds_train, preds_train_t, '../output/visualization_results_predictions_validation_images_6.png')

plot_sample(X_train, Y_train, preds_train, preds_train_t, '../output/visualization_results_predictions_validation_images_7.png')

plot_sample(X_train, Y_train, preds_train, preds_train_t, '../output/visualization_results_predictions_validation_images_8.png')

plot_sample(X_train, Y_train, preds_train, preds_train_t, '../output/visualization_results_predictions_validation_images_9.png')

plot_sample(X_train, Y_train, preds_train, preds_train_t, '../output/visualization_results_predictions_validation_images_10.png')


print("Predictions on test set")


plot_sample(X_test, Y_test, preds_test, preds_test_t, '../output/visualization_results_predictions_on_test_images_1.png')

plot_sample(X_test, Y_test, preds_test, preds_test_t, '../output/visualization_results_predictions_on_test_images_2.png')

plot_sample(X_test, Y_test, preds_test, preds_test_t, "../output/visualization_results_predictions_on_test_images_3.png")

plot_sample(X_test, Y_test, preds_test, preds_test_t, '../output/visualization_results_predictions_on_test_images_4.png')

plot_sample(X_test, Y_test, preds_test, preds_test_t, '../output/visualization_results_predictions_on_test_images_5.png')

plot_sample(X_test, Y_test, preds_test, preds_test_t, '../output/visualization_results_predictions_on_test_images_6.png')

plot_sample(X_test, Y_test, preds_test, preds_test_t, '../output/visualization_results_predictions_on_test_images_7.png')

plot_sample(X_test, Y_test, preds_test, preds_test_t, '../output/visualization_results_predictions_on_test_images_8.png')

plot_sample(X_test, Y_test, preds_test, preds_test_t, '../output/visualization_results_predictions_on_test_images_9.png')

plot_sample(X_test, Y_test, preds_test, preds_test_t, '../output/visualization_results_predictions_on_test_images_20.png')

def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted or not
    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

#pred_dict = {fn[:-4]:RLenc(np.round(preds_test_upsampled[i])) for i,fn in tqdm_notebook(enumerate(test_ids))}

#sub = pd.DataFrame.from_dict(pred_dict, orient='index')
#sub.index.names = ['id']
#sub.columns = ['rle_mask']
#sub.to_csv('../output/results-3.csv')




