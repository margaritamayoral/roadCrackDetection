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
from keras.utils import multi_gpu_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import U_Net

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from IoU_metrics import mean_iou

from import_data import get_files

from train_results_to_dbs import write_to_dbs
#=========================================================
#=                    Training                         ==
#=========================================================

def train(model_id):
	inputs, outputs = U_Net.build_model(model_id)	
	config = __import__("config_" +  str(model_id))

	X_train, Y_train, _, _, _ = get_files(model_id)
	## The model is compiled with "Adam" optimizer

	model = Model(inputs=[inputs], outputs=[outputs])

	## The binary crossentropy is used as loss function due that our variable is binary (exist or not exist)
	model = multi_gpu_model(model, gpus=2)
	model.compile(optimizer=config.training_params['optimizer'], loss=config.training_params['loss'], metrics=[mean_iou])
	model.summary()

	os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

	## Early stopping is used if the validation loss does not improve for 10 continues epochs
	earlystopper = EarlyStopping(patience=config.training_params['patience'], verbose=1)
	## Save the weights only if there is improvement in validation loss
	checkpointer = ModelCheckpoint(config.training_params['model_name'], verbose=1, save_best_only=True)

	results = model.fit(X_train, Y_train, validation_split=config.training_params['validation_split'], batch_size=config.training_params['batch_size'], epochs=config.training_params['epochs'], callbacks=[earlystopper, checkpointer])

	print(results.history)
	## Create the 4 arrays to write into the dbs ########

	val_loss_values = []
	val_mean_iou_values = []
	mean_iou_values = []
	loss_values =[]

	for e in range(config.training_params['epochs']):
		val_loss_values.append(results.history['val_loss'][e])
		val_mean_iou_values.append(results.history['val_mean_iou'][e])
		mean_iou_values.append(results.history['mean_iou'][e])
		loss_values.append(results.history['loss'][e])
	
	write_to_dbs(val_loss_values, val_mean_iou_values, mean_iou_values, loss_values, model_id)	
	
	print('=========  Plotting the loss-log learning curve  =======')
	##  Logarithmic loss which is related with cross-entropy measures the perfomrance of a classification model
	## where the prediction input is a probability value between 0 and 1. The goal of our machine learning models is to
	## minimize this value. A perfect model would have a loss-log of 0. Log loss increases as the predicted probability
	## diverges from the actual label


	ax57 = plt.figure(figsize=(8, 8))
	plt.title("Learning curve 512x512 images")
	plt.plot(results.history["loss"], label="loss")
	plt.plot(results.history["val_loss"], label="val_loss")
	plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
	plt.xlabel("Epochs")
	plt.ylabel("log_loss")
	plt.legend()
	plt.savefig('../output/showing_learning_curve_512x512_6.png')

	del model
	
	K.clear_session()

def main(model_id):
	print(model_id)
	train(model_id)
	
	if __name__ == '__main__':
		main()
