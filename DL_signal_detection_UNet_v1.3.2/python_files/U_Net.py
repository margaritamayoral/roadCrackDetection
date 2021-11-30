from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
#from import_data import im_chan, im_width, im_height
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import config

## Set up configuration 
im_chan = config.image_params['channel']
im_width = config.image_params['width']
im_height = config.image_params['height']
padding = config.u_net_params['padding']
activation1 = config.u_net_params['activation1']
activation2 = config.u_net_params['activation2']
kernel_down_size = config.u_net_params['kernel_down_size']
kernel_up_size = config.u_net_params['kernel_up_size']
pooling = config.u_net_params['pooling']
stride = config.u_net_params['stride']

## Build U-Net model

inputs = Input((im_height, im_width, im_chan))
s = Lambda(lambda x: x/255)(inputs)

c1 = Conv2D(8, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(s)  ## two consecutive Convolutional layers
c1 = Conv2D(8, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(c1)  ## c1: output tensor of Convolutional Layers
p1 = MaxPooling2D((pooling, pooling))(c1)  ## p1: output tensor of Max Pooling Layers

c2 = Conv2D(16, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(p1)
c2 = Conv2D(16, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(c2)   ## c2: output tensor of the second Convolutional Layer
p2 = MaxPooling2D((2, 2))(c2)  ## p2: output tensor of Max Pooling layers

c3 = Conv2D(32, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(p2)
c3 = Conv2D(32, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(c3)  ## c3: output tensor of the thirth Convolutional Layer
p3 = MaxPooling2D((2, 2))(c3)  ## p3: output tensor of Max Pooling layers

c4 = Conv2D(64, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(p3)
c4 = Conv2D(64, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(c4)   ## c4: output tensor of the fourth Convolutional Layer
p4 = MaxPooling2D((2,2))(c4)  ## p4: output tensor of Max Pooling Layers

c5 = Conv2D(128, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(p4)
c5 = Conv2D(128, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(c5)  ## c5: output tensor of the fifth Convolutional layer

u6 = Conv2DTranspose(64, (kernel_up_size, kernel_up_size), strides=(stride, stride), padding=padding)(c5)  ## u6: output tensor of up-sampling
u6 = concatenate([u6, c4])      ###  (transposed convolutional) layers
c6 = Conv2D(64, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(u6)
c6 = Conv2D(64, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(c6)  ## output tensor of Convolutional Layers

u7 = Conv2DTranspose(32, (kernel_up_size, kernel_up_size), strides=(stride, stride), padding=padding)(c6)  ### output tensors of up-sampling layers
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(u7)
c7 = Conv2D(32, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(c7)  ## output tensors of Convolutional layeers

u8 = Conv2DTranspose(16, (kernel_up_size, kernel_up_size), strides=(stride, stride), padding=padding)(c7)  ## output tensors of up-sampling layers
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(u8)  ### output tensors of Convolutional layers
c8 = Conv2D(16, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(c8)

u9 = Conv2DTranspose(8, (kernel_up_size, kernel_up_size), strides=(stride, stride), padding=padding)(c8)  ## output tensors of up-sampling layers
u9 = concatenate([u9, c1])
c9 = Conv2D(8, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(u9)
c9 = Conv2D(8, (kernel_down_size, kernel_down_size), activation=activation1, padding=padding)(c9) ## output tensor of the last convolutional layers

outputs = Conv2D(1, (1, 1), activation=activation2)(c9)
