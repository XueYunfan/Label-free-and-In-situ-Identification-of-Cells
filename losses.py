import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import *
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import utils
from tensorflow.keras.constraints import min_max_norm
import tensorflow as tf

pre_model = load_model('E:/Python files/focus/Three Cells/no validation/best model fold-1 134 no-validation.hdf5')
inputs = Input((388,388,3))
x = inputs
for layer in pre_model.layers[1:5]:
	x = layer(x)
loss_model = Model(inputs=inputs, outputs=x)
loss_model.trainable = False

def SSIM_loss(y_true, y_pred):
	loss = 1-tf.image.ssim(y_true,y_pred,max_val=1)
	return loss
	
def SSIM(y_true, y_pred):
	loss = tf.image.ssim(y_true,y_pred,max_val=1)
	return loss

def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def l2_loss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))
    
def UNet_loss(y_true, y_pred):
	#PL = perceptual_loss(y_true, y_pred)
	SSIM = SSIM_loss(y_true, y_pred)
	L1 = l1_loss(y_true, y_pred)
	loss = L1+SSIM*100
	return loss

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    smooth = 1
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)
