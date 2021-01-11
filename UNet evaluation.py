import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from PIL import Image
from losses import *
from UNet_model import ResUnet

IMAGE_PATH_TEST = 'XXX/input/'
LABLE_PATH_TEST = 'XXX/label/'

image_names_test = os.listdir(IMAGE_PATH_TEST)
label_names_test = os.listdir(LABLE_PATH_TEST)

test_file = []
test_label = []

for name in image_names_test:
	test_file.append(IMAGE_PATH_TEST+name)

for name in label_names_test:
	test_label.append(LABLE_PATH_TEST+name)
		
def parse_function(filename, labelname):
	image_contents = tf.io.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_contents)
	image_converted = tf.cast(image_decoded, tf.bfloat16)
	image_scaled = tf.divide(image_converted, 255.0)
	
	label_contents = tf.io.read_file(labelname)
	label_decoded = tf.image.decode_jpeg(label_contents)
	label_converted = tf.cast(label_decoded, tf.bfloat16)
	label_converted = tf.divide(label_converted, 255.0)
	
	return image_scaled, label_converted

test_filenames = tf.constant(test_file)
test_labels = tf.constant(test_label)
test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
test_dataset = test_dataset.map(parse_function)
test_dataset = test_dataset.batch(2)

model = ResUnet()
model.compile(loss=dice_coef_loss, optimizer='RMSprop', metrics=[dice_coef])
model.load_weights(
	'XXX.h5')
model.evaluate(test_dataset)
