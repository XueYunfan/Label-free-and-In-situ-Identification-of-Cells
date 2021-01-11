import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
from losses import *
from UNet_model import ResUnet, Unet

def training_vis(hist, epoch, model):
	loss = hist.history['loss']
	mae = hist.history['dice_coef']
	val_loss = hist.history['val_loss']
	val_mae = hist.history['val_dice_coef']
	
	plt.rc('font',family='Arial') 
    # make a figure
	fig = plt.figure(figsize=(8,4))

	plt.plot(loss,label='train_loss')
	plt.plot(val_loss,label='val_loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Loss on Training and Validation Data')
	
	plt.tight_layout()
	plt.savefig('XXX.png'.format(model, epoch),dpi=300)
	
	list1 = np.array([loss,mae,val_loss,val_mae])
	name1 = range(1,epoch+1)
	name2 = ['loss','dice_coef','val_loss','val_dice_coef']
	test = pd.DataFrame(columns=name1,index=name2,data=list1)
	test.to_csv('XXX.csv'.format(model, epoch),encoding='gbk')

def parse_function_train(filename, labelname):
	image_contents = tf.io.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_contents)
	image_converted = tf.cast(image_decoded, tf.float16)
	image_processed = tf.image.random_brightness(image_converted, max_delta=0.05)
	image_processed = tf.image.random_contrast(image_processed, 0.8, 1.2)
	image_scaled = tf.divide(image_converted, 255.0)
	
	label_contents = tf.io.read_file(labelname)
	label_decoded = tf.image.decode_jpeg(label_contents)
	label_converted = tf.cast(label_decoded, tf.bfloat16)
	label_converted = tf.divide(label_converted, 255.0)

	return image_scaled, label_converted

def parse_function_test(filename, labelname):
	image_contents = tf.io.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_contents)
	image_converted = tf.cast(image_decoded, tf.float16)
	image_scaled = tf.divide(image_converted, 255.0)
	
	label_contents = tf.io.read_file(labelname)
	label_decoded = tf.image.decode_jpeg(label_contents)
	label_converted = tf.cast(label_decoded, tf.bfloat16)
	label_converted = tf.divide(label_converted, 255.0)

	return image_scaled, label_converted

def training(epoch,step,modelname):

	IMAGE_PATH_TRAIN = 'XXX/input/train/'
	IMAGE_PATH_VALIDATION = 'XXX/input/validation/'
	LABLE_PATH_TRAIN = 'XXX/label/train/'
	LABLE_PATH_VALIDATION = 'XXX/label/validation/'

	image_names_train = os.listdir(IMAGE_PATH_TRAIN)
	image_names_validation = os.listdir(IMAGE_PATH_VALIDATION)
	label_names_train = os.listdir(LABLE_PATH_TRAIN)
	label_names_validation = os.listdir(LABLE_PATH_VALIDATION)

	train_file = []
	validation_file = []

	train_labels = []
	validation_labels = []

	for name in image_names_train:
		train_file.append(IMAGE_PATH_TRAIN+name)

	for name in image_names_validation:
		validation_file.append(IMAGE_PATH_VALIDATION+name)

	for name in label_names_train:
		train_labels.append(LABLE_PATH_TRAIN+name)

	for name in label_names_validation:
		validation_labels.append(LABLE_PATH_VALIDATION+name)

	train_filenames = tf.constant(train_file)
	train_labels = tf.constant(train_labels)

	validation_filenames = tf.constant(validation_file)
	validation_labels = tf.constant(validation_labels)

	train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
	train_dataset = train_dataset.shuffle(len(train_file))

	validation_dataset = tf.data.Dataset.from_tensor_slices((validation_filenames, validation_labels))

	train_dataset = train_dataset.map(parse_function_train)
	validation_dataset = validation_dataset.map(parse_function_test)

	train_dataset = train_dataset.batch(2).repeat()
	validation_dataset = validation_dataset.batch(2)
	
	model = ResUnet()
	#model.load_weights('XXX.h5')
	
	model.compile(loss=dice_coef_loss, optimizer='RMSprop', metrics=[dice_coef])
	filepath = 'XXX.h5'.format(modelname, epoch)
	checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_dice_coef', 
			verbose=0, save_best_only=True, save_weights_only=True, 
			mode='max', period=1)
	callback = [checkpoint]
	hist = model.fit(train_dataset, validation_data=validation_dataset, 
			epochs=epoch, steps_per_epoch=step, callbacks=callback)
			
	return hist

if __name__ == '__main__':
	
	hist = training(epoch=40, step=500, modelname='ResUnet')
	training_vis(hist, epoch=40, model='ResUnet')
