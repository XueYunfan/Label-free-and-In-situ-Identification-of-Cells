import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import pandas as pd

def training_vis(hist, epoch, modelname):
	loss = hist.history['loss']
	val_loss = hist.history['val_loss']
	acc = hist.history['accuracy']
	val_acc = hist.history['val_accuracy']
	
	plt.rc('font',family='Arial') 
    # make a figure
	fig = plt.figure(figsize=(8,4))
    # subplot loss
	ax1 = fig.add_subplot(121)
	ax1.plot(loss,label='train_loss')
	ax1.plot(val_loss,label='val_loss')
	ax1.set_xlabel('Epochs')
	ax1.set_ylabel('Loss')
	ax1.set_title('Loss on Training and Validation Data')
	ax1.legend()
    # subplot acc
	ax2 = fig.add_subplot(122)
	ax2.plot(acc,label='train_accuracy')
	ax2.plot(val_acc,label='val_accuracy')
	ax2.set_xlabel('Epochs')
	ax2.set_ylabel('Accuracy')
	ax2.set_title('Accuracy on Training and Validation Data')
	ax2.legend()
	plt.tight_layout()
	plt.savefig('XXX.png'.format(modelname,epoch), dpi=300)
	
	list1 = np.array([loss,acc,val_loss,val_acc])
	name1 = range(1,epoch+1)
	name2 = ['loss','accuracy','val_loss','val_accuracy']
	test = pd.DataFrame(columns=name1,index=name2,data=list1)
	test.to_csv('XXX.csv'.format(modelname,epoch),encoding='gbk')


def parse_function_train(filename, label):
	image_contents = tf.io.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_contents)
	image_processed = tf.image.random_flip_left_right(image_decoded)
	image_processed = tf.image.random_flip_up_down(image_processed)
	image_processed = tf.image.random_brightness(image_processed, max_delta=0.1)
	image_processed = tf.image.random_contrast(image_processed, 0.8, 1.2)
	image_converted = tf.cast(image_processed, tf.float32)
	image_scaled = tf.divide(image_converted, 255.0)
	return image_scaled, label

def parse_function_test(filename, label):
	image_contents = tf.io.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_contents)
	image_converted = tf.cast(image_decoded, tf.float32)
	image_scaled = tf.divide(image_converted, 255.0)
	return image_scaled, label

def step_decay(epoch):
    init_lrate = 0.0001
    drop = 0.5
    lrate = init_lrate * pow(drop, (epoch//10))
    print('lrate = '+str(lrate))
    return lrate

def CNN(input_shape=(224,224,3)):
	
	base_model = tf.keras.applications.ResNet50V2(
		weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
	output1 = base_model.output
	dropout1 = Dropout(0.5)(output1)
	predictions = tf.keras.layers.Dense(2, activation='softmax')(dropout1)
	model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
	
	return model

def training(epoch,step,modelname,TRAIN_PATH,VALIDATION_PATH):

	train_names = os.listdir(TRAIN_PATH)
	validation_names = os.listdir(VALIDATION_PATH)
	train_file = []
	validation_file = []
	train_labels = []
	validation_labels = []
	
	for name in train_names:
		train_file.append(TRAIN_PATH+name)
		if name[0]=='R':
			train_labels.append(0)
		else:
			train_labels.append(1)
		
	for name in validation_names:
		validation_file.append(VALIDATION_PATH+name)
		if name[0]=='R':
			validation_labels.append(0)
		else:
			validation_labels.append(1)

	train_filenames = tf.constant(train_file)
	train_labels = tf.keras.utils.to_categorical(train_labels, 2, dtype='int32')
	train_labels = tf.constant(train_labels)

	validation_filenames = tf.constant(validation_file)
	validation_labels = tf.keras.utils.to_categorical(validation_labels, 2, dtype='int32')
	validation_labels = tf.constant(validation_labels)

	train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
	train_dataset = train_dataset.shuffle(len(train_file))

	validation_dataset = tf.data.Dataset.from_tensor_slices((validation_filenames, validation_labels))

	train_dataset = train_dataset.map(parse_function_train)
	validation_dataset = validation_dataset.map(parse_function_test)

	train_dataset = train_dataset.batch(8).repeat()
	validation_dataset = validation_dataset.batch(8)
	
	model = CNN()
	
	for layer in model.layers:
		layer.trainable = True
		
	sgd = tf.keras.optimizers.SGD(lr=0.0, momentum=0.9, decay=0, nesterov=False)
	#model.load_weights('XXX.h5')
	model.compile(optimizer=sgd,
			loss='categorical_crossentropy',
			metrics=['accuracy'])
	model.summary()
	#======================================
	lrate = LearningRateScheduler(step_decay)
	filepath = 'XXX.h5'.format(modelname)
	checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', 
			verbose=0, save_best_only=True, save_weights_only=True, 
			mode='max', period=1)
	callback = [checkpoint,lrate]
	hist = model.fit(train_dataset, validation_data=validation_dataset, 
		epochs=epoch, steps_per_epoch=step, callbacks=callback)
			
	return hist

if __name__ == '__main__':
	TRAIN_PATH = 'XXX/train/'
	VALIDATION_PATH = 'XXX/validation/'
	hist = training(epoch=100, step=1000, modelname='ResNet',TRAIN_PATH,VALIDATION_PATH)
	training_vis(hist, epoch=100, modelname='ResNet')
