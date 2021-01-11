import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.models import *
from sklearn.metrics import confusion_matrix

def parse_function_test(filename, label):
	image_contents = tf.io.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_contents)
	image_converted = tf.cast(image_decoded, tf.float32)
	image_scaled = tf.divide(image_converted, 255.0)
	return image_scaled, label

def CNN(input_shape=(224,224,3)):
	
	base_model = tf.keras.applications.ResNet50V2(
		weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
	output1 = base_model.output
	predictions = tf.keras.layers.Dense(2, activation='softmax')(output1)
	model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
	
	return model

TEST_PATH = 'XXX/test/'

test_names = os.listdir(TEST_PATH)
test_file = []
test_labels = []

for name in test_names:
test_file.append(TEST_PATH+name)
if name[0]=='R':
	test_labels.append(0)
else:
	test_labels.append(1)

test_filenames = tf.constant(test_file)
test_labels_onehot = tf.keras.utils.to_categorical(test_labels, 2, dtype='int32')
test_labels_constant = tf.constant(test_labels_onehot)
test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels_constant))
test_dataset = test_dataset.map(parse_function_test)
test_dataset = test_dataset.batch(8)
		
model = CNN()
model.load_weights('XXX.h5')
model.compile(optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy'])
model.evaluate(test_dataset)
predictions = model.predict(test_dataset)
predictions = np.argmax(predictions, axis=1)

for x,y,z in zip(predictions,test_labels,test_file):#find wrongly classified images
	if x!=y:
		print(x,y,z)

cm = confusion_matrix(test_labels, predictions)
print(cm)
