import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import *
from UNet_model import ResUnet
from DBSCAN import cell_location
import cv2
import os

def CNN(input_shape=(224,224,3)):
	
	base_model = tf.keras.applications.ResNet50V2(
		weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
	output1 = base_model.output
	predictions = tf.keras.layers.Dense(2, activation='softmax')(output1)
	model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
	
	return model

def def_cell_type(block):
	
	cell_type = model2.predict(block)
	if cell_type[0,0] > cell_type[0,1]:
		return 'EC'
	else:
		return 'SMC'

def cell_identification(img_dapi,inputs):
	
	cells = []
	nucleus_locs = cell_location(img_dapi)
	size = img_dapi.shape
	m,n=1,1
	for loc in nucleus_locs:
		if (int(loc[0])<112) or (int(loc[1])<112) or (size[0]-int(loc[0])<112) or (size[1]-int(loc[1])<112):
			continue
		block = inputs[:,int(loc[0]-112):int(loc[0]+112),int(loc[1]-112):int(loc[1]+112),:]
		loc.append(def_cell_type(block))
		cells.append(loc)
		
	return cells

def cell_counting(img):
	
	image_contents = tf.io.read_file(img)
	image_decoded = tf.image.decode_jpeg(image_contents)
	image_converted = tf.cast(image_decoded, tf.bfloat16)
	image_scaled = tf.divide(image_converted, 255.0)
	image_scaled = tf.reshape(image_scaled,(1,1536,1536,3))

	prediction = model1.predict(image_scaled)

	prediction = np.reshape(prediction,(1536,1536))
	a, prediction = cv2.threshold(prediction,0.7,255,cv2.THRESH_BINARY)
	
	cells = cell_identification(prediction,image_scaled)
	
	return cells,prediction

def pred_to_img(INPUT_PATH,img_name,LABEL_PATH,label_name):
	
	img = cv2.imread(INPUT_PATH+img_name)
	label = cv2.imread(LABEL_PATH+label_name)
	cells,prediction = cell_counting(INPUT_PATH+img_name)
	dapi_24_bit = cv2.merge([prediction,prediction,prediction])
	EC_num=0
	SMC_num=0
	for value in cells:
		loc = (int(value[1]),int(value[0]))
		cell = value[2]
		cv2.circle(dapi_24_bit, loc, 8, (0,0,255), -1)
		if cell =='EC':
			cv2.circle(label, loc, 10, (0,255,0), -1)
			cv2.circle(img, loc, 10, (0,255,0), -1)
			EC_num+=1
		else:
			cv2.circle(label, loc, 10, (0,0,255), -1)
			cv2.circle(img, loc, 10, (0,0,255), -1)
			SMC_num+=1
	cell_num = [img_name,EC_num,SMC_num,EC_num+SMC_num]
	
	cv2.imwrite('XXX/pred/UNet-{}'.format(img_name),prediction)#predicted DAPI imags by UNet
	cv2.imwrite('XXX/pred/DBSCAN-{}'.format(img_name),dapi_24_bit)#predicted locations of cell nuclei by clustering
	cv2.imwrite('XXX/pred/Label-{}'.format(label_name),label)#fluorescent images with red and green tags on cells
	cv2.imwrite('XXX/pred/Brightfield-{}'.format(img_name),img)#brightfield images with red and green tags on cells
	
	return cell_num

if __name__ == '__main__':
	
	gpus=tf.config.experimental.list_physical_devices(device_type='GPU')
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'
	tf.config.experimental.set_memory_growth(gpus[0],True)
	tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)

	model1 = ResUnet()
	model1.load_weights('XXX.h5')
		
	model2 = CNN()
	model2.load_weights('XXX.h5')
	
	INPUT_PATH = 'XXX/input/'
	LABEL_PATH = 'XXX/label/'

	img_names = os.listdir(INPUT_PATH)
	label_names = os.listdir(LABEL_PATH)
	img_names.sort(key=lambda x:int(x[:-4]))
	label_names.sort(key=lambda x:int(x[:-4]))
	
	cell_num=[]
	for img_name,label_name in zip(img_names,label_names):
		print(img_name)
		cell_num.append(pred_to_img(INPUT_PATH,img_name,LABEL_PATH,label_name))
		n+=1
	
	data = pd.DataFrame(cell_num,columns=['Image Name','EC','SMC','Total'])
	data.to_excel('XXX/Cell Number Predictions.xlsx')
