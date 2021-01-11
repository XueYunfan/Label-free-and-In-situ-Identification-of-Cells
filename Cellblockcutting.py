from DBSCAN import cell_location
import numpy as np
import cv2
import os

def def_cell_type(block, location):
	
	b,g,r = cv2.split(block[87:137,87:137,:])
	if np.sum(r)>np.sum(g):
		return 'R'
	else:
		return 'G'

def cut_cell_block(img_f,img_b,num):
	
	#b, g ,r = cv2.split(img_f)
	#a, b = cv2.threshold(b,80,255,cv2.THRESH_BINARY)
	nucleus_locs = cell_location(img_f)
	size = img_f.shape
	m,n=1,1
	for loc in nucleus_locs:
		if (int(loc[0])<112) or (int(loc[1])<112) or (size[0]-int(loc[0])<112) or (size[1]-int(loc[1])<112):
			continue
		block_b = img_b[int(loc[0]-112):int(loc[0]+112),int(loc[1]-112):int(loc[1]+112),:]
		block_f = img_f[int(loc[0]-112):int(loc[0]+112),int(loc[1]-112):int(loc[1]+112),:]
		label = def_cell_type(block_f,loc)
		if label=='R':
			cv2.imwrite('XXX/{}-{}-{}.jpg'.format(label,num,m),block_b,[int(cv2.IMWRITE_JPEG_QUALITY),100])
			m+=1
		else:
			cv2.imwrite('XXX/{}-{}-{}.jpg'.format(label,num,n),block_b,[int(cv2.IMWRITE_JPEG_QUALITY),100])
			n+=1

if __name__ == '__main__':
	FILE_PATH1 = 'XXX/brightfield/'
	FILE_PATH2 = 'XXX/fluorescent/'
	image_name = os.listdir(FILE_PATH1)
	#image_name.remove('1-1 B_RGB.tif')
	label_name = os.listdir(FILE_PATH2)
	#label_name.remove('1-1 F_RGB.tif')

	num = 1
	for name_b,name_f in zip(image_name,label_name):
		img_b = cv2.imread(FILE_PATH1+name_b)
		img_f = cv2.imread(FILE_PATH2+name_f)
		for x in range(0,15360,2560):
			for y in range(0,15360,2560):
				print(name_b,num,x,y)
				cut_cell_block(img_f[x:x+2560,y:y+2560,:],img_b[x:x+2560,y:y+2560,:],num)
				num+=1
	
	#img_b = cv2.imread('E:/microscope files/machine learning/SMC EC co-culture/BF/1-1 B_RGB.tif')
	#img_f = cv2.imread('E:/microscope files/machine learning/SMC EC co-culture/F/1-1 F_RGB.tif')
	#num = 1
	#for x in range(0,15360,3840):
	#	for y in range(0,15360,3840):
	#		print(x,y)
	#		cut_cell_block(img_f[x:x+3840,y:y+3840,:],img_b[x:x+3840,y:y+3840,:],num)
	#		num+=1
