from DBSCAN import cell_location
import numpy as np
import pandas as pd
import cv2
import os

def def_cell_type(block, location):
	
	b,g,r = cv2.split(block[97:127,97:127,:])
	if np.sum(r)>np.sum(g):
		return 'R'
	else:
		return 'G'

def cell_num_annotation(PATH,img_name,DAPI_PATH,dapi_name):
	
	ec_num=0
	smc_num=0
	img = cv2.imread(PATH+img_name)
	dapi = cv2.imread(DAPI_PATH+dapi_name,0)
	size = dapi.shape

	nucleus_locs = cell_location(dapi)
	for loc in nucleus_locs:
		if (int(loc[0])<112) or (int(loc[1])<112) or (size[0]-int(loc[0])<112) or (size[1]-int(loc[1])<112):
			continue
		block_f = img[int(loc[0]-112):int(loc[0]+112),int(loc[1]-112):int(loc[1]+112),:]
		label = def_cell_type(block_f,loc)
		circle_loc = (int(loc[1]),int(loc[0]))
		if label =='R':
			cv2.circle(img, circle_loc, 10, (0,255,0), -1)
			ec_num+=1
		else:
			cv2.circle(img, circle_loc, 10, (0,0,255), -1)
			smc_num+=1
		cv2.imwrite('XXX/annotation/{}'.format(img_name),img)#fluorescent images with red and green tags on cells
		
	return ec_num,smc_num,ec_num+smc_num

if __name__ == '__main__':
	PATH = 'XXX/label/'#fluorescent images
	DAPI_PATH = 'XXX/DAPI-label/' #using DAPI images binarized by UNet
	image_name = os.listdir(PATH)
	image_name.sort(key=lambda x:int(x[:-4]))
	dapi_name = os.listdir(DAPI_PATH)
	dapi_name.sort(key=lambda x:int(x[:-4]))
	
	data=[]
	for name,dapi_name in zip(image_name,dapi_name):
		a,b,c=cell_num_annotation(PATH,name,DAPI_PATH,dapi_name)
		data.append([name,a,b,c])
		print(name)

	data = pd.DataFrame(data,columns=['Image Name','EC','SMC','Total'])
	data.to_excel('E:/deep learning files/ECSMC Coculture/integrated/TCPS/Cell Number Annotations.xlsx')
