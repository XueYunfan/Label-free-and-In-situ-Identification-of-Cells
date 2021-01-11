from sklearn.cluster import DBSCAN,KMeans
import cv2
import numpy as np
import os

def pixel_locations(img):
	array = img.shape
	locations = []
	for x in range(array[0]):
		for y in range(array[1]):
			if img[x,y]!=0:
				locations.append([x,y])
	locations = np.array(locations)
	return locations

def K_means(locations, labels, Low_bound=100):
	nucleus_locs = []
	for n in range(max(labels)):
		cluster=[]
		for x,y in zip(locations, labels):
			if y==n:
				cluster.append(x)
		if len(cluster) <= Low_bound:
			continue
		cluster = np.array(cluster)
		cluster = KMeans(n_clusters=1, random_state=0).fit(cluster)
		nucleus_loc = cluster.cluster_centers_
		nucleus_loc = nucleus_loc.tolist()
		nucleus_locs.append(nucleus_loc[0])
	return nucleus_locs

def cell_location(img,eps=1,min_samples=1):
	
	#b,g,r=cv2.split(img)
	#a,b = cv2.threshold(b,50,255,cv2.THRESH_TOZERO)
	#b = cv2.adaptiveThreshold(b,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
	#	cv2.THRESH_BINARY,59,0)
	locations = pixel_locations(img)
	labels = DBSCAN(eps=eps, min_samples=min_samples,n_jobs=-1).fit_predict(locations)
	nucleus_locs = K_means(locations, labels)
	#img = cv2.merge([prediction,prediction,prediction])
	#for location in nucleus_locs:
	#	loc = (int(location[1]),int(location[0]))
	#	cv2.circle(img, loc, 10, (0,0,255), -1)
	#cv2.imwrite('example.jpg',img)
	return nucleus_locs
	
if __name__ == '__main__':
	imgpath='XXX.jpg'
	img = cv2.imread(imgpath)
	cell_location(img)
