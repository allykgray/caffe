#!/usr/bin/env python
import glob
from collections import OrderedDict
import os 
import re
import json
import numpy as np
import pprint
import cPickle as pickle
import string
import sys
import matplotlib.pyplot as plt
sys.path.insert(0,'/home/agray/Documents/dltools/caffe-s2vt/python')

import caffe

MODEL_FILE = 'VGG_ILSVRC_16_layers.caffemodel'

DEPLOY_FILE = 'VGG_ILSVRC_16_layers_deploy.prototxt'

MOVIEPATH='/media/agray/2e244478-fdaa-4793-a88b-179fb56d27d6/agray/Documents/data/youtube/frames/'
#MOVIES= os.listdir(MOVIEPATH)

MOVIE_KEY_FILE='youtube_video_to_id_mapping.txt'
NAME_KEY=open(MOVIE_KEY_FILE).readlines()
KEY=[]
for i in range(len(NAME_KEY)):
	KEY.append(NAME_KEY[i].split(' ',2))

#for i in range(len(MOVIES)):
#	MovieFiles= glob.glob(MOVIEPATH+MOVIES+'/*.jpg')
#DEVICE_ID = 0
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('/home/agray/Documents/dltools/caffe-s2vt/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
#Load CNN to evaluate features
cnn_net = caffe.Net(DEPLOY_FILE,      # defines the structure of the model
                MODEL_FILE,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout

transformer = caffe.io.Transformer({'data': cnn_net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
cnn_net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 227x227

############### LSTM Info ####################

fc6_out=[]
fc7_out=[]
fc8_out=[]

#fc6_file = open("fc6_pool.txt","w") 
fc7_file = open("fc7_pool.txt","w")
#fc8_file = open("fc8_pool.txt","w")
for j in range(MOVIES):
	MOVIES=KEY[j][0]
        MovieFiles= glob.glob(MOVIEPATH+MOVIES+'/*.jpg')
	#print len(MovieFiles)
	frame_array=np.zeros((len(MovieFiles),4096))
	# add loop for multiple movies 
	for i in range(len(MovieFiles)):
		cnn_image = caffe.io.load_image(MovieFiles[i])
		transformed_image = transformer.preprocess('data', cnn_image)
		cnn_net.blobs['data'].data[...] = transformed_image
		output = cnn_net.forward()	
		frame_array[i,:]=cnn_net.blobs['fc7'].data[0]
	#fc6_parameters=[[np.mean(evaluate_fc6,axis=0)],[np.median(evaluate_fc6,axis=0)],[np.std(evaluate_fc6,axis=0)],[np.std(evaluate_fc6,axis=0)]]
	fc7_parameters=[[np.mean(frame_array,axis=0)],[np.median(frame_array,axis=0)],[np.std(frame_array,axis=0)],[np.std(frame_array,axis=0)]]
#evaluate_pool=np.array(pool_out)
	#fc8_parameters=[[np.mean(evaluate_fc8,axis=0)],[np.median(evaluate_fc8,axis=0)],[np.std(evaluate_fc8,axis=0)],[np.std(evaluate_fc8,axis=0)]]
	#fc6_file.write((str(fc6_parameters[0][0].tolist())[1:-1]+'\n'))
        fc7_file.write((str(fc7_parameters[0][0].tolist())[1:-1]+'\n'))
        #fc8_file.write((str(fc8_parameters[0][0].tolist())[1:-1]+'\n'))

#fc6_file.close()
fc7_file.close()
#fc8_file.close()
