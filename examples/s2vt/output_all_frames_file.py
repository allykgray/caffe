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

MODEL_FILE = '../youtube/VGG_ILSVRC_16_layers.caffemodel'

DEPLOY_FILE = '../youtube/VGG_ILSVRC_16_layers_deploy.prototxt'

MOVIEPATH='/media/agray/2e244478-fdaa-4793-a88b-179fb56d27d6/agray/Documents/data/youtube/frames/'
#MOVIES= os.listdir(MOVIEPATH)

MOVIE_KEY_FILE='../youtube/youtube_video_to_id_mapping.txt'
NAME_KEY=open(MOVIE_KEY_FILE).readlines()
KEY=[]
for i in range(len(NAME_KEY)):
	KEY.append(NAME_KEY[i].split(' ',2))

NAME_KEY.close()

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

#fc6_file = open("fc6_allframes.txt","w") 
fc7_file = open("vid_1180_fc7_all_frames.txt","w")
#fc8_file = open("fc8_allframes.txt","w")
for j in range(1):#len(KEY)):#MOVIES):
	MOVIES=KEY[j+1179][0]
        MovieFiles= glob.glob(MOVIEPATH+MOVIES+'/*.jpg')
	#print len(MovieFiles)
	# add loop for multiple movies 
	for i in range(len(MovieFiles)):
		cnn_image = caffe.io.load_image(MovieFiles[i])
		transformed_image = transformer.preprocess('data', cnn_image)
		cnn_net.blobs['data'].data[...] = transformed_image
		output = cnn_net.forward()	
	#fc6_file.write((MOVIES[j]+','+str(fc6_parameters[0][0].tolist())+'\n'))
		fc7_file.writelines((KEY[j][1][:-1]+'_frame_'+str(i+1)+','+str(cnn_net.blobs['fc7'].data.tolist()[0])[1:-1]+'\n'))
	#fc8_file.write((MOVIES[j]+','+str(fc8_parameters[0][0].tolist())+'\n'))
        

#fc6_file.close()
fc7_file.close()
#fc8_file.close()
