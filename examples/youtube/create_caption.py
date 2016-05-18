from collections import OrderedDict
import argparse
import cPickle as pickle
import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys

sys.path.append('../../python/')
import caffe

#from captioner import Captioner


def predict_single_word(net,descriptor, previous_word, output='probs'):
    	cont = 0 if previous_word == 0 else 1
    	cont_input= np.array([cont])
    	word_input= np.array([previous_word])
	mean_fc7 = np.zeros_like(net.blobs['mean_fc7'].data)
    	mean_fc7[:] = descriptor
    	net.forward(mean_fc7=mean_fc7,cont_sentence=cont_input, input_sentence=word_input)
    	output_preds = net.blobs[output].data[0, 0, :]
    	return output_preds

def softmax(softmax_inputs, temp):
	shifted_inputs = softmax_inputs - softmax_inputs.max()
	exp_outputs = np.exp(temp * shifted_inputs)
  	exp_outputs_sum = exp_outputs.sum()
  	if math.isnan(exp_outputs_sum):
    		return exp_outputs * float('nan')
  	assert exp_outputs_sum > 0
  	if math.isinf(exp_outputs_sum):
    		return np.zeros_like(exp_outputs)
  	eps_sum = 1e-20
  	return exp_outputs / max(exp_outputs_sum, eps_sum)

def generate_sentence(net,descriptor, temp=float('inf'), output='predict', max_words=50):
	cont_input = np.array([0])
    	word_input = np.array([0])
	mean_fc7 = np.zeros_like(net.blobs['mean_fc7'].data)
        mean_fc7[:] = descriptor
    	sentence = []
    	while len(sentence) < max_words and (not sentence or sentence[-1] != 0):
        	net.forward(mean_fc7=mean_fc7,cont_sentence=cont_input, input_sentence=word_input)
        	output_preds = net.blobs[output].data[0, 0, :]
        	sentence.append(random_choice_from_probs(output_preds, temp=temp))
        	cont_input[0] = 1
        	word_input[0] = sentence[-1]
    	return sentence

def sentence(vocab_indices,vocab):
    	sentence = ' '.join([vocab[i] for i in vocab_indices])
    	if not sentence: return sentence
    	sentence = sentence[0].upper() + sentence[1:]
  	# If sentence ends with ' <EOS>', remove and replace with '.'
    	# Otherwise (doesn't end with '<EOS>' -- maybe was the max length?):
    	# append '...'
    	suffix = ' ' + vocab[0]
    	if sentence.endswith(suffix):
      		sentence = sentence[:-len(suffix)] + '.'
    	else:
      		sentence += '...'
    	return sentence

def random_choice_from_probs(softmax_inputs, temp=1, already_softmaxed=False):
	# temperature of infinity == take the max
	if temp == float('inf'):
		return np.argmax(softmax_inputs)

	if already_softmaxed:
		probs = softmax_inputs
		assert temp == 1
	else:
    		probs = softmax(softmax_inputs, temp)

  	r = random.random()
  	cum_sum = 0.

  	for i, p in enumerate(probs):
    		cum_sum += p
    		if cum_sum >= r: return i

  	return 1  # return UNK?

def predict_single_word_from_all_previous(net,descriptor, previous_words):
	for word in [0] + previous_words:
		probs = predict_single_word(net,descriptor, word)
	return probs


VOCAB_FILE = 'vocabulary_buff100.txt'  #_buff100.txt'
FRAMEFEAT_FILE_PATTERN = 'vid1180_fc7_pool.txt'
LSTM_NET_FILE = 'poolmean_deploy.prototxt'
#RESULTS_DIR = './results'
#MODEL_FILE = 'Models/naacl15_pool_vgg_fc7_mean_fac2.caffemodel'
#MODEL_FILE = 'Models/pool_fc7_mean_fac_2layer_iter_110000.caffemodel'
#MODEL_FILE = 'Models/pool_fc7_mean_fac_adam_2layer_iter_85000.caffemodel'
MODEL_FILE = 'Models/pool_fc7_mean_fac_nesterov_2layer_iter_70000.caffemodel'

#IMAGENET_DEPLOY=
#IMAGENET_W=
strategy={}
strategy['beam'] = 1
strategy['beam_size'] = 1
#strategy['temp']=1
vocab = ['<EOS>']
with open(VOCAB_FILE, 'r') as vocab_file:
    vocab += [line.strip() for line in vocab_file.readlines()]
frames_vids=open(FRAMEFEAT_FILE_PATTERN, 'r')
frames =  frames_vids.readlines()


lstm_net=caffe.Net(LSTM_NET_FILE,MODEL_FILE,caffe.TEST)
[(k, v.data.shape) for k, v in lstm_net.blobs.items()]
batch_size=1
lstm_net.blobs['cont_sentence'].reshape(1, batch_size)
lstm_net.blobs['input_sentence'].reshape(1, batch_size)
lstm_net.blobs['mean_fc7'].reshape(batch_size,*lstm_net.blobs['mean_fc7'].data.shape[1:])

#input to LSTM
descriptor=np.array(map(float, frames[0].strip().split(',')),dtype=np.float)
vals=predict_single_word(lstm_net,descriptor, 0, output='probs')

out_sentence=generate_sentence(lstm_net,descriptor)
print 'predicted sentence with beam width of 1'
print out_sentence
print sentence(out_sentence,vocab)

out_sentence=generate_sentence(lstm_net,descriptor,temp=1)
print 'predicted sentence with temp=1'
print sentence(out_sentence,vocab)
out_sentence=generate_sentence(lstm_net,descriptor,temp=2)
print 'predicted sentence with temp=2'
print sentence(out_sentence,vocab)
