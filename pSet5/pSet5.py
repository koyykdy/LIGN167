# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 17:55:12 2018

@author: koyyk_000
"""

import torch
import torch.nn as nn
import torch.optim as optim
#import spacy
import spacy; nlp = spacy.load('en')

###### DATA PROCESSING STARTS #########


def load_corpus():
	#this loads the data from sample_corpus.txt
	with open('sample_corpus.txt','r',encoding='utf-8') as f:
		corpus = f.read().replace('\n','')
	return corpus

def remove_infrequent_words(sents):
	word_counts = {}
	for s in sents:
		for w in s:
			if w in word_counts:
				word_counts[w] += 1
			else:
				word_counts[w] = 1

	threshold = 2
	filtered_sents = []
	for s in sents:
		new_s = []
		for w in s:
			if word_counts[w] < threshold:
				new_s.append('<UNKNOWN>')
			else:
				new_s.append(w)
		filtered_sents.append(new_s)
	return filtered_sents

def segment_and_tokenize(corpus):
	#make sure to run: python -m spacy download en 
	#in the command line before using this!

	#corpus is assumed to be a string, containing the entire corpus
	nlp = spacy.load('en')
	tokens = nlp(corpus)
	sents = [[t.text for t in s] for s in tokens.sents if len([t.text for t in s])>1]
	sents = remove_infrequent_words(sents)
	sents = [['<START>']+s+['<END>'] for s in sents]
	return sents

def make_word_to_ix(sents):
	word_to_ix = {}
	num_unique_words = 0
	for sent in sents:
		for word in sent:
			if word not in word_to_ix:
				word_to_ix[word] = num_unique_words
				num_unique_words += 1


	return word_to_ix

def sent_to_onehot_vecs(sent,word_to_ix):
	#note: this is not how you would do this in practice! 

	vecs = []
	for i in range(len(sent)):
		word = sent[i]
		word_index = word_to_ix[word]

		vec = torch.zeros(len(word_to_ix), dtype=torch.float32,requires_grad=False)
		vec[word_index] = 1
		vecs.append(vec)

	return vecs

def vectorize_sents(sents,word_to_ix):
	one_hot_vecs = []
	for s in sents:
		one_hot_vecs.append(sent_to_onehot_vecs(s,word_to_ix))
	return one_hot_vecs

def get_data():
	corpus = load_corpus()
	sents = segment_and_tokenize(corpus)
	word_to_ix = make_word_to_ix(sents)

	vectorized_sents = vectorize_sents(sents,word_to_ix)

	vocab_size = len(word_to_ix)

	return vectorized_sents, vocab_size




###### DATA PROCESSING ENDS #########




###### RNN DEFINITION STARTS #########

def elman_unit(word_embedding,h_previous,W_x,W_h,b):
	return torch.sigmoid(torch.matmul(W_x,word_embedding)+torch.matmul(W_h,h_previous)+b)

def embed_word(word,W_e):
	#word is a one-hot vector
	return torch.matmul(W_e,word)



def single_layer_perceptron(h,W_p):
	s = torch.matmul(W_p,h)
	softmax = nn.Softmax(dim=0)
	return softmax(s)


def network_forward(sent,param_dict):
	W_e = param_dict['W_e']
	W_x = param_dict['W_x'] 
	W_h = param_dict['W_h']
	W_p = param_dict['W_p']
	b = param_dict['b']


	h_previous = initialize_hidden_state(W_h.size()[1])

	predictions = []
	for i in range(len(sent)-1):
		current_word = sent[i]

		current_word_embedding = embed_word(current_word,W_e)

		h_current = elman_unit(current_word_embedding,h_previous, W_x,W_h,b)

		prediction = single_layer_perceptron(h_current,W_p)
		predictions.append(prediction)

		h_previous = h_current

	return predictions



###### RNN DEFINITION ENDS #########


#### LOSS FUNCTION BEGINS #######

def word_loss(word_probs, word):
	#outcome is a one-hot vector
	prob_of_word = torch.dot(word_probs,word)
	return -1*torch.log(prob_of_word)

def sent_loss(predictions, sent):
	L = torch.tensor(0,dtype=torch.float32)

	for i in range(len(predictions)):
		word_probs = predictions[i]
		observed_word = sent[i+1]
		L+=word_loss(word_probs,observed_word)

	return L


##### LOSS FUNCTION ENDS #######



#####WEIGHT INITIALIZATION STARTS #######

def initialize_weight_matrix(shape):
	return torch.rand(shape,dtype=torch.float32,requires_grad=True)

def initialize_hidden_state(shape):
	return torch.zeros(shape,dtype=torch.float32,requires_grad=False)


def initialize_parameters(embedding_dim,vocab_size,hidden_state_dim):
	W_e, W_x, W_h,W_p, b = (initialize_weight_matrix([embedding_dim,vocab_size]), 
		initialize_weight_matrix([hidden_state_dim,embedding_dim]), 
		initialize_weight_matrix([hidden_state_dim,hidden_state_dim]),
		initialize_weight_matrix([vocab_size,hidden_state_dim]),
		initialize_weight_matrix(hidden_state_dim))


	param_dict = {}
	param_dict['W_e'] = W_e
	param_dict['W_x'] = W_x
	param_dict['W_h'] = W_h
	param_dict['W_p'] = W_p
	param_dict['b'] = b

	return param_dict


#####WEIGHT INITIALIZATION ENDS #######




def train():
	
	vectorized_sents, vocab_size = get_data()
	

	num_epochs = 100

	hidden_state_dim = 10
	embedding_dim = 10
	learning_rate = 0.00001

	


	param_dict = initialize_parameters(embedding_dim,vocab_size,hidden_state_dim)

	optimizer = optim.SGD(list(param_dict.values()), lr=learning_rate)

	for i in range(num_epochs):

		total_loss = 0
		for s in vectorized_sents:
			optimizer.zero_grad()
			predictions = network_forward(s,param_dict)
			loss = sent_loss(predictions,s)
			total_loss += loss

			loss.backward() #this is the crucial step! this is where PyTorch automatically calculates gradients for all of the model parameters
			optimizer.step() #this is where gradient descent occurs: for each weight vector w, w = w-lr*w.grad
		print(total_loss)


if __name__=='__main__':
	train()

