

import numpy as np
import tensorflow as tf
import sys
import time 
import glove as glove 


reload(sys)
sys.setdefaultencoding('utf8')
vector = [] 

def word_to_glove(sentence):

  # sentences which have form of [ [..] , [..], [..] , ... ] 
  # SHOULD BE FLATTENED 
  sentence = [item for sublist in sentence for item in sublist]
  
  words = set(sentence)
  voca_size = len(words)
  #print words
  #print ("Size of Voca : " , voca_size)

  dictionary_s = dict()
  for word in words:
  	dictionary_s[word] = len(dictionary_s)


  wordVectors = glove.loadWordVectors(dictionary_s)

  return wordVectors,dictionary_s
  

def word_to_idx(sentences,dictionary):


	word = []  			# word vectors are stored in here.
	temp = [] 

	# GIVEN SENTENCES 
	for sen in sentences:
		for voca in sen:
			idx = dictionary[voca]
			temp.append(idx) 
		
		word.append(temp)
		temp = [] 


	return word


		
 	


