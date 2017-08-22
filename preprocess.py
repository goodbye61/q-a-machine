import numpy as np 
import time 
import re 
from collections import OrderedDict 


def pre_process():

	#train_file = open("./dataset/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt","r").read().lower()
	#train_file = open("./sample.txt","r").read()
	train_file = open("./dataset/own_dat.txt","r").read().lower()
	train_file = train_file.split('\n')

	del(train_file[-1])

	inputs = []
	sentences = [] 	
	questions = [] 
	labels = [] 
	sequence = 0 
	max_length = 7
	unk = "<zero>" 


	for sen in train_file:
	
 		inputs = re.split('(\W+)',sen) 		#remove the space 
		del(inputs[0])				#
		del(inputs[-1])				#remove the number
	
		if sequence % 3 == 2:
			temp = ''.join(str(e) for e in inputs)
			temp = temp.split('\t')
			del(temp[-1])
			labels.append([temp[-1]])
			del(temp[-1])
			
			temp = re.split('(\W+)',temp[0])
			
			temp = list(OrderedDict.fromkeys(temp))
			temp[-1] = "?"
			del(temp[0])
			del(temp[0])
			
			questions.append(temp)
			

		else:
			inputs = list(OrderedDict.fromkeys(inputs))
			del(inputs[0])
			if len(inputs) < max_length :
				while(len(inputs)<max_length):
					inputs.append(unk) 			
			
			sentences.append(inputs) 

		
		sequence = sequence + 1 
		
		

	return sentences, questions, labels 


def val_data_process():
	
	#test_file = open("./dataset/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt","r").read().lower()
	test_file = open("./dataset/own_val.txt","r").read().lower()
	test_file = test_file.split('\n')
	del(test_file[-1])

	inputs = [] 
	sentences = []
	questions = [] 
	labels = [] 
	sequence = 0 
	max_length = 7
	unk = "<zero>"


	for sen in test_file:
		inputs = re.split('(\W+)',sen)
		del(inputs[0])
		del(inputs[-1])

		if sequence % 3 == 2:
			temp = ''.join(str(e) for e in inputs)
			temp = temp.split('\t')
			del(temp[-1])
			labels.append([temp[-1]])
			del(temp[-1])
			temp = re.split('(\W+)',temp[0])

			temp = list(OrderedDict.fromkeys(temp))
			temp[-1] = "?"
			del(temp[0])
			del(temp[0])
		
			questions.append(temp)

		else:
			inputs = list(OrderedDict.fromkeys(inputs))
			del(inputs[0])
			if len(inputs) < max_length:
				while(len(inputs) < max_length):
					inputs.append(unk)
			
			sentences.append(inputs)
		
		sequence = sequence + 1 



	return sentences,questions,labels 



def test_data_process():

	
	test_file = open("./dataset/own_test2.txt","r").read().lower()
	test_file = test_file.split('\n')
	del(test_file[-1])

	inputs = [] 
	sentences = []
	questions = [] 
	labels = [] 
	sequence = 0 
	max_length = 7
	unk = "<zero>"


	for sen in test_file:
		inputs = re.split('(\W+)',sen)
		del(inputs[0])
		del(inputs[-1])

		if sequence % 3 == 2:
			temp = ''.join(str(e) for e in inputs)
			temp = temp.split('\t')
			del(temp[-1])
			labels.append([temp[-1]])
			del(temp[-1])
			temp = re.split('(\W+)',temp[0])

			temp = list(OrderedDict.fromkeys(temp))
			temp[-1] = "?"
			del(temp[0])
			del(temp[0])
		
			questions.append(temp)

		else:
			inputs = list(OrderedDict.fromkeys(inputs))
			del(inputs[0])
			if len(inputs) < max_length:
				while(len(inputs) < max_length):
					inputs.append(unk)
			
			sentences.append(inputs)
		
		sequence = sequence + 1 



	return sentences,questions,labels 



