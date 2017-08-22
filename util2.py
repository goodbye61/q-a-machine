

import numpy as np 
import time 
from numpy.linalg import norm 
from numpy import dot

def generate_batch(batch_size,sentence,question,ans):
	
	num_train = np.shape(sentence)[0]
	batch_mask = np.random.choice(num_train,batch_size)
	#batch_mask = np.asarray([0])
	sen_batch = sentence[batch_mask]
	que_batch = question[batch_mask]
	ans_batch = ans[batch_mask]
	
	ans_batch = np.reshape(ans_batch,(-1))


	return sen_batch,que_batch,ans_batch,batch_mask


def cosine_similarity(a,b):

	# a : ans_pred 
 	# b : word_vector 
 
	# a : [batch_size, 50] 
	# b : [voca_size , 50]

	pred = []
	score = [] 

	for vec in a:
		temp = (dot(b,vec.T) / (norm(vec) * norm(b)))
		print temp
		max_score = np.argmax(temp,axis=0)		
		pred.append(max_score)
		
	


	#print pred

"""
a = np.arange(100).reshape(10,1,10)
b = np.array([100,101,102,103,140,105,106,107,108,109])
sim = [] 

for vec in a:
	
	temp = np.float(dot(vec,b.T) / (norm(vec) * norm(b)))
	sim.append(temp)

print np.shape(sim)	


print np.argmax(sim)

#sim = np.dot(a,b.T)/(norm(a) * norm(b))
#print sim

"""
