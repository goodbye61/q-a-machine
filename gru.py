


from __future__ import absolute_import 
from __future__ import division 

import sys
import time 

import tensorflow as tf
import numpy as np 

from q3_gru_cell import GRUCell 



class Config:
	
 	max_length = 20 
	batch_size = 100 
	n_epochs = 40 
	lr = 0.2 
	max_grad_norm = 5. 
	n_classes = 22 	
	hidden_size = 200


class SequencePredictor(Model):

  def add_placeholders(self):

	# Make input ~ label placeholder ! 
	self.inputs_placeholder = tf.placeholder(tf.float32,[None,50])
	self.labels_placeholder = tf.placeholder(tf.float32,[1,50])	# labels as vector 
	#self.labels_placeholder = tf.placeholder(tf.float32,[1,1]) 	# labels as idx 
	


  def create_feed_dict(self,inputs_batch,labels_batch=None):


	# Make FEED DICT ! 

  	feed_dict = {self.inputs_placeholder : inputs_batch}
	if labels_batch in not None:
		feed_dict[self.labels_placeholder] = labels_batch
	
	return feed_dict 	



  def add_prediction_op(self):

	cell = GRUCell(50,self.config.hidden_size)        	# MEANING OF  ( a,b ) ? : 
	
	x = self.inputs_placeholder 
	init = tf.zeros(tf.shape(x[:,0,:])) 		# initial_state (Zero-state) 
	output,last_state = tf.nn.dynamic_rnn(cell=cell,dtype=tf.float32,initial_state=init,inputs=x)
	preds = tf.sigmoid(last_state)


	return preds 

  def add_loss_op(self,preds):

	y = self.labels_placeholder
	loss = tf.reduce_mean(tf.nn.l2_loss(y-preds))

	return loss 



  def add_training_op(self,loss):

	optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
	gradients = optimizer.compute_gradients(loss)
	grad = [i[0] for i in gradients]
	var  = [i[1] for i in gradients]

	self.grad_norm = tf.global_norm(grad)
	if self.config.clip_gradients == True:
		grad,self.grad_norm = tf.clip_by_global_norm(grad,self.config.max_grad_norm)

	train_op = optimizer.apply_gradients(zip(grad,var))


	return train_op 



  def run_epoch(self,sess,train):
	
	losses,grad_norms = [],[]
	for i batch in enumerate(minibatches(train,self.config.batch_size)):
		loss,grad_norm = self.train_on_batch
		losses.append(loss)
		grad_norms.append(grad_norm)
		

	return losses , grad_norms



  def fit(self,sess,train):
	losses,grad_norms = [],[]
	for epoch in range(self.config.n_epochs):
	
		loss,grad_norm = self.run_epoch(sess,train)
		losses.append(loss)
		grad_norms.append(grad_norm)

	return losses , grad_norms 



  def __init__(self,config):
	self.config = config
	self.inputs_placeholder = None 
	self.labels_placeholder = None 
	self.grad_norm = None 
	self.build() 








	
	










		


	









  
  












