
from __future__ import absolute_import
from __future__ import division 

import sys
import time

import tensorflow as tf 
import numpy as np 


class GRUCell(tf.nn.rnn_cell.RNNCell):

  def __init__(self,input_size,state_size,name):
	self.input_size = input_size
	self._state_size = state_size 
 	self.name = name

  @property
  def state_size(self):
	return self._state_size

  @property
  def output_size(self):
	return self._state_size

  def __call__(self,inputs,state,scope=None):
  

	scope = self.name

	scope = scope or type(self).__name__
        with tf.variable_scope(scope):
            
	    #tf.get_variable_scope().reuse ==True
	    
	  	
            init = tf.contrib.layers.xavier_initializer(tf.int32)
            W_r  = tf.get_variable("W_r",(self._state_size,self._state_size),tf.float32,init)
            U_r  = tf.get_variable("U_r",(self.input_size,self._state_size),tf.float32,init)
            b_r  = tf.get_variable("b_r",(self._state_size),tf.float32,init)

            W_z  = tf.get_variable("W_z",(self._state_size,self._state_size),tf.float32,init)
            U_z  = tf.get_variable("U_z",(self.input_size,self._state_size),tf.float32,init)
            b_z  = tf.get_variable("b_z",(self._state_size),tf.float32,init)

            W_o  = tf.get_variable("W_o",(self._state_size,self._state_size),tf.float32,init)
            U_o  = tf.get_variable("U_o",(self.input_size,self._state_size),tf.float32,init)
            b_o  = tf.get_variable("b_o",(self._state_size),tf.float32,init)

            state = tf.cast(state,tf.float32)
            
	    z_t = tf.sigmoid(tf.matmul(inputs,U_z) + tf.matmul(state,W_z) + b_z )
            r_t = tf.sigmoid(tf.matmul(inputs,U_r) + tf.matmul(state,W_r) + b_r )
            o_t = tf.tanh(tf.matmul(inputs,U_o) + r_t*tf.matmul(state,W_o) + b_o)
            h_t = z_t * state + (1-z_t) * o_t


            new_state = h_t
	    output = new_state
	    

 	    return output,new_state



