
from collections import OrderedDict 
from preprocess import *
from word2vec import *
from gru_cell import *
from numpy.linalg import norm
#from util import minibatches
from util2 import *
from forpp import *


import matplotlib.pyplot as plt 
import time
import numpy as np
import tensorflow as tf


a,b,c = pre_process()
e,f,g = val_data_process() 
h,i,j = test_data_process()

#  a :  vector of ' SENTENCES ' 
#  b :  vector of ' QUESTIONS ' 
#  c :  vector of '  LABELS   '

#  e :  vector of ' V.SENTENCES '
#  f :  vector of ' V.QUESTIONS ' 
#  g :  vector of ' V.LABELS    ' 

#  h :  vector of ' TEST SENTENCES ' 
#  i :  vector of ' TEST QUESTIONS ' 
#  j :  vector of ' TEST LABELS    ' 


# splicing : vector of all words in SENTENCES & QUESTIONS 
# WE'VE GOT WORD_VECTOR


splicing = a + b + e + f + h + i
word_vec , dic_s = word_to_glove(splicing)
reversed_dic = dict(zip(dic_s.values() , dic_s.keys()))
voca_size = len(dic_s)

print dic_s 

# FROM NOW ON, WE'VE MAKE WORD - TO - VECTOR DICTIONARY 
# NOW, LET'S MAKE INPUT AS [ 14,2,1,2,3,10 ... ] 
# FOR USING tf.nn.embedding_lookup ! 

idxes_s = word_to_idx(a,dic_s)	# THIS IS WHOLE IDXES OF SENTENCES. 
idxes_q = word_to_idx(b,dic_s)
idxes_a = word_to_idx(c,dic_s)
idxes_a = np.reshape(idxes_a,(-1,1,1))		# JUST IN CASE 


idxes_st = word_to_idx(e,dic_s)
idxes_qt = word_to_idx(f,dic_s)
idxes_at = word_to_idx(g,dic_s)
idxes_at = np.reshape(idxes_at,(-1,1,1)) 	# JUST IN CASE 

idxes_test_s = word_to_idx(h,dic_s)
idxes_test_q = word_to_idx(i,dic_s)
idxes_test_a = word_to_idx(j,dic_s)
idxes_test_a = np.reshape(idxes_test_a,(-1,1,1))


num_sen = np.shape(idxes_q)[0]			# The number of trinaing ques
num_val = np.shape(idxes_qt)[0] 		# The number of validation ques 
num_test = np.shape(idxes_test_q)[0]		# The number of test ques


idxes_s = np.reshape(idxes_s,(num_sen,-1))	
idxes_st = np.reshape(idxes_st,(num_val,-1))
idxes_test_s  = np.reshape(idxes_test_s,(num_test,-1))


inputs = idxes_s
#test_inputs = idxes_st 
val_inputs = idxes_st
test_inputs = idxes_test_s


#s1 =  word_vec[idxes_s[0]]	# make word_vectors of SENTENCE_1 
#s2 =  word_vec[idxes_s[1]]	# make word_vectors of SENTENCE_2 

#########################################
#  MAKING VOCA_DATA IN GLOVE VECTOR 	#
#########################################

#print inputs
re_inputs = inputs[:,::-1]
re_test   = val_inputs[:,::-1]
re_tt	  = test_inputs[:,::-1]

#print r_inputs

s_inputs = np.asarray([])
s_inputs = word_vec[inputs]

r_inputs  = np.asarray([])
r_inputs = word_vec[re_inputs]


idxes_q = np.reshape(idxes_q,(num_sen,-1))
idxes_qt = np.reshape(idxes_qt,(num_val,-1))
idxes_test_q = np.reshape(idxes_test_q,(num_test,-1))

q  = word_vec[idxes_q]
ans = np.asarray(word_vec[idxes_a],np.float32)  # ............ Train data ...............


t_inputs = np.asarray([])
t_inputs = word_vec[val_inputs]
tr_inputs = np.asarray([])
tr_inputs = word_vec[re_test]

t_q = word_vec[idxes_qt]
t_ans = np.asarray(word_vec[idxes_at],np.float32) # ........... Val data ...............


v_inputs = np.asarray([])
v_inputs = word_vec[test_inputs]
vr_inputs = np.asarray([])
vr_inputs = word_vec[re_tt]

v_q = word_vec[idxes_test_q]
v_ans = np.asarray(word_vec[idxes_test_a],np.float32) # .......... Test Data .............



	#################################################
		#  #  #  TENSORFLOW # # # # 
	#################################################

 # MAKING OF ALL INPUT'S PLCAEHOLDER 

input_placeholder = tf.placeholder(tf.float32,[None,14,200])
r_input_placeholder = tf.placeholder(tf.float32,[None,14,200])
labels_placeholder = tf.placeholder(tf.float32,[None,1,1])
question_placeholder = tf.placeholder(tf.float32,[None,4,200])
temp_placeholder = tf.placeholder(tf.int32,[None,])
dropout_rate	= tf.placeholder("float")


init = tf.contrib.layers.xavier_initializer(tf.int32)
W_s2 = tf.get_variable("W_s2",(190,voca_size),tf.float32,init)
bias2 = tf.get_variable("bias2",(voca_size),tf.float32,init)
W_concat = tf.get_variable("W_concat",(600,300),tf.float32,init)
bias_concat = tf.get_variable("bias_concat",(300),tf.float32,init)

#last_sr = tf.concat(1,[last_s,last_r]) 	# [Batch_size , 600]
cell_s = GRUCell(200,190,"s")
cell_q = GRUCell(200,190,"q")
cell_q_s = GRUCell(200,190,"q_s")


x = input_placeholder 
y = labels_placeholder
z = question_placeholder
t = temp_placeholder
r = r_input_placeholder


	#############################################
      # # #  GRUCell with SENTENCES & QUESTIONS # # # # #
	############################################# 

_,last_s = tf.nn.dynamic_rnn(cell=cell_s,dtype=tf.float32,
				time_major=False,inputs=x)

_, last_q = tf.nn.dynamic_rnn(cell=cell_q,dtype=tf.float32,
			time_major=False,initial_state = last_s,inputs=z)


_,last_q_s = tf.nn.dynamic_rnn(cell=cell_q_s,dtype=tf.float32,
				time_major=False,initial_state=last_q,inputs=x)

#predic_1 = tf.concat(1,[last_q_s,last_q])
predic_1 = last_q_s + last_q
predic_drop = tf.nn.dropout(predic_1,dropout_rate)
score     = (tf.matmul(predic_drop,W_s2) + bias2)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(score,t))
reg1  = 0.01 * tf.nn.l2_loss(W_s2)
cost = tf.reduce_mean(loss + reg1)

global_step = tf.Variable(0,trainable=False)
starter_learning_rate = 5e-3
lr_rate = tf.train.exponential_decay(starter_learning_rate,global_step,
					300,0.1,staircase=True)

optimizer = tf.train.AdamOptimizer(lr_rate).minimize(cost,global_step=global_step)


batch_size = 85
val_batch_size = 10
test_batch_size =10
val_trial = 10
d_rate = 0.7


  # Training begins . 

start_time = time.time()
loss_history = []
train_history = [] 
val_history = [] 
test_history = [] 

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())


	#######################################	
	# 	Training on-batch !	      #
	#######################################


	for step in range(501):

		s_bat,r_bat,q_bat,ans_bat,_ = generate_batch2(batch_size,s_inputs,r_inputs,q,idxes_a)
		sess.run(optimizer,feed_dict={input_placeholder : s_bat,
					     question_placeholder : q_bat,
					      temp_placeholder : ans_bat,
					dropout_rate : d_rate
					 })

		
		if step % 5 == 0:
	
			count_t = 0 
			mask_num_t = 0



			loss_input  = sess.run(cost,feed_dict={input_placeholder : s_bat,
						question_placeholder : q_bat,	
						temp_placeholder:ans_bat,
						dropout_rate : d_rate})
			


			#print "{0} step loss : ".format(step),loss_input
			loss_history.append(loss_input)	

			train_pred = sess.run(score,feed_dict = {input_placeholder : s_bat,
								 question_placeholder : q_bat,
								 dropout_rate : 1.0})
			train_ans = np.argmax(train_pred, axis=1)

			for idx in train_ans:
				if(reversed_dic[idx] == reversed_dic[ans_bat[mask_num_t]]):
					count_t += 1
				mask_num_t +=1 
			train_history.append((count_t*100) / batch_size)






		
			acc_avg = []
			
			for step in range(val_trial):
				count = 0
				mask_num = 0 
 				st_bat, _ , qt_bat,at_bat ,mask = generate_batch2(val_batch_size,t_inputs,tr_inputs,
									t_q,idxes_at)
				ans_pred = sess.run(score,feed_dict={input_placeholder : st_bat,
									question_placeholder : qt_bat,
									dropout_rate : 1.0})
				ans = np.argmax(ans_pred,axis=1)		

				
				for idx in ans:
			
					sen = []
					que = []			
			
					if(reversed_dic[idx] == reversed_dic[at_bat[mask_num]]):
							count+=1
					mask_num+=1
				
				acc_avg.append((count*100) / val_batch_size)
				#val_history.append((count*100)/test_batch_size)
			val_history.append(np.mean(acc_avg))

	

			for epo in range(10):
				#for ee in range(5):
					acc_avg = [] 
					count = 0
					mask_num = 0

					test_s_bat, _ , test_q_bat, test_a_bat , mask = generate_batch2(test_batch_size, v_inputs, vr_inputs,
												v_q,idxes_test_a)		# add 
		
					pred = sess.run(score,feed_dict={input_placeholder : test_s_bat,
										question_placeholder : test_q_bat,
										dropout_rate : 1.0})
	
					ans = np.argmax(pred,axis=1)
		
					for idx in ans:
				
						sen=[]
						que=[]	
					
						#print "sentence : "
						i = mask[mask_num]
						ii = idxes_test_s[i]
						qq = idxes_test_q[i]
					
						for tp in ii:
							sen.append(reversed_dic[tp])
						#print sen
					
						for qp in qq:
							que.append(reversed_dic[qp])
						#print que 

						#print " Prediction : " , reversed_dic[idx]		# add 
						#print " Answer     : " , reversed_dic[test_a_bat[mask_num]]
				
						if(reversed_dic[idx] == reversed_dic[test_a_bat[mask_num]]):
							count+=1
						mask_num += 1 

					acc_avg.append((count*100) / test_batch_size)
					#print " ======================================================================"
			test_history.append(np.mean(acc_avg))
			if(np.mean(acc_avg) == 100):
				print " This epoch reaches 100% accuracy ! " 


	print("Training Done !\n")
	print("---------%s seconds --------------" %(time.time() - start_time))

	plt.subplot(3,1,1)
	plt.plot(loss_history,'-o',color='r')
	plt.xlabel('iteration / 100')
	plt.ylabel('loss')
	
	plt.subplot(3,1,2)
	plt.plot(train_history,'-o')
	plt.plot(val_history,'-o')
	plt.legend(['train','val'],loc='lower right')
	plt.xlabel('Sampling trial')
	plt.ylabel('Accuracy')
	
	plt.subplot(3,1,3)
	plt.plot(test_history,'-o')
	plt.legend(['test'],loc = 'lower right')
	plt.xlabel('Test Sampling Trial')
	plt.ylabel('Accuracy')


	plt.show()
	



	"""
	for i in range(5):

		st_bat,qt_bat,at_bat,mask = generate_batch(test_batch_size,t_inputs,t_q,t_ans)	
		ans_pred = sess.run(predic,feed_dict={input_placeholder : st_bat,
						question_placeholder : qt_bat})
	"""
	#	for i in mask:
	#		print idxes_at[i]
	

	#print " Cosine sim : " ,sess.run(cos_sim,feed_dict={input_placeholder : st_bat,
	#					question_placeholder : qt_bat})	
     
	#a = sess.run(final_label,feed_dict={input_placeholder : st_bat,
	#					question_placeholder:qt_bat})
	
	#print " argmax label : " , a
	#count = 0 	

	#for idx in a :
		#print "prediction : ",reversed_dic[idx]
		#print " answer    : ",g[idx]	
		
		#if reversed_dic[idx] == g[idx][0] :
		#	count+=1 	


	#print " accuracy : " , (count / test_batch_size)*100 

			#print " preds : " , sess.run(preds,feed_dict={input_placeholder : s_inputs,
			#					labels_placeholder : ans})
			
			#print "cos_Sim : " ,sess.run(cos_sim,feed_dict= {input_placeholder : s_inputs,
			#					question_placeholder : q , 
			#					labels_placeholder : ans})

	#print "pred_s : " , sess.run(pred_s,feed_dict={input_placeholder : s_inputs})
	
	#print "pred_q : " , sess.run(pred_q,feed_dict={input_placeholder : s_inputs }) 



			#print " W_s : " ,sess.run(W_s)
			#print " W_q : " , sess.run(W_q)

			#print "predic : " ,sess.run(predic,feed_dict={input_placeholder : s_bat,
			#					question_placeholder : q_bat })

		

			#print " label - predic " , sess.run(what,feed_dict={input_placeholder : s_bat,
                        #                        question_placeholder : q_bat,
                        #                        labels_placeholder:ans_bat})
			#print "gradients : " , sess.run(gradients,feed_dict={input_placeholder :s_bat,
			#						question_placeholder:q_bat,
			#					`	labels_placeholder :ans_bat})







