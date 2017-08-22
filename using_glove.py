
import numpy as np
import glove as glove
import sys
import re 
import collections 
import time 

reload(sys)
sys.setdefaultencoding('utf8')

#context = open('./text.txt','r').read()
#context = context.split('\n')
#context = unicode(context,errors = 'ignore')
#print context

words = [] 
for sen in context :
	words = words + re.split('(\W+)',sen)
words = set(words)

voca_size = len(words)
print("Size of Voca : ",voca_size)



count = [['UNK',-1]]
count.extend(collections.Counter(words).most_common(voca_size))
dictionary = dict()
for word in words:
        dictionary[word] = len(dictionary)


voca_size = len(dictionary)
wordVectors = glove.loadWordVectors(dictionary)
 
# wordVectors << vectors of text file "sample.txt" ! 


