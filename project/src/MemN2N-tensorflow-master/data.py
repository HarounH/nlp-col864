from __future__ import absolute_import

import os
import re
import numpy as np
from collections import Counter

def read_data(fname, count, word2idx):
	if os.path.isfile(fname):
		with open(fname) as f:
			lines = f.readlines()
	else:
		raise("[!] Data %s not found" % fname)

	words = []
	for line in lines:
		words.extend(line.split())

	if len(count) == 0:
		count.append(['<eos>', 0])

	count[0][1] += len(lines)
	count.extend(Counter(words).most_common())

	if len(word2idx) == 0:
		word2idx['<eos>'] = 0

	for word, _ in count:
		if word not in word2idx:
			word2idx[word] = len(word2idx)

	data = list()
	for line in lines:
		for word in line.split():
			index = word2idx[word]
			data.append(index)
		data.append(word2idx['<eos>'])

	print("Read %s words from %s" % (len(data), fname))
	return data

def read_dstc2_data(fname):
	print('Reading ', fname)
	data=[]
	with open(fname) as f:
		mem = []
		for line in f.readlines():
			if line.strip():
				nid, line = line.split(' ', 1)
				nid = int(nid)
				line = line.rstrip()
				if '\t' in line:
					q,a= line.split('\t',1)
					q = q.split(' ')
					data.append((mem[:], q, a))
					mem.append(q)
					mem.append(a.split(' '))
				else:
					mem.append(line.split(' '))
			else:
				mem = []
	return data

def vectorize_data(raw_data, max_sentence_length, n_memory_cells, word2idx, candidate2idx):
	'''
		Takes in raw_data and converts it into vectorised form depending on word2idx, candidate2idx

		max_sentence_length also accomodates for time and utterer .
	'''
	contexts = []
	queries = []
	answers = []

	for context, query, answer in raw_data:
		c = [] # vectorised context
		# q = [] # vectorised query
		# a = [] # vectorised answer

		# sentence is a list of words.
		for i, sentence in enumerate(context, 1): # What is enumerate(context,1)?? :P
			padding_size = max(0, max_sentence_length - len(sentence)) 
			c.append([ word2idx[w] for w in sentence ] + [0]*padding_size)
			c[i-1][-1] = 2 + i%2 # utterer Either system or user.... not 0 because 0 means empty for me.


		# Ignore really old sentences that don't fit in memory.
		c = c[::-1][:n_memory_cells][::-1]

		for i in range(len(c)):
			c[i][-2] = 4 + (n_memory_cells - i - 1) # Range from 1 to n_memory_cells 

		n_empty_cells = max(0, n_memory_cells - len(c))
		for _ in range(n_empty_cells):
			c.append([0]*max_sentence_length) # Empty memory.

		lq = max(0, max_sentence_length - len(query)) # >=0 
		q = [word2idx[w] for w in query] + [0] * lq # Padded with 0s. 

		a = np.zeros(len(candidate2idx) + 1) # 0 is reserved for nil word
		a[candidate2idx[answer]] = 1

		contexts.append(c)
		queries.append(q)
		answers.append(a)

	return np.array(contexts), np.array(queries), np.array(answers)
