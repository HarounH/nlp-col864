from __future__ import absolute_import

import os
import re
import numpy as np
import pdb
from collections import Counter

def read_dstc2_data_template(fname):
	print('Reading ', fname)
	data=[]
	with open(fname) as f:
		mem = []
		query_results = {} # resto_name -> list of words(str)
		for line in f.readlines():
			if line.strip():
				nid, line = line.split(' ', 1)
				nid = int(nid)
				line = line.rstrip('\r\n ')
				if '\t' in line:
					# Handle query_results
					'''
					if len(query_results)>0:
						# Append them to memory
						for key in query_results:
							mem.append(query_results[key])
						query_results = {}
					'''
					q,a,aidx = line.split('\t')
					q = q.split(' ')
					aidx = int(aidx)
					data.append((mem[:], q, a, aidx))
					mem.append(q)
					mem.append(a.split(' '))
				elif line=='api_call no result': # Unmodified no result line.
					mem.append(['api_call','no_result'])
				else:
					toks = (line.split(' '))
					'''
					resto_name = toks[0]
					if not(resto_name in query_results):
						query_results[resto_name] = [resto_name]
					query_results[resto_name].append(toks[2].rstrip('\r\n '))
					'''
					mem.append(toks)
			else:
				'''
				if len(query_results)>0:
					# Append them to memory
					for key in query_results:
						mem.append(query_results[key])
						query_results = {}
				'''
				mem = []
		'''
		if len(query_results)>0:
			# Append them to memory
			for key in query_results:
				mem.append(query_results[key])
				query_results = {}
		'''
	return data

def vectorize_data_template(raw_data, max_sentence_length, n_memory_cells, word2idx, template2idx):
	'''
		Takes in raw_data and converts it into vectorised form depending on word2idx, template2idx
		max_sentence_length also accomodates for time and utterer .
	'''
	contexts = []
	queries = []
	answers = []

	for context, query, answer, answeridx in raw_data:
		cc = [] # vectorised context
		# q = [] # vectorised query
		# a = [] # vectorised answer

		# sentence is a list of words.
		for i, sentence in enumerate(context, 1): # What is enumerate(context,1)?? :P
			padding_size = max(0, max_sentence_length - len(sentence)) 
			cc.append([ word2idx[w] for w in sentence ] + [0]*padding_size )
			cc[i-1][-1] = 2 + i%2 # utterer Either system or user.... not 0 because 0 means empty for me.


		# Ignore really old sentences that don't fit in memory.
		cc = cc[::-1][:n_memory_cells][::-1]

		for i in range(len(cc)):
			cc[i][-2] = 4 + (n_memory_cells - i - 1) # Range from 1 to n_memory_cells 

		n_empty_cells = max(0, n_memory_cells - len(cc))
		for _ in range(n_empty_cells):
			cc.append([0]*max_sentence_length) # Empty memory.
		# pdb.set_trace()
		
		lq = max(0, max_sentence_length - len(query)) # >=0 
		q = [word2idx[w] for w in query] + [0] * lq # Padded with 0s. 

		a = np.zeros(len(template2idx)) # 0 is NOT reserved for nil word
		a[answeridx] = 1

		contexts.append(cc)
		queries.append(q)
		answers.append(a)

	return np.array(contexts), np.array(queries), np.array(answers)


def ohe_templates(template2idx, word2idx, extra_words_count):
	nwords = len(word2idx) + extra_words_count
	retval = np.zeros((nwords, len(template2idx)), dtype=np.float32)

	# pdb.set_trace()

	for template,template_idx in template2idx.items():
		toks = template.rstrip('\r\n ').split(' ')
		word_list= []
		for i in range(0, len(toks)):
			if toks[i]=='?':
				continue
			if len(toks[i])==0 or toks[i][0]=='{':
				continue
			word_list.append(toks[i].lower())
		for word in word_list:
			word_idx = word2idx[word]
			temp = retval[word_idx]
			val = temp[template_idx]
			retval[word_idx][template_idx] = 1

	return retval