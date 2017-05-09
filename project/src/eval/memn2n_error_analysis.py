import os,sys,pdb
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt

class Option:
	def __init__(self, argv):
		if len(argv)<2:
			print(self.usageString(argv))
			exit()
		self.pred_file = argv[1]
	def usageString(self, argv):
		return 'python ' + argv[0] + ' prediction-truth-csv'

def load_data(option):
	predictions = []
	truths = []
	with open(option.pred_file,'r') as f:
		for line in f:
			toks = line.rstrip('\r\n ').split(',')
			predictions.append(int(toks[0]))
			truths.append(int(toks[1]))
	return np.array(predictions), np.array(truths)

def statistics(ps,ts): # np.array
	# Normal accuracy
	acc = accuracy_score(ts, ps)
	print('Accuracy=' + str(acc))

	# Largest confusions
	cm = confusion_matrix(ts, ps)
	most_confused_ij = [-1,-1]
	max_confusion = 0
	for i in range(0, len(cm)):
		for j in range(0, len(cm[i])):
			if i==j:
				continue
			if cm[i,j]>max_confusion:
				max_confusion = cm[i,j]
				most_confused_ij = [i,j]
	print('Most confused pair=' + str(most_confused_ij[0]) + ',' + str(most_confused_ij[1]))
	
	# Now figure out the other numbers.
	matches = [0,0]
	totals = [0,0]
	state = 0
	for i in range(0, len(ps)):
		if ts[i]==68:
			state = 0
		if ts[i]==8:
			state = 1
		if ps[i]==ts[i]:
			matches[state] += 1
		totals[state] += 1
	print('Step 0 accuracy=' + str(float(matches[0])/totals[0]))
	print('Step 1 accuracy=' + str(float(matches[1])/totals[1]))

	# print('api_call counts ' + str(float(np.sum(cm[:][8]))/len(ts)))
	max_misprediction = -1
	max_misprediction_fraction = 0.0
	for i in range(0, len(cm)):
		if (np.sum(cm[:,i]) - cm[i,i])/float(np.sum(cm[:,i]))>max_misprediction_fraction:
			max_misprediction_fraction = (np.sum(cm[:,i]) - cm[i,i])/float(np.sum(cm[:,i]))
			max_misprediction = i
	print('Prediction for ' + str(max_misprediction) + ' went wrong too often, ' + str(float(max_misprediction_fraction)))
	# pdb.set_trace()
	error_vec = [ np.sum(cm[:,i]) - cm[i,i] for i in range(0, len(cm)) ]
	inaccuracy_frac_vec = [ 1 - (float(cm[i,i])/np.sum(cm[:,i])) for i in range(0, len(cm)) ]
	pdb.set_trace()
	# bar chart of inaccuracies
	
	# pdb.set_trace()
if __name__ == '__main__':
	option = Option(sys.argv)
	predictions, truths = load_data(option)
	statistics(predictions, truths)