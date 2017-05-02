import sys,os,pdb;

'''
	This file specifies function about handling templates.
'''


def kbField2templateField(field):
	'''
		Takes in things like 'name' and returns '{-name-}'
	'''
	return '{-' + field + '-}'

def templateField2kbFields(templateField):
	return templateField[2:-2].split('-')

def isTemplateField(s):	
	'''
		Returns True if a certain word is a blank
	'''
	return len(s)>0 and s[0]=='{' and s[-1]=='}'


def getTemplateID(templates, val2slot, answer):
	# pdb.set_trace()
	for idx in range(0, len(templates)):
		correct=True
		
		t = templates[idx][0:]
		# Try to merge templates[i]
		
		if len(t)==len(answer):
			# pdb.set_trace()
			
			for lidx in range(0, len(answer)):
				# pdb.set_trace()
				if answer[lidx]==t[lidx]:
					pass
				elif isTemplateField(t[lidx]):
					# do things
					if answer[lidx] in val2slot:
						slot = val2slot[answer[lidx]]

						if slot.split('-')[1] in t[lidx].split('-'):
							pass
						else:
							correct=False
							break
					else:
						correct = False
						break
				else:
					correct=False
					break
			if correct:
				# print('Success!')
				return idx
		else:
			continue
		
	return -1
