import json;
import sys,os;
import numpy as np;
import pdb;

ontology_file  ='ontology_dstc2.json'
candidates_file='weston_baseline/data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-candidates.txt'
templates_file='./dialog-babi-task6-dstc2-templatised-candidates.txt'

# read ontology
req_key = 'requestable'
inf_key = 'informable'

real_fields = ['pricerange', 'food', 'name', 'area']
fake_fields = ['addr', 'postcode', 'phone']


val2slot = {}

def txtfield2tmpfield(field):
	return '{ ' + field + ' }'

def isTemplateField(s):
	
	return len(s)>0 and s[0]=='{' and s[-1]=='}'

with open(ontology_file, 'r') as f:
	raw_ontology_data = json.load(f)
	for field in real_fields:
		items = raw_ontology_data[inf_key][field] #  a list
		for item in items:
			item = item.replace(' ','_')
			val2slot[item]=txtfield2tmpfield(field)


	for name in raw_ontology_data[inf_key]['name']:
		val2slot[name.replace(' ','_') + ('_address')] = txtfield2tmpfield('addr')
		val2slot[name.replace(' ','_') + ('_post_code')] = txtfield2tmpfield('postcode')
		val2slot[name.replace(' ','_') + ('_phone')] = txtfield2tmpfield('phone')

val2slot['R_location'] = txtfield2tmpfield('area')
val2slot['R_price'] = txtfield2tmpfield('pricerange')
val2slot['R_cuisine'] = txtfield2tmpfield('food')

# pdb.set_trace()
# string -> field.

templates = {} # int-> (list-of-strings)


def addTemplate(templates, new_template):
	'''
		Takes in int-> (template_t) and template_t.
		returns True if a new template was encountered
			False if unification happened.
	'''
	
	def unifyTemplate(old, new):
		'''
			Returns bool*template_t
		'''
		if len(old)!=len(new):
			return False,None
		unification = []
		for i in range(0, len(old)):
			# if old[i]==new[i]
			if old[i]==new[i]: # Doesnt matter if template or normal string.
				unification.append(old[i])
			elif isTemplateField(old[i]) and isTemplateField(new[i]):
				# Unify!
				oldFields = set(old[i].split(' ')[1:-1])
				newFields = set(new[i].split(' ')[1:-1])
				allFields = oldFields.union(newFields)
				unification.append(txtfield2tmpfield(' '.join(list(allFields))))
			else:
				return False, None
		return True,unification


	for idx in range(0, len(templates)):
		(can, modified_temp) = unifyTemplate(templates[idx], new_template)
		if can:
			templates[idx] = modified_temp
			return True
		else:
			continue
	templates[len(templates)] = new_template
	return False
# templatise candidates
with open(candidates_file, 'r') as f:
	for line in f:
		toks = line.rstrip('\r\n').split(' ')
		modded_toks = []
		for tok in toks:
			if tok in val2slot:
				modded_toks.append(val2slot[tok])
			else:
				modded_toks.append(tok)
		# See if modded_toks is a template we have seen already
		addTemplate(templates, modded_toks)

# pdb.set_trace()
with open(templates_file, 'w') as g:
	for idx in range(0, len(templates)):
		g.write(' '.join(templates[idx]) + '\n')
