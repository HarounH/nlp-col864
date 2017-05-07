import json;
import sys,os;
import pdb;

infiles = ['small-dev.txt','small-tst.txt','small-trn.txt', 'small.txt', 'dialog-babi-task6-dstc2-dev.txt','dialog-babi-task6-dstc2-trn.txt','dialog-babi-task6-dstc2-tst.txt']
outfiles= [s.replace('.txt','-template.txt') for s in infiles]
directory = 'weston_baseline/data/dialog-bAbI-tasks'

templates_file = 'dialog-babi-task6-dstc2-templatised-candidates.txt'
vals2slots_file='vals2slots.json'

val2slot = {}
templates = {}
n=0

from template_utils import *

anonymise_slots = ['name', 'phone', 'addr', 'postcode']
anonymise_slots = [kbField2templateField(x) for x in anonymise_slots]

for i in range(0, len(infiles)):
	infiles[i] = directory + '/' + infiles[i]
	outfiles[i] = directory + '/' + outfiles[i]


# Load templates from file.

with open(templates_file,'r') as f:
	for line in f:
		nid, line = line.split(' ', 1)
		line = line.rstrip('\r\n')
		template = line.split(' ')
		templates[n] = template
		n += 1

with open(vals2slots_file) as f:
	val2slot = json.load(f)

def anonymise_string(s, tok2anon_tok):
	global anonymise_slots, val2slot
	'''
		@in s a non-anonymised list of words
		@in tok2anon_tok a dictionary that contains mappings from anonymised things to (type, anonname) pairs
		@return anon_s anonymised version of s, list of words
	'''
	anon_s = []
	for tok in s:
		# if tok is already in tok2anon_tok
		if tok in tok2anon_tok:
			anon_s.append(tok2anon_tok[tok][1])
		else:
			if tok=='ask' and 'Cambridge' in s and 'system' in s:
				anon_s.append('ask')
				continue
			if tok in val2slot:
				slot = val2slot[tok]
				#if slot == '{-food-}' or slot == '{-area-}' or slot == '{-pricerange-}':
				#	anon_s.append(tok)
				#else:
				idx = 1
				for (k,v) in tok2anon_tok.items():
					if v[0] == slot:
						idx += 1

				anon_name = templateField2kbFields(slot)[0] + '_' + str(idx) # [0] works because theres only one, really
				tok2anon_tok[tok] = (slot, anon_name)
				anon_s.append(anon_name)
			else:
				anon_s.append(tok)
	return anon_s

def handle_file(infile, outfile, anonymise=False):
	with open(infile, 'r') as f:
		with open(outfile, 'w') as g:
			anonymisation_dict = {}
			for line in f:
				# if new line, then ignore
				if line == '\n':
					anonymisation_dict = {}
					g.write('\n')
					continue
				# Split the line
				nid, line = line.split(' ', 1) # Split once.
				nid = int(nid)
				if '\t' in line:
					q,a = line.split('\t')
					a = a.rstrip('\r\n')

					# merge a with template	
					aidx = getTemplateID(templates, val2slot, a.split(' '))
					q = ' '.join(anonymise_string(q.split(' '), anonymisation_dict))
					a = ' '.join(anonymise_string(a.split(' '), anonymisation_dict))
					if (aidx == -1):
						print('Could not templatise this response: \n\t' + a)
						print('While handling ' + infile)
						exit()
					if not (q == '<SILENCE>' and 'Cambridge restaurant system' in a):
						g.write(str(nid) + ' ' + q + '\t' + a + '\t' + str(aidx) + '\n')
				else:
					# These are system responses to you.
					a = line.rstrip('\r\n')
					# aidx = getTemplateID(templates, val2slot, a.split(' '))
					# if (aidx == -1):
					# 	print('Could not templatise this response: \n\t' + a)
					# 	exit()
					a = ' '.join(anonymise_string(a.split(' '), anonymisation_dict))
					g.write(str(nid) + ' ' + a + '\n')

for i in range(0, len(infiles)):
	handle_file(infiles[i], outfiles[i])