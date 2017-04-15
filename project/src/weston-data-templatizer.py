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

def isTemplateField(s):
	return len(s)>0 and s[0]=='{' and s[-1]=='}'

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

'''
	MORE CODE!
	The part following this uses the templates that we found to modify datafiles.
'''
def get_template_id(templates, val2slot, answer):
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

def handle_file(infile, outfile):
	with open(infile, 'r') as f:
		with open(outfile, 'w') as g:
			for line in f:
				# print(line)
				# if new line, then ignore
				if line == '\n':
					g.write('\n')
					continue
				# Split the line
				nid, line = line.split(' ', 1) # Split once.
				nid = int(nid)
				if '\t' in line:
					q,a = line.split('\t')
					a = a.rstrip('\r\n')

					# merge a with template	
					aidx = get_template_id(templates, val2slot, a.split(' '))
					if (aidx == -1):
						print('Could not templatise this response: \n\t' + a)
						print('While handling ' + infile)
						exit()
					g.write(str(nid) + ' ' + q + '\t' + a + '\t' + str(aidx) + '\n')
				else:
					# These are system responses to you.
					a = line.rstrip('\r\n')
					# aidx = get_template_id(templates, val2slot, a.split(' '))
					# if (aidx == -1):
					# 	print('Could not templatise this response: \n\t' + a)
					# 	exit()
					g.write(str(nid) + ' ' + a + '\n')


for i in range(0, len(infiles)):
	handle_file(infiles[i], outfiles[i])