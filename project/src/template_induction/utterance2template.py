import sys,os,pdb;
import json,re;
import numpy as np

'''
	py3 script
	Takes in folder containing DSTC-2 Cleaned logs.
	Takes in output directory to which to write templatised logs AND a template file.
'''

slotsKey = u'slots'
systemResponseKey = u'system'
userResponseKey = u'user'
templateIDKey = u'systemTemplateID'
notReallyTemplateKey = u'notReallyTemplate'
LEFT_DELIM = '{'
RIGHT_DELIM= '}'

def templateStr2templateArr(string):
	string = string.lstrip(' ').rstrip(' ')
	stringsUsed = re.split(LEFT_DELIM + '.*?' + RIGHT_DELIM, string)
	attributesUsed = re.findall(LEFT_DELIM + '.*?' + RIGHT_DELIM, string)

	arr = [stringsUsed[0].lstrip(' ').rstrip(' ')]
	
	for i in range(0, len(attributesUsed)):
		arr.append(attributesUsed[i].lstrip(' ').rstrip(' '))
		arr.append(stringsUsed[i+1].lstrip(' ').rstrip(' '))
	return arr

def templateArr2templateStr(arr):	
	return ' '.join(arr).lstrip(' ').rstrip(' ')

# str * str -> 
def unifyTemplateStrings2templateArr(str1, str2):
	list1 = templateStr2templateArr(str1)
	list2 = templateStr2templateArr(str2)
	if not(len(list1)==len(list2)):
		return None
	unifiedArr = []
	for i in range(0, len(list1)):
		if i%2==0: # string
			if list1[i]==list2[i]:
				unifiedArr.append(list1[i])
			else:
				return None
		else: # attribute
			attr1 = list1[i].lstrip(LEFT_DELIM + ' ').rstrip(RIGHT_DELIM + ' ').split(' ')
			attr2 = list2[i].lstrip(LEFT_DELIM + ' ').rstrip(RIGHT_DELIM + ' ').split(' ')
			unifiedAttribute = LEFT_DELIM + ' '.join(list(set(attr1+attr2))) + RIGHT_DELIM
			unifiedArr.append(unifiedAttribute)
	return unifiedArr

def addToTemplates(template, templates): # string * (id->string)
	for key in templates:
		unifiedArr = unifyTemplateStrings2templateArr(template, templates[key])
		if unifiedArr is None:
			continue
		else:
			templates[key] = templateArr2templateStr(unifiedArr)
			return key
	nkey = len(templates)
	templates[nkey] = template
	return nkey

def replaceValuesByAttributes(string, avDict): # str * (string->string)
	attributes = sorted(list(avDict), key=lambda t: -len(avDict[t])) # Longest to shortest!
	newString = string
	for attribute in attributes:
		newString = re.sub(avDict[attribute], LEFT_DELIM + attribute + RIGHT_DELIM, newString, flags=re.I) # re.I ignores case.
	return newString

def main(inputDir, outputDir, templatesFile):
	templates = {} # ID to String
	# Process inputDir
	try:
		os.stat(outputDir)
	except:
		print(outputDir + ' does not exist... creating it')
		os.mkdir(outputDir)
	for dirpath, dirs, files in os.walk(inputDir):
		for filename in files:
			filepath = os.path.join(dirpath,filename);
			with open(filepath) as file:
				data = json.load(file)

			# Handle data
			for exchange in data:
				systemTemplateString = replaceValuesByAttributes(exchange[systemResponseKey], exchange[slotsKey])
				templateID = addToTemplates(systemTemplateString, templates)
				systemTemplateString = templates[templateID]
				exchange[notReallyTemplateKey] = systemTemplateString
				exchange[templateIDKey] = templateID
			# Save the templatised data
			outputpath = os.path.join(outputDir, filename);
			with open(outputpath, 'wb') as f:
				f.write(json.dumps(data, indent=4));
		print('Handled ' + filename)
	# Write templatesDir
	with open(os.path.join(outputDir, templatesFile), 'wb') as f:
		f.write(json.dumps(templates, indent=4))
	for key in templates:
		print(str(key) + ':' + templates[key])

if __name__ == '__main__':
	class CmdOptions:
		def __init__(self, execName):
			self.inputDir = None
			self.outputDir= None
			self.templatesFile= None
			self.usageString = 'USAGE: python ' + execName + ' inputDir outputDir [templatesFileName=templates.json]'
		def load(self, argv):
			if len(argv)<3:
				print(self.usageString)
				exit()
			self.inputDir = argv[1]
			self.outputDir= argv[2]
			if len(argv)<4:
				self.templatesFile = 'templates.json'
			else:
				self.templatesFile = argv[3]
			
	options = CmdOptions(sys.argv[0])
	options.load(sys.argv)
	# testTemplateStr = "Sure ,  {name} is on {addr}"
	# print(templateStr2templateArr(testTemplateStr))
	# print(templateArr2templateStr(templateStr2templateArr(testTemplateStr)))
	# print(templateStr2templateArr(templateArr2templateStr(templateStr2templateArr(testTemplateStr))))
	# exit()
	main(options.inputDir, options.outputDir, options.templatesFile)

