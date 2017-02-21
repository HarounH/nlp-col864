import json;
import sys,os;


'''
	Purpose of code: cleanup dstc2 data.
	needs
'''
usage_string = 'USAGE: ' + '[input filename =/Users/dineshraghu/IRL/Data/DSTC/DSTC-2]' + '[output filename = /Users/dineshraghu/IRL/Data/DSTC/DSTC-2-Clean/]'
if len(sys.argv)<2:
	print(usage_string)
	input = '/Users/dineshraghu/IRL/Data/DSTC/DSTC-2'
	output = '/Users/dineshraghu/IRL/Data/DSTC/DSTC-2-Clean/'
else:
	input = sys.argv[1];
	output = sys.argv[2];
try:
    os.stat(output)
except:
	print(output + ' doesn\'t exist ... creating now')
	os.mkdir(output)

count=0;
noDataCount=0;

for dirpath, dirs, files in os.walk(input):	
	for filename in files:
		if filename == 'label.json':
			labelFile = os.path.join(dirpath,filename);
			logFile = os.path.join(dirpath,'log.json');
			# print labelFile;
			with open(labelFile) as label_file:
				label_data = json.load(label_file);
			with open(logFile) as log_file:
				log_data = json.load(log_file);
			label_turns = label_data["turns"];
			log_turns = log_data["turns"];
			
			noData = False;
			lst = []
			nextSlu = "";
			start = True;
			for i in range(len(log_turns)):
				d= {}
				if "transcript" in log_turns[i]["output"]:
					d['system'] = log_turns[i]["output"]["transcript"];
					# print d['system'];
				else:
					noData = True;
				if "transcription" in label_turns[i]:
					d['user'] = label_turns[i]["transcription"];
					# print d['user'];
				else:
					noData = True;
				lst.append(d);

			if noData == True:
				noDataCount=noDataCount+1;
			else:
				count=count+1
				f = open(output + str(count) + '.json' ,'w')
				f.write(json.dumps(lst));
				f.close();
print count;