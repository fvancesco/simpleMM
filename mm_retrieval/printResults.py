import os
import sys

m = sys.argv[1]

print "direction,method,trainsize,testsize,avg_better,best1,best5"
path="/home/fbarbieri/imagemusic/results/mm_retrieval/"
for f in os.listdir(path):
	toPrint = ""
	if "m3" in f:
		if m in f and "_results_" in f:
			s = f.split("_")
			toPrint = s[2]+","+s[0]+","+s[3]+","+s[4].replace(".txt","")+","			
			lines = open(path+f, 'r').read().strip().split("\n")
			toPrint += lines[1]+","+lines[2].split("\t")[0]+","+lines[3].split("\t")[0]
			print toPrint
	else:
		if m in f and "_results_" in f:
			s = f.split("_")
			toPrint = s[3]+","+s[0]+",-,"+s[3].replace(".txt","")+","			
			lines = open(path+f, 'r').read().strip().split("\n")
			toPrint += lines[1]+","+lines[2].split("\t")[0]+","+lines[3].split("\t")[0]
			print toPrint
	
	
	
