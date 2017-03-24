import sys

v = sys.argv[1]
prediction = open("./results/pred5_"+v+"_1000.txt", 'r').read().strip().split("\n")
pic_index = open("/home/francesco.barbieri/FEWUSER-MULTIMODALTWITTER/out/vectors/250users/ok/index_pics.txt", 'r').read().strip().split("\n")
out = open("./results/html/pred5_"+v+"_1000.html", 'w')

out.write("<html>")
basePics = "../yoavModel/ppics/"

for line in prediction:
	pred = line.split("\t")	
	print pred[0]
	print pred[1]
	pic1 = basePics+pic_index[int(pred[0])].replace("\r","").split("/")[-1]
	pic2 = basePics+pic_index[int(pred[1])].replace("\r","").split("/")[-1]
	print pic1
	print pic2
	out.write("<img src="+pic1+"> <img src="+pic2+">"+"\n")
	out.write("<hr>"+"\n")
	
out.write("</html>")
out.close()