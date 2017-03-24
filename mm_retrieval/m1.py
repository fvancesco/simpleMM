#...
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import sys

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

vectors = sys.argv[1]
v4l = int(sys.argv[2])
test_elements = 4649 #int(sys.argv[3]) #4649
users_max = test_elements # int(sys.argv[3]) #total vectors to use

if "train" in vectors:
	fromUser = 0 #we use train vectors
else:
	fromUser = 21384 #we use validation and test

m = 5
print(str(v4l))
print(str(test_elements))

path= "/home/fbarbieri/imagemusic/vectors/501-pred_12_MSD-IGT/mm_retrieval/"
print "Loading vectors..."
mm_audio = np.load(path+"audio_"+vectors.split("_")[0]+".npy")
mm_visual = np.load(path+"visual_"+vectors.split("_")[0]+".npy")
	
#output
out = open("/home/fbarbieri/imagemusic/results/mm_retrieval/"+vectors+"m1_pred_"+str(v4l)+"_"+str(test_elements)+".txt", 'w')
out5 = open("/home/fbarbieri/imagemusic/results/mm_retrieval/"+vectors+"m1_pred5_"+str(v4l)+"_"+str(test_elements)+".txt", 'w')
out_results = open("/home/fbarbieri/imagemusic/results/mm_retrieval/"+vectors+"m1_results_"+str(v4l)+"_"+str(test_elements)+".txt", 'w')

if v4l == 1:
	mmX = mm_audio[fromUser:users_max+fromUser,:]
	mmY = mm_visual[fromUser:users_max+fromUser,:]
else:
	mmX = mm_visual[fromUser:users_max+fromUser,:]
	mmY = mm_audio[fromUser:users_max+fromUser,:]

#normalize vectors
print "Normalizing"
mmX = normalized(mmX,1)
mmY = normalized(mmY,1)

gold = np.zeros(test_elements)
predicted = np.zeros(test_elements) #np.zeros((test_elements,len(Nnn)))

true = 0
false = 0

trueNN = 0
falseNN = 0

n_closer_vectors = 0

print "Starting..."	
for i in range(test_elements):
	#print("t:"+str(i))
	#gold is the same user (but in the other modality)
	gold[i]=i
	
	#compute nearest neighbours of the input vector, then select closest one
	x = mmX[i,:]
	dsims = np.dot(mmX, x.transpose())
	dsims[i] = -1 #exclude yourself!
	ss = np.argsort(dsims * -1)  #sorted similarity
	closest = ss[0]
	#print("c:"+str(ss[0]))

	#take nearest neighbours of closest vector (in the other modality)
	y = mmY[closest,:]
	dsims = np.dot(mmY, y.transpose())
	ss = np.argsort(dsims * -1)  #sorted similarity
	predicted[i] = ss[0]
	n_closer_vectors += ss.tolist().index(i)

	if i == ss[0]: 
		true += 1 
		out.write(str(i)+"\t"+str(closest)+"\n")
		#print("found!!" + str(i)+"\t"+str(closest))
	else: false += 1
		
	if i in ss[:m]: 
		trueNN += 1 
		out5.write(str(i)+"\t"+str(closest)+"\n")
		#print("found5!!" + str(i)+"\t"+str(closest))
	else: falseNN += 1
		
	#print(toPrint)
	#if(i%1000 == 0): print(str(i))

out_results.write("\nv: "+str(v4l)+", test elements: "+str(test_elements)+"\n")
out_results.write(str(n_closer_vectors/test_elements)+"\n")
out_results.write(str(true)+"\t"+str(false) + "\t" + str(true/test_elements)+"\n")
out_results.write(str(trueNN)+"\t"+str(falseNN) + "\t" + str(trueNN/test_elements)+"\n")
out_results.write(str(precision_recall_fscore_support(gold, predicted, average='weighted')))	

out.close()
out5.close()
out_results.close()