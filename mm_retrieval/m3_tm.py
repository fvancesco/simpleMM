import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import sys

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

print "Loading vectors..."

vectors = sys.argv[1]
v4l = int(sys.argv[2])
train_size = 21384	#int(sys.argv[3]) #total vectors to use for training
test_size = 4649	#int(sys.argv[4]) #np.shape(trainX)[0]  #elements to test

# = 1000 #total vectors to use
print(str(v4l))
print(str(test_size))

if "train" in vectors:
	fromUser = 0 #we use train vectors for testing
else:
	fromUser = 21384 #we use validation and test

path= "/home/fbarbieri/imagemusic/vectors/501-pred_12_MSD-IGT/mm_retrieval/"
print "Loading vectors..."
mm_audio = np.load(path+"audio_"+vectors.split("_")[0]+".npy")
mm_visual = np.load(path+"visual_"+vectors.split("_")[0]+".npy")

#output
out_results = open("/home/fbarbieri/imagemusic/results/mm_retrieval/"+vectors+"m3_results_"+str(v4l)+"_"+str(train_size)+"_"+str(test_size)+".txt", 'w')

if v4l == 1:
	trainX = mm_audio[:train_size,:]
	trainY = mm_visual[:train_size,:]
	testX = mm_audio[fromUser:fromUser+test_size,:]
	testY = mm_visual[fromUser:fromUser+test_size,:]
else:
	trainX = mm_visual[:train_size,:]
	trainY = mm_audio[:train_size,:]
	testX = mm_visual[fromUser:fromUser+test_size,:]
	testY = mm_audio[fromUser:fromUser+test_size,:]

#normalizing
gold = np.zeros(test_size)
predicted = np.zeros(test_size) #np.zeros((test_size,len(Nnn)))

true = 0
false = 0

trueNN = 0
falseNN = 0

n_closer_vectors = 0

print "Learning TM..."	
translation_matrix = np.linalg.pinv(trainX).dot(trainY).T

#print "Normalizing..."
testX = normalized(testX,1)
testY = normalized(testY,1)

print "Testing..."
for i in range(test_size):
	gold[i]=i
	
	x = testX[i,:]
	pred_vec = translation_matrix.dot(x)  #predicted vector in the other space
	pred_vec = pred_vec / np.linalg.norm(pred_vec)
 
	dsims = np.dot(testY, pred_vec.transpose()) #both testY and pred_vec are normalised so this is cos sim
	ss = np.argsort(dsims * -1)  #sorted similarity
	predicted[i] = ss[0]
	n_closer_vectors += ss.tolist().index(i)

	if i == ss[0]: 
		true += 1 
		#out.write(str(i)+"\t"+str(closest)+"\n")
		#print("found!!" + str(i)+"\t"+str(closest))
	else: false += 1
		
	if i in ss[:5]: 
		trueNN += 1 
		#out5.write(str(i)+"\t"+str(closest)+"\n")
		#print("found5!!" + str(i)+"\t"+str(closest))
	else: falseNN += 1
		
	#print(toPrint)
	#if(i%1000 == 0): print(str(i))

out_results.write("\nv: "+str(v4l)+", test elements: "+str(test_size)+"\n")
out_results.write(str(n_closer_vectors/test_size)+"\n")
out_results.write(str(true)+"\t"+str(false) + "\t" + str(true/test_size)+"\n")
out_results.write(str(trueNN)+"\t"+str(falseNN) + "\t" + str(trueNN/test_size)+"\n")
out_results.write(str(precision_recall_fscore_support(gold, predicted, average='weighted')))	

out_results.close()