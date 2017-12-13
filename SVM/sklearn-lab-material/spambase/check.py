import numpy as np
import matplotlib.pyplot as pl
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score ,train_test_split
import _thread 

inn=[]
out=[]
tst=[]
accuracy_scores = []
gammaset=dict()
infile,outfile,testfile = open('train-data.csv','r') , open('train-targets.csv','r'), open('test-data.csv','r')
inDirty,outDirty,tstDirty = (infile.readlines()) , (outfile.readlines()), (testfile.readlines())
inClean,out,tstClean = [item.replace('\n', '') for item in inDirty] , [int(item.replace('\n', '')) for item in outDirty] , [item.replace('\n', '') for item in tstDirty]
for element in range(len(inClean)) : inn.append(inClean[element].split(','))
for element in range(len(tstClean)) : tst.append(tstClean[element].split(',')) 

in_train,in_test,out_train,out_test = train_test_split(inn,out,test_size=0.2,random_state=49)

clf = SVC(C=7, kernel='rbf', gamma=1)
clf.fit(in_train,out_train)     
pred = clf.predict(in_test)
print(metrics.classification_report(out_test, pred))
print(metrics.accuracy_score(out_test,pred))

kf = KFold(n_splits=7 , shuffle=True, random_state=49)
gamma_values1 =  np.arange(0.02, 1, 0.02)
gamma_values2 =  np.arange(1.02, 2, 0.02)
gamma_values3 =  np.arange(2.02, 3, 0.02)
gamma_values4 =  np.arange(3.02, 4, 0.02)
gamma_values5 =  np.arange(4.02, 5, 0.02)

def fbg(thread,gamma_values):
    for gamma in gamma_values:
        print(gamma)
        clf = SVC(C=7, kernel='rbf', gamma=gamma)
        scores = cross_val_score(clf, in_train, out_train, cv=kf.split(in_train), scoring='accuracy')
        accuracy_score = scores.mean()
        gammaset[format(accuracy_score, '.4f')] = gamma
#        accuracy_scores.append(format(accuracy_score, '.4f'))
        print(thread,gammaset)
        
        
try:
   _thread.start_new_thread(fbg, ("Thread-1", gamma_values1 ) )
   _thread.start_new_thread(fbg, ("Thread-2", gamma_values2 ) )
   _thread.start_new_thread(fbg, ("Thread-3", gamma_values3 ) )
   _thread.start_new_thread(fbg, ("Thread-4", gamma_values4 ) )
   _thread.start_new_thread(fbg, ("Thread-5", gamma_values5 ) )
except:
   print ("Error: unable to start thread")
while 1:
   pass


#best_index = np.array(accuracy_scores).argmax()
#best_gamma = gamma_values[best_index]

#clf = SVC(C=7, kernel='rbf', gamma=best_gamma)
#clf.fit(in_train,out_train)     
#pred = clf.predict(in_test)
#print(metrics.classification_report(out_test, pred))
#print(metrics.accuracy_score(out_test,pred))