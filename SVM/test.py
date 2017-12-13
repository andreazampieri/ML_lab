#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, learning_curve
from sklearn.model_selection import GridSearchCV
import threaded_svm as tSVC

# digits = load_digits()

# x = digits.data
# y = digits.target

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# possible_parameters = {
#     'C': [1e0, 1e1, 1e2, 1e3],
#     'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
# }

# svc = SVC(kernel='rbf')

# # The GridSearchCV is itself a classifier
# # we fit the GridSearchCV with the training data
# # and then we use it to predict on the test set
# clf = GridSearchCV(svc, possible_parameters, n_jobs=4) # n_jobs=4 means we parallelize the search over 4 threads
# clf.fit(x_train, y_train)

# y_pred = clf.predict(x_test)
# accuracy = metrics.accuracy_score(y_test, y_pred)
# report = metrics.classification_report(y_test,y_pred)
# cm = metrics.confusion_matrix(y_test,y_pred)
# print(accuracy)
# print(report)
# print(cm)
# print(clf.best_params_)

base = 'sklearn-lab-material/ocr/'
test_data_path = base +'test-data.csv'
test_targets_path = base +'test-targets.csv'
train_data_path = base +'train-data.csv'
train_target_path = base + 'train-targets.csv'

def slice(data,cardinal,tot):
	dim = len(data)
	a = int(dim*cardinal*1.0/tot)
	b = int(dim*(cardinal+1)*1.0/tot)
	return np.append(data[0:a],data[b:dim])

with open(train_data_path) as file:
	train_data = []
	for line in file:
		train_data.append([int(_) for _ in line.strip().split(',')])

with open(train_target_path) as file:
	train_target = []
	for line in file:
		train_target.append(line.strip())

with open(test_data_path) as file:
	test_data = []
	for line in file:
		test_data.append([int(_) for _ in line.strip().split(',')])

train_data = np.array(train_data)
train_target = np.array(train_target)
test_data = np.array(test_data)

c = [1,10,100,1000]
#gamma = [0.1,0.05,0.02,0.01]
gamma = [1e-1, 1e-2, 1e-3, 1e-4]

svm = SVC(C=10,gamma=0.1,kernel='rbf')
svm.fit(train_data,train_target)

pred = svm.predict(test_data)
with open(test_targets_path,'w') as file:
 	for v in pred:
 		file.write(str(v)+'\n')

quit()
#######################
svm = []

idxs = np.array(range(len(train_data)))
np.random.shuffle(idxs)

i=0
n_folds = 5
for c_i in c:
	for gamma_i in gamma:
		batch_idx = slice(idxs,i,n_folds)
		svm.append(tSVC.tSVC(SVC(C=c_i,kernel='rbf',gamma=gamma_i),train_data[batch_idx],train_target[batch_idx]))
		i = (i+1)%n_folds

print('Classifiers: {}'.format(len(svm)))
#wait for all training
for clf in svm:
	clf.start()

for clf in svm:
	clf.join()

scores = []
i=0
for clf in svm:
	batch_idx = slice(idxs,i,n_folds)
	scores.append(metrics.accuracy_score(clf.predict(train_data[batch_idx]),train_target[batch_idx]))
	i+=1

best = max(scores)
print(best)
print(str(np.argmax(scores)))
print(svm[np.argmax(scores)].get_params())