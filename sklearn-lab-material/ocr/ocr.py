#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, learning_curve

def view(pxls):
	height=16
	width=8
	# for i in range(height):
	# 	s=""
	# 	for j in range(width):
	# 		s += ("." if pxls[width*i+j]==0 else "0")
	# 	print(s)
	img = np.array(pxls)
	img = img.reshape((height,width))
	# plt.plot(img)
	# plt.show()
	plt.imshow(img)
	plt.show()

test_data_path = 'test-data.csv'
test_targets_path = 'test-targets.csv'
train_data_path = 'train-data.csv'
train_target_path = 'train-targets.csv'


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


kf = KFold(n_splits=5, shuffle=True, random_state=42)

clf = SVC(C=10,kernel='linear',gamma=0.02)
#clf.fit(train_data,train_target)
scores = cross_val_score(clf,train_data,train_target,cv=kf.split(train_data),scoring='accuracy')

print(scores)
print(scores.mean())	# [0.80790893  0.80285235  0.80488974  0.81136146  0.81124161]
						# 0.80765081714


# test_prediction = clf.predict(test_data)

# with open(test_targets_path,'w') as file:
# 	for v in test_prediction:
# 		file.write(str(v)+'\n')