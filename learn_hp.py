#!/usr/bin/env python3
import argparse
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn import metrics
from sklearn.svm import SVC


#learn_hp.py config_file
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('config_file')
	args = parser.parse_args()

	with open(args.config_file,'r') as file:
		opt = {}
		for line in file:
			k,v = line.strip().split(':')
			opt[k] = v

	if 'c' in opt:
		key = 'c'
		tmp = opt[key]
		opt[key] = [float(_) for _ in tmp.split(',')]

	if 'gamma' in opt:
		key = 'gamma'
		tmp = opt[key]
		opt[key] = [float(_) for _ in tmp.split(',')]

	if 'n_jobs' in opt:
		key = 'n_jobs'
		tmp = opt[key]
		opt[key] = int(tmp)

	if 'folds' in opt:
		key = 'folds'
		tmp = opt[key]
		opt[key] = int(tmp)

	if 'kernel' in opt:
		key = 'kernel'
		tmp = opt[key]
		opt[key] = tmp.split(',')

	base_path = opt['base_path']
	train_data_path = base + opt['train_data_path']
	train_labels_path = base + opt['train_labels_path']
	test_data_path = base + opt['test_data_path']
	test_labels_path = base + opt['test_labels_path']

	with open(train_data_path) as file:
		train_data = []
		for line in file:
			train_data.append([int(_) for _ in line.strip().split(',')])

	with open(train_target_path) as file:
		train_labels = []
		for line in file:
			train_labels.append(line.strip())

	with open(test_data_path) as file:
		test_data = []
		for line in file:
			test_data.append([int(_) for _ in line.strip().split(',')])

	train_data = np.array(train_data)
	train_target = np.array(train_target)
	test_data = np.array(test_data)

	x_train, x_val, y_train, y_val = train_test_split(train_data,train_labels,test_size=0.2)

	params = {}
	keys = ['c','kernel','gamma']
	for k in keys:
		if k in opt:
			params[k] = opt[k]

	svc = SVC()

	clf = GridSearchCV(svc,params,n_jobs=opt['n_jobs'],cv=KFold(n_splits=opt['folds']).split(x_train))
	clf.fit(x_train,y_train)

	clf.predict(x_val)
	



if __name__ == '__main__':
	main()
