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

	opt['float'] = opt['float'].split(',')
	opt['int'] = opt['int'].split(',')

	for k in opt['float']:
		opt[k] = [float(_) for _ in opt[k].split(',')]
		if len(opt[k]) == 1:
			opt[k] = opt[k][0]

	for k in opt['int']:
		opt[k] = [int(_) for _ in opt[k].split(',')]
		if len(opt[k]) == 1:
			opt[k] = opt[k][0]

	opt['params'] = opt['params'].split(',')
	opt['kernel'] = opt['kernel'].split(',')

	base_path = opt['base_path']
	train_data_path = base_path + opt['train_data_path']
	train_labels_path = base_path + opt['train_labels_path']
	test_data_path = base_path + opt['test_data_path']
	results_path = base_path + opt['results']
	bestparams_path = base_path + opt['bestparams']

	with open(train_data_path) as file:
		train_data = []
		for line in file:
			train_data.append([int(_) for _ in line.strip().split(',')])

	with open(train_labels_path) as file:
		train_labels = []
		for line in file:
			train_labels.append(line.strip())

	with open(test_data_path) as file:
		test_data = []
		for line in file:
			test_data.append([int(_) for _ in line.strip().split(',')])

	train_data = np.array(train_data)
	train_labels = np.array(train_labels)
	test_data = np.array(test_data)

	params = {}
	keys = opt['params']
	for k in keys:
		if k in opt:
			params[k] = opt[k]
	

	svc = SVC()
	x_train, x_val, y_train, y_val = train_test_split(train_data,train_labels,test_size=0.2)

	clf = GridSearchCV(svc,params,n_jobs=opt['n_jobs'],cv=KFold(n_splits=opt['folds']).split(x_train))
	clf.fit(x_train,y_train)

	pred = clf.predict(x_val)
	acc = metrics.accuracy_score(y_val,pred)

	with open(results_path,'w') as file:
		file.write('Accuracy: '+str(acc)+'\n')
		file.write('Params: '+str(clf.best_params_)+'\n')
	
	with open(bestparams_path,'w') as file:
		for k,v in clf.best_params_.items():
			file.write(str(k)+':'+str(v)+'\n')



if __name__ == '__main__':
	main()
