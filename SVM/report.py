#!/usr/bin/env python3
import argparse
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, learning_curve
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from collections import defaultdict
from pprint import pprint
from time import time

def arr_to_str(arr):
	s = ""
	for v in arr:
		s += v +','

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
			train_data.append([float(_) for _ in line.strip().split(',')])

	with open(train_labels_path) as file:
		train_labels = []
		for line in file:
			train_labels.append(line.strip())

	with open(test_data_path) as file:
		test_data = []
		for line in file:
			test_data.append([float(_) for _ in line.strip().split(',')])

	train_data = np.array(train_data)
	train_labels = np.array(train_labels)
	test_data = np.array(test_data)

	params = {}
	keys = opt['params']
	for k in keys:
		if k in opt:
			try:
				if len(opt[k]) > 0:
					params[k] = opt[k]
			except TypeError:
				params[k] = [opt[k]]
	kf = KFold(n_splits=4,shuffle=True, random_state=10)
	gamma_values = opt['gamma']
	c_values = opt['C']
	dd = lambda:defaultdict(dd)
	#scores = dd()
	logfile = open('log_scores','w')
	scores = []
	score_param=['accuracy','precision_weighted','recall_weighted','f1_weighted']
	for c in c_values:
		for gamma in gamma_values:
			for s in score_param:
				ts = time()
				classifier = SVC(C=c,gamma=gamma,kernel='rbf')
				curr_score=cross_val_score(classifier,train_data,train_labels,cv=kf.split(train_data),n_jobs=-1,scoring=s)
				for v in curr_score:
					logfile.write(str(v))
					logfile.write(', ')
				logfile.write('\n')
				print(time()-ts)

	# acc = [s['acc'] for s in scores]
	# prec = [s['prec'] for s in scores]
	# rec = [s['rec'] for s in scores]
	# f1 = [s['f1'] for s in scores]


	index_of_best = np.argmax(acc)
	best_c = c_values[int(index_of_best / len(c_values))]
	best_gamma = gamma_values[index_of_best % len(gamma_values)]

	best_svc = SVC(C=best_c,gamma=best_gamma,kernel='rbf')

	train_sizes, train_scores, test_scores = learning_curve(best_svc, train_data, train_labels, scoring='accuracy')


	plt.figure()
	plt.title("Learning curve")
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	plt.grid()
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	with open("plot",'w') as file:
		file.write(arr_to_str(train_sizes))
		file.write('\n')
		file.write(arr_to_str(train_scores))

		file.write('\n')
		file.write(arr_to_str(test_scores))
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="learning_curve")

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
	                 train_scores_mean + train_scores_std, alpha=0.1, color="r")


	plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
	                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

	plt.legend()
	plt.show()
	quit()
	###############
	x_train, x_val, y_train, y_val = train_test_split(train_data,train_labels,test_size=0.2)

	clf = GridSearchCV(svc,params,n_jobs=opt['n_jobs'],cv=KFold(n_splits=opt['folds']).split(x_train))
	clf.fit(x_train,y_train)

	pred = clf.predict(x_val)
	acc = metrics.accuracy_score(y_val,pred)
	f1 = metrics.f1_score(y_val,pred)
	rec = metrics.precision(y_val,pred)

	with open(results_path,'w') as file:
		file.write('Params: '+str(clf.best_params_)+'\n')
		file.write('Accuracy: '+str(acc)+'\n')
	
	with open(bestparams_path,'w') as file:
		for k,v in clf.best_params_.items():
			file.write(str(k)+':'+str(v)+'\n')

		file.write('train_data_path:'+train_data_path+'\n')
		file.write('train_labels_path:'+train_data_path+'\n')
		file.write('test_data_path:'+train_data_path+'\n')
		file.write('test_labels_path:'+train_data_path+'\n')


def inline():

	base_path = 'sklearn-lab-material/ocr/'
	train_data_path = base_path + 'train-data.csv'
	train_labels_path = base_path + 'train-targets.csv'
	test_data_path = base_path + 'test-data.csv'
	results_path = base_path + 'test-targets.csv'

	with open(train_data_path) as file:
		train_data = []
		for line in file:
			train_data.append([float(_) for _ in line.strip().split(',')])

	with open(train_labels_path) as file:
		train_labels = []
		for line in file:
			train_labels.append(line.strip())

	with open(test_data_path) as file:
		test_data = []
		for line in file:
			test_data.append([float(_) for _ in line.strip().split(',')])

	train_data = np.array(train_data)
	train_labels = np.array(train_labels)
	test_data = np.array(test_data)
	best_c = 100
	best_gamma = 0.1 
	best_svc = SVC(C=best_c,gamma=best_gamma,kernel='rbf')

	train_sizes, train_scores, test_scores = learning_curve(best_svc, train_data, train_labels, scoring='accuracy')


	plt.figure()
	plt.title("Learning curve")
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	plt.grid()
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	with open("plot",'w') as file:
		file.write(train_sizes)
		file.write('\n')
		file.write(train_scores)

		file.write('\n')
		file.write(test_scores)
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="learning_curve")

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
	                 train_scores_mean + train_scores_std, alpha=0.1, color="r")


	plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
	                 test_scores_mean + test_scores_std, alpha=0.1, color="g")

	plt.legend()
	plt.show()
	quit()

def stats():
	c_values=[1,10,100]
	gamma_values=[0.1,0.2,0.3]
	score_param=['accuracy','precision_weighted','recall_weighted','f1_weighted']
	scores={'accuracy':[],'precision_weighted':[],'recall_weighted':[],'f1_weighted':[]}
	with open('log_scores','r') as file:
		for c in c_values:
			for gamma in gamma_values:
				for s in score_param:
					scores[s].append([float(_) for _ in file.readline().strip().split(',')[:-1]])
	acc =[]
	for a in scores['accuracy']:
		acc.append(np.mean(a))

	prec = []
	for p in scores['precision_weighted']:
		prec.append(np.mean(p))

	rec = []
	for r in scores['recall_weighted']:
		rec.append(np.mean(r))

	f1 = []
	for f in scores['f1_weighted']:
		f1.append(np.mean(f))

	with open('stats','w') as file:
		for i in range(len(c_values)):
			for j in range(len(gamma_values)):
				idx = i*len(gamma_values)+j
				file.write(c_values[i]+','+gamma_values[j]+','+acc[idx]+','+prec[idx]+','+rec[idx]+','+f1[idx])

					
		

if __name__ == '__main__':
	#main()
	#inline()
	stats()
