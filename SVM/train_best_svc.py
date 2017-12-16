import argparse
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn import metrics
from sklearn.svm import SVC

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('config_file')
	args = parser.parse_args()

	with open(args.config_file,'r') as file:
		opt = {}
		for line in file:
			k,v = line.strip().split(':')
			opt[k] = v

	C = int(opt['C'])
	kernel = opt['kernel']
	gamma = float(opt['gamma'])

	train_data_path = opt['train_data_path']
	train_labels_path = opt['train_labels_path']
	test_data_path = opt['test_data_path']
	results_path = opt['test_labels_path']
	bestparams_path = opt['bestparams']

	svc = SVC(C=C, kernel=kernel, gamma=gamma)



if __name__ == '__main__':
	main()