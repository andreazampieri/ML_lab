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
	test_labels_path = opt['test_labels_path']

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

	svc = SVC(C=C, kernel=kernel, gamma=gamma)
	svc.fit(train_data,train_labels)

	test_labels = svc.predict(test_data)

	with open(test_labels_path,'r') as file:
		for v in test_labels:
			file.write(str(v)+'\n')



if __name__ == '__main__':
	main()