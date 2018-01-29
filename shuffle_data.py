#!/usr/bin/env python3

import numpy as np

def shuffle(path,perc):
	data = []
	with open(path) as file:
		header = file.readline()
		for line in file:
			data.append(line)

	indexes = list(range(len(data)))
	np.random.shuffle(indexes)
	limit = int(perc*len(data))
	train_file = open('train.dat','w')
	test_file = open('test.dat','w')
	
	train_file.write(header)
	test_file.write(header)

	for i in indexes[:limit]:
		train_file.write(data[i])

	for i in indexes[limit:]:
		test_file.write(data[i])

def main():
	shuffle('leukemia.dat',.8)


if __name__ == '__main__':
	main()