import keras
from keras.models import load_model
import numpy as np

def rshp(img):
	return np.array(img).reshape((16,8,1))

def one_hot(char):
	oh = [0]*26
	oh[ord(char)-ord('a')] = 1
	return np.array(oh)

def get_char(v):
	return chr(v+ord('a'))

def compute_accuracy(y,y_hat):
	counter = 0
	for value,pred in zip(y,y_hat):
		if np.argmax(value) == np.argmax(pred):
			counter += 1
	return float(counter)/len(y)


input_path='train-data.csv'
target_path='train-targets.csv'
test_data_path='test-data.csv'

input_data = []
with open(input_path,'r') as file:
	for line in file:
		input_data.append(rshp([int(_) for _ in line.strip().split(',')]))
input_data = np.array(input_data)
		
targets = []
with open(target_path,'r') as file:
	for line in file:
		targets.append(one_hot(line.strip()))
targets= np.array(targets)

test_data = []
with open(test_data_path,'r') as file:
	for line in file:
		test_data.append(rshp([int(_) for _ in line.strip().split(',')]))
test_data = np.array(test_data)


model = load_model('keras_model.h5')

print(compute_accuracy(model.predict(input_data),targets))