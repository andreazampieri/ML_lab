import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense,Dropout
from keras import backend as K
import numpy as np

def one_hot(char):
	oh = [0]*26
	oh[ord(char)-ord('a')] = 1
	return np.array(oh)

def rshp(img):
	return np.array(img).reshape((16,8,1))

def get_char(v):
	return chr(v+ord('a'))

def compute_accuracy(y,y_hat):
	counter = 0
	for value,pred in zip(y,y_hat):
		if np.argmax(value) == np.argmax(pred):
			counter += 1
	return float(counter)/len(y)


model = Sequential([
	Conv2D(64,(4,4),padding="same",input_shape=(16,8,1)),
	Activation("relu"),
	MaxPooling2D(pool_size=(2,2),strides=(2,2)),
	Conv2D(128,(4,4),padding="same"),
	Activation("relu"),
	MaxPooling2D(pool_size=(2,2),strides=(2,2)),
	Flatten(),
	Dense(1024),
	Activation("relu"),
	Dropout(0.5),
	Dense(512),
	Activation("relu"),
	Dropout(0.5),
	Dense(26),
	Activation("softmax")])

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

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#model.fit(input_data,targets,epochs=250,batch_size=2048,validation_split=0.2)
idxs = np.array(range(len(input_data)))
validation_split = 0.15
cut = int(validation_split*len(input_data))
epochs = 70
np.random.shuffle(idxs)
train_idxs = idxs[cut:]
val_idxs = idxs[:cut]

for i in range(epochs):
	np.random.shuffle(train_idxs)
	print('Epoch: {}'.format(i))
	model.fit(input_data[train_idxs],targets[train_idxs],batch_size = 300,epochs=1,verbose=1)

	#accuracy
	pred = model.predict(input_data[val_idxs])
	val_accuracy = compute_accuracy(targets[val_idxs],pred)
	print('\tval_accuracy: {}'.format(val_accuracy))


test_data = []
with open(test_data_path,'r') as file:
	for line in file:
		test_data.append(rshp([int(_) for _ in line.strip().split(',')]))

test_data = np.array(test_data)
y_pred = model.predict(test_data)

with open('results_keras.dat','w') as file:
	for value in y_pred:
		file.write(get_char(np.argmax(value))+'\n')

model.save('keras_model.h5')
