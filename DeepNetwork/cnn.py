import tensorflow as tf
import numpy as np
from math import ceil

def one_hot(letter):
	value = [0]*26
	value[ord(letter)-ord('a')] = 1
	return value

def get_batch(data,index,batch_size):
	dim = len(data)
	
	start = index * batch_size
	finish = min((index+1)*batch_size,dim)
	return data[start:finish]

def w_var(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def b_var(shape):
	return tf.Variable(tf.constant(0.1,shape=shape))

def conv2d(x,w):
	return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def main():
	# reading input data
	# 16x8 images, 1 channel [0,1]
	
	base_path = 'ocr/'
	train_data_path = base_path + 'train-data.csv'
	train_labels_path = base_path + 'train-targets.csv'
	test_data_path = base_path + 'test-data.csv'

	print('Reading data')
	with open(train_data_path,'r') as file:
		train_data = []
		for line in file:
			train_data.append([int(_) for _ in line.strip().split(',')])

	with open(train_labels_path,'r') as file:
		train_labels = []
		for line in file:
			train_labels.append(one_hot(line.strip()))

	with open(test_data_path,'r') as file:
		test_data = []
		for line in file:
			test_data.append([int(_) for _ in line.strip().split(',')])

	train_data = np.array(train_data)
	train_labels = np.array(train_labels)
	test_data = np.array(test_data)

	idxs = list(range(len(train_data)))

	x = tf.placeholder(tf.float32,[None,16*8])
	x_rshp = tf.reshape(x,[-1,16,8,1])

	y = tf.placeholder(tf.float32,[None,26])

	# first convolution:
	# 	4x2 patches
	# 	24 filters
	
	conv1_nfeats = 24
	conv1_w = w_var([4,2,1,conv1_nfeats])
	conv1_b = b_var([conv1_nfeats])

	conv1 = tf.nn.relu(conv2d(x_rshp,conv1_w)+conv1_b)
	pool1 = max_pool_2x2(conv1)

	# after the first pooling, the input has shape [batch_size, 8,4, conv1_nfeats]
	# [,8,4,24]
	conv2_nfeats = 48
	conv2_w = w_var([4,2,conv1_nfeats,conv2_nfeats])
	conv2_b = b_var([conv2_nfeats])

	conv2 = tf.nn.relu(conv2d(pool1,conv2_w)+conv2_b)
	pool2 = max_pool_2x2(conv2)
	# pool2 has shape [,4,2,48]; 4*2*48 = 384
	# 2 fully connected layers of size (resp) 100 and 26
	keep_prob = tf.placeholder(tf.float32)

	fcl1_w = w_var([4*2*48,100])
	fcl1_b = b_var([100])

	pool2_flat = tf.reshape(pool2,[-1,4*2*48])
	fcl1 = tf.nn.relu(tf.matmul(pool2_flat,fcl1_w)+fcl1_b)
	dropout_fcl1 = tf.nn.dropout(fcl1,keep_prob)

	fcl2_w = w_var([100,26])
	fcl2_b = b_var([26])

	output = tf.matmul(dropout_fcl1,fcl2_w) + fcl2_b
	y_hat = tf.nn.softmax(output)

	#loss function and optimizer
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_hat),reduction_indices=[1]))
	train_opt = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	correctness = tf.equal(tf.argmax(y_hat,1),tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correctness,tf.float32))

	# params for the execution
	n_epochs = 20
	batch_size = 200
	batch_number = int(ceil(len(train_data)/batch_size))
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	for i in range(n_epochs):
		print('Epoch: {}'.format(i))
		np.random.shuffle(idxs)
		for i in range(batch_number):
			curr_idx = get_batch(idxs,i,batch_size)
			sess.run(train_opt,feed_dict={x:train_data[curr_idx],y:train_labels[curr_idx],keep_prob:0.5})

	# accuracy on training set
	print(sess.run(accuracy,feed_dict={x: train_data, y:train_labels, keep_prob: 1.0}))

if __name__ == '__main__':
	main()