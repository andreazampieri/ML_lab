import numpy as np
import pandas as pd
import tensorflow as tf
from math import ceil
from sklearn.model_selection import train_test_split, KFold


def LearnBatch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx]
    labels_shuffle = labels[idx]
    labels_shuffle = np.reshape(labels_shuffle, (num, len(labels[0])))
    return data_shuffle, labels_shuffle

def PredictBatch(data,indx,win):
	till = len(data)
	From = indx * win
	To = min((indx+1)*win,till)
	return data[From:To]

X = np.genfromtxt('train-data.csv', delimiter=',')
P = np.genfromtxt('test-data.csv', delimiter=',')
T = np.genfromtxt('train-target.csv', dtype=None, delimiter=',')
Y = pd.get_dummies(T).values

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=False)
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


y = tf.placeholder(tf.float32, [None, 26])
x = tf.placeholder(tf.float32, [None, 128])
x_image = tf.reshape(x, [-1, 16, 8, 1]) 

W_conv1 = weight_variable([4,4,1,64])# [filter_height, filter_width, in_channels, out_channels]
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) #shape=(?, 8, 4, 64)

W_conv2 = weight_variable([4, 4, 64, 128])
b_conv2 = bias_variable([128]) 
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) #shape=(?, 4, 2, 128)

W_fc1 = weight_variable([4 * 2 * 128, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 2 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 26])
b_fc2 = bias_variable([26])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_hat = tf.nn.softmax(y_conv)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predict = tf.argmax(y_hat,1)

sess.run(tf.global_variables_initializer())
kf=KFold(n_splits=7 , shuffle=True, random_state=42)
for tr_i, ts_i in kf.split(X):
        X_train, X_test = X[tr_i], X[ts_i]
        Y_train, Y_test = Y[tr_i], Y[ts_i]
accuracy_values = []
for i in range(3000):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    batch_xs, batch_ys = LearnBatch(300, X_train, Y_train)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        print('step {}, Training accuracy {}'.format(i, train_accuracy))
        batch_xs, batch_ys = LearnBatch(2000, X_test, Y_test)
        accuracy_values.append(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0}))
        print('step {}, Test accuracy {}'.format(i,np.mean(accuracy_values)))
    train_step.run(feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
    
 

results = []
batch_number = int(ceil(len(X)/400))
id = list(range(len(P)))
for i in range(batch_number): 
    curr_id = PredictBatch(id,i,400)
    results.append(sess.run(predict,feed_dict={x:P[curr_id],keep_prob:0.5}))
out=open('result.txt','w')
for items in results:
			for element in items:
				out.write(chr(element+ord('a'))+'\n')      