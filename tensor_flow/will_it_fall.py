'''
    File name: will_it_fall.py
    Author: Alun Evabs
    Date created: 24/07/2013
    Date last modified: 27/07/2013
    Python Version: 3.5
'''

TRAIN_FILE_NAME = "will_it_fall_train.txt" # 27000
TEST_FILE_NAME = "will_it_fall_test.txt" # ~11000
PROTOBUF_NAME = "./wif.ckpt"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # stops tf warnings
import tensorflow as tf
import numpy as np
from libs import utils

print("Reading data...")

def readDataFile(input_file):
	#open file and read data
	axes_list = []
	angles_list = []
	with open(input_file) as f:
		for line in f:
			words = line.split()
			axes_list.append( [float(words[0]), float(words[1]), float(words[2])] )
			angles_list.append( float(words[4]) )

	axes = np.asarray(axes_list)
	angles = np.asarray(angles_list).reshape(len(angles_list),1)

	axes = (axes - np.mean(axes)) / np.std(axes)
	return axes, angles

axes, angles = readDataFile(TRAIN_FILE_NAME);
axes_t, angles_t = readDataFile(TEST_FILE_NAME);

print("Done. Creating neural net...")

#tf placeholders
n_input = 3
n_output = 1
X = tf.placeholder(tf.float32, [None, n_input], name='X')
Y = tf.placeholder(tf.float32, [None, n_output], name='Y')

n_neurons = [3, 64, 64, 64, 64, 64, 64, 1]

current_input = X # start with just coords [4096, 2]
for layer_i in range(1, len(n_neurons)): 
	current_input, W = utils.linear(
		x=current_input,
		n_output=n_neurons[layer_i],
		activation=tf.nn.relu if (layer_i+1) < len(n_neurons) else None,
		name='layer_' + str(layer_i))
Y_pred = current_input



#create linear layer
#Y_pred, W = utils.linear(
	#x=X,
	#n_output=n_output,
	#activation=tf.nn.relu, #softmax, tanh, sigmoid, relu
	#name='layer1')

#loss function
#cost = tf.reduce_mean(distance(Y_pred, Y))
#cost = tf.reduce_mean(tf.reduce_sum(tf.abs(Y - Y_pred), 1))
cost = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(Y, Y_pred), 1))

#optimizer
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

print("Done. Starting training...")

#training the network
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 500
n_epochs = 30
training_cost = 0
for epochs_i in range(n_epochs):
	#print("Epoch " + str(epochs_i))
	idxs = np.random.permutation(range(len(axes))) #randomly shake up the array
	n_batches = len(idxs) // batch_size
	for batch_i in range(n_batches):
		#print("Batch " + str(batch_i))
		idxs_i = idxs[batch_i * batch_size : (batch_i + 1) * batch_size]
		
		sess.run(
			optimizer,
			feed_dict={X:axes[idxs_i], Y: angles[idxs_i]})

		training_cost = sess.run(
			cost,
			feed_dict={X: axes, Y: angles} )

	#if epochs_i % 10 == 0:
	print(epochs_i, training_cost)
print("Done. Starting testing...")

print( sess.run(cost, feed_dict={X: axes_t, Y: angles_t} ) )

print("Done. Saving network...")


#access graph
# g = tf.get_default_graph()
# names = [op.name for op in g.get_operations()]
# print(names)

# input_name = names[0] + ':0'
# x = g.get_tensor_by_name(input_name)
# print(x)
# output = g.get_tensor_by_name(names[-1] + ':0')
# print(output)

#saver = tf.train.Saver()
#saver.save(sess, './wif')

print("Done.")