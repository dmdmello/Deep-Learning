import sys
import numpy as np
import tensorflow as tf
import time
import resource

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
if int(sys.argv[3]) != 0:
	shape = (int(sys.argv[3]), int(sys.argv[2]), 1)
	time_file = 'DynamicRNN-%s-shape-%s-%s' % (device_name,shape[1], shape[2])
else: 
	shape = (int(sys.argv[2]), int(sys.argv[2]))
	time_file = 'DynamicRNN-%s-shape-%s-%s' % (device_name,shape[0], shape[1])
if device_name == "gpu":
	device_name = "/gpu:0"
else:
	device_name = "/cpu:0"


with tf.device(device_name):
	random_input= tf.random_uniform(shape=shape, minval=0, maxval=7999, dtype=tf.int32)
	num_words = 8000;

	num_hidden = 150
	cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)

	batch_size = tf.shape(random_input)[0]
	init_state = cell.zero_state(batch_size, tf.float32)

	embedding = tf.Variable(tf.truncated_normal([num_words, num_hidden]))

	x_e = tf.gather_nd(embedding, random_input)

	output, state = tf.nn.dynamic_rnn(
	   cell = cell, 
	   initial_state = init_state, 
	   dtype=tf.float32,
	   inputs = x_e)


t1 = time.time()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as session:
	init_op = tf.initialize_all_variables()
	session.run(init_op)
	result = session.run(output)
t2 = time.time()
time = (t2 - t1) * 1000.



# It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 5)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", ((t2 - t1) * 1000.))
print("\n" * 5)
#rint(result)
print("\n" * 5)

print("\n" * 2)
print("Memory: {} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
#print(result)
print("\n" * 2)


print("saved data to %s.npy" % (time_file))
try:
	load = np.load(('%s.npy' % time_file))
	np.save(time_file, np.append(load, time))
except: 
	np.save(time_file, time)
