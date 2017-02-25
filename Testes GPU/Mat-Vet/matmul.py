import sys
import numpy as np
import tensorflow as tf
import time
import resource

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
if int(sys.argv[3]) != 0:
	shape1 = (int(sys.argv[3]), int(sys.argv[2]), int(sys.argv[2]))
	shape2 = (int(sys.argv[3]), 1, int(sys.argv[2]))
else: 
	shape = (int(sys.argv[2]), int(sys.argv[2]))
if device_name == "gpu":
	device_name = "/gpu:0"
else:
	device_name = "/cpu:0"



with tf.device(device_name):
	random_matrix = tf.random_uniform(shape=shape1, minval=0, maxval=1)
	random_vector = tf.random_uniform(shape=shape2, minval=0, maxval=1)
	dot_operation = tf.matmul(random_vector, random_matrix)

t1 = time.time()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
	result = session.run(dot_operation)
t2 = time.time()

# It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 5)
print("Shape:", shape1, "Device:", device_name)
print("Time taken:", ((t2 - t1) * 1000.))

print("\n" * 5)
print("{} Kb".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print("\n" * 5)