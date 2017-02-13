import tensorflow as tf
import numpy as np

tf.reset_default_graph()

x = tf.placeholder(tf.int32, shape=(2, 3))
y = tf.placeholder(tf.int32, shape=(2, 3, 3))


xT = tf.transpose(x)
yT = tf.transpose(y, [2, 1, 0])




xi = [[3, 4, 1], [2, 0, 4]]

yi = [[[2, 5, 5], 
	  [1, 1, 1],
	  [2, 7, 1]],

	  [[7, 9, 2],
	   [5, 4, 5],
	   [7, 1, 4]]]


sess = tf.Session()
print(sess.run(xT, feed_dict = {x : xi}))

print(sess.run(yT, feed_dict = {y : yi}))


yT = tf.transpose(y, [1, 1, 0])
print(sess.run(yT, feed_dict = {y : yi}))


yT = tf.transpose(y, [1, 1, 0])
print(sess.run(yT, feed_dict = {y : yi}))


with tf.Session() as sess:
    print(sess.run(output, state, feed_dict={x : rand_array}))



"Proxima tarefa: reproduzir o resultado de masked_loss do WildML, usando dynamic network, "
"e imprimir os resultados" 