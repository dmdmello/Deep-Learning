import numpy as np
import tensorflow as tf 
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator

class GRUTensorflow:
    
    def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Initialize the network parameters

        E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        c = np.zeros(word_dim)
        
        self.E = tf.Variable(initial_value = E, name='E', dtype = tf.float32)
        self.V = tf.Variable(initial_value = V, name='V', dtype = tf.float32)
        self.c = tf.Variable(initial_value = c, name='c', dtype = tf.float32)
    
        # We store the Tensorflow graph here
        self.tf = {}
        self.__tf_build__()
    
    def __tf_build__(self):
 
        E, V, c = self.E, self.V, self.c
        
        x = tf.placeholder(dtype = tf.int32, shape = [None, None, 1], name='x')
        y = tf.placeholder(dtype = tf.int32, shape = [None, None], name = 'y')

        # This is how we calculated the hidden state in a simple RNN. No longer!
        # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))
            
        cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
        batch_size = tf.shape(x)[0]
        init_state = cell.zero_state(batch_size, tf.float32)

        # Word embedding layer 
        x_e = E[:,x_t]

        output_t, s_t1 = tf.nn.dynamic_rnn(
            cell = cell,
            initial_state = init_state, 
            dtype=tf.float32,
            inputs = x_e)

        weight = tf.Variable(tf.truncated_normal([num_hidden, num_words]))
        bias = tf.Variable(tf.constant(0.1, shape=[num_words]))

        output_flat = tf.reshape(output, [-1, num_hidden])

        logits_flat = tf.matmul(output_flat, weight) + bias
        flat_probs = tf.nn.softmax(logits_flat)

        # Calculate the losses 
        target_flat = tf.reshape(target, [-1])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_flat, labels =target_flat)

        optimizer = tf.train.AdamOptimizer()
        minimize = optimizer.minimize(losses)

        mistakes = tf.not_equal(target_flat, tf.to_int32(tf.argmax(flat_probs, 1)))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

        init_op = tf.initialize_all_variables()

        self.predict = sess.run(output_flat, {data: inp, target: out})
        # self.predict_class = sess.run(output_flat, {data: inp, target: out})
        self.ce_error = sess.run(cost, {data: inp, target: out})

        def sgd_step(inp, out):
            sess.run(minimize,{data: inp, target: out})

        self.sgd_step = sgd_step
        
    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)
        