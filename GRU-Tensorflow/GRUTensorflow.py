import numpy as np
import tensorflow as tf 
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator

class GRUTensorflow:
    
    def __init__(self, word_dim, hidden_dim=128, embedding_dim = 100, bptt_truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.bptt_truncate = bptt_truncate
        self.sess = tf.Session()
        # Initialize the network parameters
        '''
        E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        c = np.zeros(word_dim)
        
        self.E = tf.Variable(initial_value = E, name='E', dtype = tf.float32)
        self.V = tf.Variable(initial_value = V, name='V', dtype = tf.float32)
        self.c = tf.Variable(initial_value = c, name='c', dtype = tf.float32)
        '''
        # We store the Tensorflow graph here
        self.tf = {}
        self.__tf_build__()
    
    def __tf_build__(self):
 
        sess = self.sess

        x_t = tf.placeholder(dtype = tf.int32, shape = [None, None, 1], name='x_t')
        y_t = tf.placeholder(dtype = tf.int32, shape = [None, None], name = 'y_t')

        # This is how we calculated the hidden state in a simple RNN. No longer!
        # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))
            
        cell = tf.contrib.rnn.GRUCell(num_units = self.hidden_dim)
        batch_size = tf.shape(x_t)[0]
        init_state = cell.zero_state(batch_size, tf.float32)

        # Word embedding layer 

        self.E = tf.Variable(tf.constant(0.0, shape=[self.word_dim, self.embedding_dim+1]), trainable=False, name = 'embedding')
        E = self.E

        embedding_placeholder = tf.placeholder(tf.float32, [self.word_dim, self.embedding_dim+1])
        #embedding_init = E.assign(embedding_placeholder)

        #indexamento de uma vetor de dim = num_hidden por uma das 8 mil palavras. 
        x_e = tf.gather_nd(E, x_t)

        output, s_t1 = tf.nn.dynamic_rnn(
            cell = cell,
            initial_state = init_state, 
            dtype=tf.float32,
            inputs = x_e)

        W_gates = sess.graph.get_tensor_by_name('rnn/gru_cell/gates/kernel:0')
        self.W_gates = W_gates
        b_gates = sess.graph.get_tensor_by_name('rnn/gru_cell/gates/bias:0')
        self.b_gates = b_gates

        W_candidate = sess.graph.get_tensor_by_name('rnn/gru_cell/candidate/kernel:0')
        self.W_candidate = W_candidate
        b_gate = sess.graph.get_tensor_by_name('rnn/gru_cell/candidate/bias:0')
        self.b_gate = b_gate


        V = tf.Variable(tf.truncated_normal([self.hidden_dim, self.word_dim]), name = 'weights_V')
        self.V = V
        
        c = tf.Variable(tf.constant(0.1, shape=[self.word_dim]), name = 'bias_c')
        self.c = c

        output_flat = tf.reshape(output, [-1, self.hidden_dim])

        logits_flat = tf.matmul(output_flat, V) + c
        flat_probs = tf.nn.softmax(logits_flat)

        # Calculate the losses 
        target_flat = tf.reshape(y_t, [-1])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_flat, labels =target_flat)

        optimizer = tf.train.AdamOptimizer()
        minimize = optimizer.minimize(losses)

        mistakes = tf.not_equal(target_flat, tf.to_int32(tf.argmax(flat_probs, 1)))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        def predict(inp):
            inp = [np.array([inp]).transpose()]
            return sess.run(flat_probs, {x_t: inp})

        self.predict = predict

        # self.predict_class = sess.run(output_flat, {data: inp, target: out})
        def ce_error(inp, out):
            return sess.run(tf.reduce_sum(losses), {x_t: inp, y_t: out})
        
        self.ce_error = ce_error

        def sgd_step(inp, out):

            inp = [inp]
            out = [out]

            sess.run(minimize,{x_t: inp, y_t: out})

        self.sgd_step = sgd_step

        def embedding_init(embedding_path):
            sess.run(E.assign(embedding_placeholder), feed_dict={embedding_placeholder: np.load(embedding_path)}) 
        self.embedding_init = embedding_init


    def calculate_total_loss(self, X, Y):
        #eturn np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])

        return np.sum([self.ce_error(X[i:i+1], Y[i:i+1]) for i in range(len(X))])


    def assign_parameter(self, value, parameter):
       
        self.sess.run(parameter.assign(value))


    def eval_parameter(self, parameter):

        return(parameter.eval(session = self.sess))


    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X,Y)/float(num_words)
    
    def save_parameters(self, outfile):

        sess = self.sess 
        saver = tf.train.Saver()
        saver.save(sess, outfile)

    def restore_parameters(self, outfile):
        
        sess = self.sess 
        saver = tf.train.Saver()
        saver.restore(sess, outfile)
    