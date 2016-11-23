import numpy as np
from utils import *
import operator
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class RNNTensorflow:

	def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):

	# Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        #Tensorflow: Create variables
        self.U = tf.Variable(tf.float32, U.shape)
        self.V = tf.Variable(tf.float32, V.shape)
        self.W = tf.Variable(tf.float32, W.shape)

        def __tensorflow_build__(self):
        	U, V, W = self.U, self.V, self.W
        	x = tf.placeceholder(tf.float32, [None, word_dim])
                y = 

        tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
