import tensorflow as tf
import numpy as np
import sys
import os
import time
from utils import *
from datetime import datetime
from random import shuffle
 

tf.reset_default_graph()

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "8000"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "35"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NEPOCH = int(os.environ.get("NEPOCH", "20"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "reddit_comments.csv")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "25000"))
LOADORNOT = os.environ.get("LOADORNOT", 'True')
EXAMPLES_SIZE = int(os.environ.get("EXAMPLES_SIZE", "500000"))


if not MODEL_OUTPUT_FILE:
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  MODEL_OUTPUT_FILE = "GRU-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

# Load data
x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)


'---------------------GRAPH Construction---------------------'

#Para tratar o problema da GRU com frases variáveis em tamanho, faremos o batch equivalente à própria frase, enquanto 1 exemplo 
#do batch será equivalente à palavra. No caso, cada palavra é uni-dimensional.

tf.reset_default_graph()

num_words = 8000;

data = tf.placeholder(tf.int32, [None, None, 1])
target = tf.placeholder(tf.int32, [None, None])


num_hidden = 50
#cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
#nesse ponto estamos definindo batch_size como um tensor de dimensão 0, pois este variará dinamicamente 
#indicando o cumprimento do batch variável

batch_size = tf.shape(data)[0]
max_size = tf.shape(data)[1]
#num_classes = tf.shape(target)[1]


init_state = cell.zero_state(batch_size, tf.float32)

embedding = tf.Variable(tf.truncated_normal([num_words, num_hidden]))

#indexamento de uma vetor de dim = num_hidden por uma das 8 mil palavras. 
x_e = tf.gather_nd(embedding, data)

output, state = tf.nn.dynamic_rnn(
   cell = cell, 
   initial_state = init_state, 
   dtype=tf.float32,
   inputs = x_e)

[(t.get_shape(), t.name) for t in tf.trainable_variables()]

[t.name for t in tf.trainable_variables()]

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


'---------------------Execution of the GRAPH---------------------'
x_train_list = x_train.tolist()
y_train_list = y_train.tolist()

#conversão dados do reddit para o formato daqui
x_train_numpy = [np.array([x_train_list[i]]).transpose() for i in range(len(x_train_list))]
y_train_numpy = [np.array([y_train_list[i]]).transpose() for i in range(len(y_train_list))]

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

batch_size = 1

no_of_batches = int(len(x_train)/batch_size)
epoch = 1
for i in range(epoch):
    t1 = time.time()
    for j in np.random.permutation(len(y_train[7:34])):
        inp, out = x_train_numpy[j:j+batch_size], y_train_list[j:j+batch_size]
        try:
        	sess.run(minimize,{data: inp, target: out})
        except: 
        	print ("Erro inesperado")
        print "Example ", str(j)
        #print "Batch ", str(j)
    t2 = time.time()
    sys.stdout.flush()
    print "Time beetween epochs: %f milliseconds" % ((t2 - t1) * 1000.)    
    print "Epoch - ",str(i)

epoch = 1
totalincorrect = 0
for i in range(epoch):
    t1 = time.time()
    for j in np.random.permutation(len(y_train[35000:])):
        inp, out = x_train_numpy[j:j+batch_size], y_train_list[j:j+batch_size]
        try:
            incorrect = sess.run(error,{data: inp, target: out})
            print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
            totalincorrect = totalincorrect + incorrect
        except: 
            print ("Erro inesperado")
        print "Example ", str(j)
        #print "Batch ", str(j)
    t2 = time.time()
    sys.stdout.flush()
    print("totalincorrect %f" % totalincorrect )
    print "Time beetween epochs: %f milliseconds" % ((t2 - t1) * 1000.)    
    print "Epoch - ",str(i)




incorrect = sess.run(error,{data: x_train_numpy[0:1], target: y_train_list[0:1]})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))






sess.run(output, {data: x_train_numpy[0:1]})

sess.run()

inp, out = x_train_numpy[0:1], y_train_list[0:1]


aux = 11320
sess.run(minimize,{data: x_train_numpy[aux:aux+1], target: y_train_list[aux:aux+1]})



inp, out = x_train_numpy[3:4], y_train_list[3:4]

[(sess.run(t), t.name) for t in tf.trainable_variables()]


for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = x_train_numpy[ptr:ptr+batch_size], y_train_list[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print "Epoch - ",str(i)


import sys
import os
import time

t1 = time.time()
ptr = 0
for j in range(100):
    inp, out = x_train_numpy[ptr:ptr+batch_size], y_train_list[ptr:ptr+batch_size]
    ptr+=1
    sess.run(minimize,{data: inp, target: out})
    print "Batch ", str(j)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
sys.stdout.flush()
