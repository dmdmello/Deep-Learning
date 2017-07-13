import tensorflow as tf
import numpy as np
import sys
import os
import time
from utils import *
from datetime import datetime
from random import shuffle
path_TFRecord = '../TFRec/TFRecordfile15k'
path_TFRecord = '../TFRec/TFRecordfile500k'
path_TFRecord_test = '../TFRec/TFRecordfile500k_test'
tf.reset_default_graph()

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "8000"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "100"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "140"))
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

x_train_list = x_train.tolist()
y_train_list = y_train.tolist()

#ordenar frases por quantidade de palavras
#x_train_list.sort(key=len)
#y_train_list.sort(key=len)

#conversão dados do reddit para o formato numpy
x_train_numpy = [np.array([x_train_list[i]]).transpose() for i in range(len(x_train_list))]
y_train_numpy = [np.array([y_train_list[i]]).transpose() for i in range(len(y_train_list))]


#data = tf.placeholder(tf.int32, [None, None, 1])
#target = tf.placeholder(tf.int32, [None, None])


#---------------TFRecord-Format-----------------------
sequences = [[1, 2, 3], [4, 5, 1], [1, 2]]
label_sequences = [[0, 1, 0], [1, 0, 0], [1, 1]]

#sequences = [[[1], [2], [3]], [[4], [5], [1]], [[1], [2]]]
#label_sequences = [[0, 1, 0], [1, 0, 0], [1, 1]]

sequences = y_train_numpy[0:15001]
label_sequences = y_train_list[0:15001]

sequences_test = y_train_numpy[200000:225000]
label_sequences_test = y_train_list[200000:225000]

def make_example(sequence, labels):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(sequence)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list["tokens"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    for token, label in zip(sequence, labels):
        fl_tokens.feature.add().int64_list.value.append(token)
        fl_labels.feature.add().int64_list.value.append(label)
    return ex

# Write all examples into a TFRecords file

#------------------------------TRAINING SET------------------
writer = tf.python_io.TFRecordWriter(path_TFRecord)
for sequence, label_sequence in zip(sequences, label_sequences):
    ex = make_example(sequence, label_sequence)
    writer.write(ex.SerializeToString())
writer.close()


#--------------------------------TEST SET------------------
writer_test = tf.python_io.TFRecordWriter(path_TFRecord_test)
for sequence, label_sequence in zip(sequences_test, label_sequences_test):
    ex = make_example(sequence, label_sequence)
    writer_test.write(ex.SerializeToString())
writer.close()


filename_queue = tf.train.string_input_producer([path_TFRecord]) 
reader = tf.TFRecordReader()
key, value = reader.read(filename_queue)

context_features = {
    "length": tf.FixedLenFeature([], dtype=tf.int64)
}
sequence_features = {
    "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
}

context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=value,
        context_features=context_features,
        sequence_features=sequence_features
    )
'''
data_tokens = tf.to_int32(sequence_parsed['tokens'])
data_tokens = tf.transpose([data_tokens])

data_labels = tf.to_int32(sequence_parsed['labels'])

data = (data_tokens, data_labels)
'''

data_tokens = tf.to_int32(sequence_parsed['tokens'])
data = tf.transpose([data_tokens])

#-----------------Graph-----------------------------

input_queue = tf.RandomShuffleQueue(
	capacity = 100000,
	min_after_dequeue = 10000,
	dtypes=[tf.int32])

input_enqueue_op = input_queue.enqueue(data)

qr_input = tf.train.QueueRunner(input_queue, [input_enqueue_op] * 40)
tf.train.add_queue_runner(qr_input)

non_paddled_input = input_queue.dequeue()


batch_size = 15
padding_queue_cap = 1000

padding_queue = tf.PaddingFIFOQueue(
    capacity=padding_queue_cap,
    dtypes=[tf.int32],
    shapes=[[None, 1]])

padding_enqueue_op = padding_queue.enqueue(non_paddled_input)

qr_padding = tf.train.QueueRunner(padding_queue, [padding_enqueue_op] * 20)
tf.train.add_queue_runner(qr_padding)

inputs = padding_queue.dequeue_many(batch_size)
#x_t = tf.slice(inputs, [0,1,0], [batch_size, tf.shape(inputs)[1]-1, 1])
#y_t = tf.slice(inputs, [0,0,0], [batch_size, tf.shape(inputs)[1]-1, 1])

num_words = 8000
num_hidden = 128
#embedding = tf.Variable(tf.truncated_normal([num_words, num_hidden]))

#embedding_matrix = np.load('embedding_matrix_WIKI_100D.npy')
#embedding_matrix.astype(np.float32)


embedding = tf.Variable(tf.constant(0.0, shape=[num_words, EMBEDDING_DIM+1]),
trainable=False, name="embedding")

embedding_placeholder = tf.placeholder(tf.float32, [num_words, EMBEDDING_DIM+1])
embedding_init = embedding.assign(embedding_placeholder)

x_e = tf.gather_nd(embedding, inputs)


#-------RNN DEFINITION--------#

#cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple = True)
cell = tf.contrib.rnn.GRUCell(num_hidden)

init_state = cell.zero_state(batch_size, tf.float32)

output, state = tf.nn.dynamic_rnn(
   cell = cell, 
   initial_state = init_state, 
   dtype=tf.float32,
   inputs = x_e)


weight = tf.Variable(tf.truncated_normal([num_hidden, num_words]))
bias = tf.Variable(tf.constant(0.1, shape=[num_words]))

output = tf.slice(output, [0,0,0], [batch_size, tf.shape(output)[1]-1, num_hidden])
y_t = tf.slice(inputs, [0,1,0], [batch_size, tf.shape(inputs)[1]-1, 1])

output_flat = tf.reshape(output, [-1, num_hidden])

logits_flat = tf.matmul(output_flat, weight) + bias
flat_probs = tf.nn.softmax(logits_flat)


# Calculate the losses 
y_t_flat = tf.reshape(y_t, [-1])
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_flat, labels =y_t_flat)

#Loss Masking
mask = tf.sign(tf.to_float(y_t_flat))
masked_losses = mask*losses

# Bring back to [B, T] shape
masked_losses = tf.reshape(masked_losses,  tf.shape(y_t))
mean_masked_losses = tf.divide(tf.reduce_sum(masked_losses), tf.reduce_sum(mask))

optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(masked_losses)

# Calculates mean accuracy of classification for an entire batch of examples
mistakes_real = tf.not_equal(y_t_flat, tf.to_int32(mask*tf.to_float(tf.argmax(flat_probs, 1))))
error_real = tf.reduce_mean(tf.cast(mistakes_real, tf.float32))


#-----------------Executuion-----------------------------

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
sess.run(embedding_init, feed_dict={embedding_placeholder: np.load('embedding_matrix_WIKI_100D.npy')})



sess.run(input_queue.size())


sess.run((x_e, inputs))
sess.run((tf.shape(x_e), tf.shape(inputs)))

sess.run((tf.shape(x_e), tf.shape(output)))

sess.run((tf.shape(output_flat), tf.shape(output)))

sess.run((tf.shape(flat_probs), flat_probs))

sess.run((tf.shape(output), tf.shape(losses), tf.shape(masked_losses), losses, masked_losses))


def error_k(sess, k):
    losses_ac = 0.0
    mean_error = 0.0
    num_int = k
    for i in range(num_int):
    	t1 = time.time()
        try:
            mean_error = mean_error + sess.run(error_real)
            losses_ac = losses_ac + sess.run(mean_masked_losses) 
        except: 
            print ("Erro inesperado")      
        #print "Batch ", str(j)
        t2 = time.time()
        #print "Epoch - ",str(i)
        #print "Time beetween epochs: %f milliseconds" % ((t2 - t1) * 1000.)

    print"---------------------RESULTS----------------------"
    print"Mean accuracy for %d iterations:" % (k)
    print(1-(mean_error/num_int))
    print"Mean losses for %d iterations: " % (k) 
    print(losses_ac/num_int)
	

epoch = 2
for i in range(epoch):
    t1 = time.time()
    for j in range(0, 5000):
        try:
        	sess.run(minimize)
        except: 
        	print ("Erro inesperado")      
        #print "Batch ", str(j)
    t2 = time.time()
    sys.stdout.flush()
    #print("compute_gradients")
    #print(sess.run(compute_gradients))
    
    #print("output and losses")
    #print(sess.run(output))
    #print(sess.run(losses))


    print "Time beetween epochs: %f milliseconds" % ((t2 - t1) * 1000.)    
    print "Epoch - ",str(i)
    
    error_k(sess, 150 )
    saver.save(sess, "tmp/GRU-hd%d-b%d-200k" % (num_hidden, batch_size ))


t1 = time.time()
sess.run(minimize)
t2 = time.time()
print "Time beetween epochs: %f milliseconds" % ((t2 - t1) * 1000.)   

sess.run((tf.shape(flat_probs), flat_probs, tf.shape(y_t_flat), y_t_flat))
sess.run((tf.shape(tf.argmax(flat_probs, 1)), tf.argmax(flat_probs, 1), tf.shape(y_t_flat), y_t_flat))


sess.run((tf.shape(tf.argmax(flat_probs, 1)), tf.to_int32(mask*tf.to_float(tf.argmax(flat_probs, 1))), tf.shape(y_t_flat), y_t_flat))





#---------------------------Print Sentence-----------------------------------


new_sentence = [[[7998], [7994], [7941], [7878], [7985], [7996], [7447], [5767]]]

new_sentence = [[[7998], [7929], [7994], [0]]]

new_sentence = [[[7998], [0]]]

def generate_sent(new_sentence):
    while (not new_sentence[0][-1][0] == 7999) and (len(new_sentence[0]) < 40):
        samples = np.random.multinomial(1, sess.run(flat_probs, feed_dict={inputs: new_sentence*batch_size})[0]*0.95)
        sampled_word = np.argmax(samples)
        new_sentence[0][-1][0] = sampled_word
        new_sentence[0].append([0])
       # print_sentence(np.transpose(new_sentence[0])[0], index_to_word)  
    return new_sentence

sess.run(flat_probs, feed_dict={inputs: new_sentence*30})

def print_sentence(s, index_to_word):
    sentence_str = [index_to_word[x] for x in s[1:-1]]
    try:
        print(" ".join(sentence_str))
    except UnicodeEncodeError:new_sentence
        print "UnicodeEncodeError!"
    except:
        print "Unhandled Exception!"
    sys.stdout.flush()

new_sentence = generate_sent(new_sentence)
print_sentence(np.transpose(new_sentence[0])[0], index_to_word)  




losses_ac = 0.0
mean_error = 0.0
t1 = time.time()
losses_ac = losses_ac + (sess.run(tf.reduce_sum(losses)) / sess.run(tf.reduce_sum(mask)))
t2 = time.time()
print "Time beetween epochs: %f milliseconds" % ((t2 - t1) * 1000.)  


sess.run(data, {data:x_train_numpy[4:4+1]})
threads = tf.train.start_queue_runners(sess = sess)


epoch = 1
for i in range(epoch):
    t1 = time.time()
    for j in range(0, 100):
        inp, out = x_train_numpy[j:j+1], y_train_list[j:j+1]
        try:
        	sess.run(data, {data: inp})
        except: 
        	print ("Erro inesperado")
        print "Example ", str(j)
        #print "Batch ", str(j)
    t2 = time.time()
    sys.stdout.flush()
    print "Time beetween epochs: %f milliseconds" % ((t2 - t1) * 1000.)    
    print "Epoch - ",str(i)


t1 = time.time()
sess.run(padding_queue.dequeue_many(50))
t2 = time.time()
sess.run(padding_queue.dequeue_many(50))
t3 = time.time()


const = tf.constant(45) 

output = tf.add(inputs, const)




many_inputs= queue.enqueue_many([[4],[6], [10], [8]])


qr = tf.train.QueueRunner(queue, [enqueue_op])

sess = tf.Session()


enqueue_threads = qr.create_threads(sess, start=True)

sess.run(enqueue_op, {data: [5]})

sess.run(output)