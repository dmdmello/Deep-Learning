import tensorflow as tf
import numpy as np
import sys
import os
import time
from load_text import *
from datetime import datetime
from random import shuffle
from collections import deque
path_TFRecord_train = 'TFRec2/TFRecordfile200k'
path_TFRecord_test = 'TFRec2/TFRecordfile200k_test'
tf.reset_default_graph()

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "8000"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "100"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "140"))
NEPOCH = int(os.environ.get("NEPOCH", "20"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "reddit_comments500.csv")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "25000"))
LOADORNOT = os.environ.get("LOADORNOT", 'True')
EXAMPLES_SIZE = int(os.environ.get("EXAMPLES_SIZE", "500000"))

# Load data to numpy format (optional)
#x_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)


#------------------------------TRAINING SET READER-----------------------------
filename_queue_train = tf.train.string_input_producer([path_TFRecord_train]) 
reader_train = tf.TFRecordReader()
key_train, value_train = reader_train.read(filename_queue_train)

#------------------------------TEST SET READER---------------------------------
filename_queue_test = tf.train.string_input_producer([path_TFRecord_test]) 
reader_test = tf.TFRecordReader()
key_test, value_test = reader_test.read(filename_queue_test)


sequence_features_train = {
    "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64)
}

sequence_features_test = {
    "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64)
}


context_parsed_train, sequence_parsed_train = tf.parse_single_sequence_example(
        serialized=value_train,
        sequence_features=sequence_features_train
    )

context_parsed_test, sequence_parsed_test = tf.parse_single_sequence_example(
        serialized=value_test,
        sequence_features=sequence_features_test
    )


data_tokens_train = tf.to_int32(sequence_parsed_train['tokens'])
data_train = tf.transpose([data_tokens_train])

data_tokens_test = tf.to_int32(sequence_parsed_test['tokens'])
data_test = tf.transpose([data_tokens_test])


#---------------------------------------------------------------------
#----------------------GRAPH CONSTRUCTION-----------------------------
#---------------------------------------------------------------------


class RandomPaddingFIFOQueue():

    def __init__(self, data, batch_size, padding_queue_cap = 1000, random_queue_cap = 400000, dtypes = [tf.int32], shapes = [[None, 1]], threads_q1 = 100, threads_q2 = 10):

        input_queue = tf.RandomShuffleQueue(
        capacity = random_queue_cap,
        min_after_dequeue = int(random_queue_cap * 0.5),
        dtypes=dtypes)

        input_enqueue_op = input_queue.enqueue(data)
        qr_input = tf.train.QueueRunner(input_queue, [input_enqueue_op] * threads_q1)
        tf.train.add_queue_runner(qr_input)
        non_paddled_input = input_queue.dequeue()
        self.non_paddled_input = non_paddled_input

        padding_queue = tf.PaddingFIFOQueue(
            capacity=padding_queue_cap,
            dtypes=dtypes,
            shapes=shapes)

        padding_enqueue_op = padding_queue.enqueue(non_paddled_input)
        qr_padding = tf.train.QueueRunner(padding_queue, [padding_enqueue_op] * threads_q2)
        tf.train.add_queue_runner(qr_padding)
        self.dequeue_batch = padding_queue.dequeue_many(batch_size)

batch_size =  8
padding_queue_cap = 1000

train_q = RandomPaddingFIFOQueue(data_train, batch_size, padding_queue_cap)
test_q = RandomPaddingFIFOQueue(data_test, batch_size, padding_queue_cap)

inputs = train_q.dequeue_batch
#------------------------------Embedding-----------------------------------

num_words = 8000
num_hidden1 = 132
num_hidden2 = 132

#embedding = tf.Variable(tf.truncated_normal([num_words, EMBEDDING_DIM]), trainable=False)

embedding = tf.Variable(tf.constant(0.0, shape=[num_words, EMBEDDING_DIM]),
trainable=False, name="embedding")
embedding_placeholder = tf.placeholder(tf.float32, [num_words, EMBEDDING_DIM])
embedding_init = embedding.assign(embedding_placeholder)

x_e = tf.gather_nd(embedding, inputs)

#----------------------------RNN Definition---------------------------------

#cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple = True)
cell1 = tf.contrib.rnn.GRUCell(num_hidden1)
cell2 = tf.contrib.rnn.GRUCell(num_hidden2)

multi_cell =  tf.contrib.rnn.MultiRNNCell([cell1, cell2])
init_state = multi_cell.zero_state(batch_size, tf.float32)

output, state = tf.nn.dynamic_rnn(
   cell = multi_cell, 
   initial_state = init_state, 
   dtype=tf.float32,
   inputs = x_e)

weight = tf.Variable(tf.truncated_normal([num_hidden2, num_words]))
bias = tf.Variable(tf.constant(0.1, shape=[num_words]))

output = tf.slice(output, [0,0,0], [batch_size, tf.shape(output)[1]-1, num_hidden2])
y_t = tf.slice(inputs, [0,1,0], [batch_size, tf.shape(inputs)[1]-1, 1])

output_flat = tf.reshape(output, [-1, num_hidden2])

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

optimizer = tf.train.AdamOptimizer(0.0006)

grads = optimizer.compute_gradients(masked_losses)
capped_grads = [(tf.clip_by_value(grad, -1e16, 1e16), var) for grad, var in grads]

grad_placeholder = [(tf.placeholder(tf.float32, shape =grad[0].get_shape()), grad[1]) for grad in grads]
capped_grads_placeholder = [(tf.clip_by_value(grad, -1e16, 1e16), var) for grad, var in grad_placeholder]

apply_grads = optimizer.apply_gradients(capped_grads)
minimize = optimizer.minimize(masked_losses)

# Calculates mean accuracy of classification for an entire batch of examples
mistakes_real = tf.not_equal(y_t_flat, tf.to_int32(mask*tf.to_float(tf.argmax(flat_probs, 1))))
error_real = tf.divide(tf.reduce_sum(tf.cast(mistakes_real, tf.float32)), tf.reduce_sum(mask))

#-----------------Executuion-----------------------------

sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
sess.run(embedding_init, feed_dict={embedding_placeholder: np.load('embedding_matrix_gensim_100D.npy')})
epoch_counter = 0

'''
sess.run(input_queue_train.size())


sess.run((x_e, inputs))
sess.run((tf.shape(x_e), tf.shape(inputs)))

sess.run((tf.shape(x_e), tf.shape(output)))

sess.run((tf.shape(output_flat), tf.shape(output)))

sess.run((tf.shape(flat_probs), flat_probs))

sess.run((tf.shape(output), tf.shape(losses), tf.shape(masked_losses), losses, masked_losses))
'''


def performance_k(sess, k, feed_inp = False, inp = None):
    losses_ac = 0.0
    classification_error_ac = 0.0
    num_int = k
    for i in range(num_int):
        t1 = time.time()
        try:
            if( not feed_inp):
                (classification_error, losses) = sess.run((error_real, mean_masked_losses))
            else:
                (classification_error, losses) = sess.run((error_real, mean_masked_losses), feed_dict={inputs: sess.run(inp)})
            
            classification_error_ac = classification_error_ac + classification_error
            losses_ac = losses_ac + losses    

        except KeyboardInterrupt : 
            print ("KeyboardInterrupt")
            break 

        except: 
            print ("Erro inesperado")

        #print "Batch ", str(j)
        t2 = time.time()
        #print "Epoch - ",str(i)
        #print "Time beetween epochs: %f milliseconds" % ((t2 - t1) * 1000.)

    return(1-(classification_error_ac/num_int), losses_ac/num_int)

#last_grads = deque([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#last_v = deque([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


performance_test_hist =[]
performance_train_hist = []
train_set_size = 200000
test_set_size = 60000
num_iterations_train = train_set_size/batch_size
num_iterations_test = test_set_size/batch_size

flag_break = False
epoch = 1
for i in range(epoch):
    if flag_break:
        break

    print "--------------------------------------------------"
    print "Epoch - ",str(i)
    print "--------------------------------------------------"

    t1 = time.time()
    for j in range(0, int(num_iterations_train)):
        
        if (j % int(num_iterations_train/2) == 0):
            
            print(j)
            '''(acc_train, loss_train) = performance_k(sess, int(0.1*num_iterations_train))
            (acc_test, loss_test) = performance_k(sess, int(0.4*num_iterations_test), True, test_q.dequeue_batch)

            performance_train_hist.append((acc_train, loss_train))
            performance_test_hist.append((acc_test, loss_test))

            print"---------------------RESULTS----------------------"
            print"Train accuracy and losses for %d iterations:" % (int(0.1*num_iterations_train))
            print(acc_train, loss_train)
            print"Test accuracy and losses for %d iterations:" % (int(0.4*num_iterations_test))
            print(acc_test, loss_test)
            '''
        try:
            '''
            #grad_vals = sess.run([grad[0] for grad in grads])
            grad_vals = sess.run(grads)            
            last_grads.popleft()
            last_grads.append(grad_vals)


            #print(grad_vals)
            feed_dict={}
            for i in xrange(len(grad_placeholder)):
                feed_dict[grad_placeholder[i][0]] = grad_vals[i][0]
            sess.run(apply_grads, feed_dict=feed_dict)
            

            v_values = [sess.run(optimizer.get_slot(var, 'v')) for var in tf.trainable_variables()]
            last_v.popleft()
            last_v.append(v_values)
            '''
            sess.run(apply_grads)


            if np.isnan(sess.run(mean_masked_losses)):
                flag_break = True
                break   

        except KeyboardInterrupt : 
            flag_break = True
            print ("KeyboardInterrupt")
            break

        except:
            print ("Exception")

        #print "Batch ", str(j)
    t2 = time.time()
    print "Time beetween epochs: %f milliseconds" % ((t2 - t1) * 1000.)    
    sys.stdout.flush()
    epoch_counter = epoch_counter + 1

    np.save(performance_train_hist, performance_test_hist, "perf_rec_GRU2ly-1hd%d-2hd%d-b%d-200k" % (num_hidden1, num_hidden2, batch_size))
    saver.save(sess, "tmp/GRU2ly-1hd%d-2hd%d-b%d-200k" % (num_hidden1, num_hidden2, batch_size))

saver.save(sess, "tmp/GRU-hd%d-b%d-200k-%dEp-4.13Loss" % (num_hidden, batch_size , epoch_counter))
'''
ac = 0
ac2 = 0
for i in range(0, 100):
	t1 = time.time()
	sess.run(minimize)
	t2 = time.time()
	ac = ac + ((t2-t1)*1000)
	ac2 = ac2 + 1
ac = ac/ac2

sess.run((tf.shape(flat_probs), flat_probs, tf.shape(y_t_flat), y_t_flat))
sess.run((tf.shape(tf.argmax(flat_probs, 1)), tf.argmax(flat_probs, 1), tf.shape(y_t_flat), y_t_flat))


sess.run((tf.shape(tf.argmax(flat_probs, 1)), tf.to_int32(mask*tf.to_float(tf.argmax(flat_probs, 1))), tf.shape(y_t_flat), y_t_flat))

t1 = time.time()
(acc_train, loss_train) = performance_k(sess, int(0.1*num_iterations_train))
t2 = time.time()
print "Time beetween epochs: %f milliseconds" % ((t2 - t1) * 1000.)   


t1 = time.time()
(acc_test, loss_test) = performance_k(sess, int(0.4*num_iterations_test), True, test_q.dequeue_batch)
t2 = time.time()
print "Time beetween epochs: %f milliseconds" % ((t2 - t1) * 1000.)   

(acc_train, loss_train) = performance_k(sess, int(0.1*num_iterations_train))
(acc_test, loss_test) = performance_k(sess, int(0.4*num_iterations_test), True, test_q.dequeue_batch)
'''

#---------------------------Print Sentence-----------------------------------

'''
new_sentence = [[[7998], [7994], [7941], [7878], [7985], [7996], [7447], [5767]]]
new_sentence = [[[7998], [7929], [7994], [0]]]
'''
new_sentence = [[[7998], [0]]]

def generate_sent(new_sentence):
    while (not new_sentence[0][-1][0] == 7999) and (len(new_sentence[0]) < 40):
        samples = np.random.multinomial(1, sess.run(flat_probs, feed_dict={inputs: new_sentence*batch_size})[0]*0.95)
        sampled_word = np.argmax(samples)
        if sampled_word == 7999:
            break
        new_sentence[0][-1][0] = sampled_word
        new_sentence[0].append([0])
       # print_sentence(np.transpose(new_sentence[0])[0], index_to_word)  
    return new_sentence

sess.run(flat_probs, feed_dict={inputs: new_sentence*30})

def print_sentence(s, index_to_word):
    sentence_str = [index_to_word[x] for x in s[1:-1]]
    try:
        print(" ".join(sentence_str))
    except UnicodeEncodeError:
        print 'UnicodeEncodeError!'
    except:
        print 'Unhandled Exception!'
    sys.stdout.flush()

new_sentence = generate_sent(new_sentence)
print_sentence(np.transpose(new_sentence[0])[0], index_to_word)  



#-----Misc code-----
'''

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
sess.run(padding_queue_train.dequeue_many(50))
t2 = time.time()
sess.run(padding_queue_train.dequeue_many(50))
t3 = time.time()


const = tf.constant(45) 

output = tf.add(inputs, const)




many_inputs= queue.enqueue_many([[4],[6], [10], [8]])


qr = tf.train.QueueRunner(queue, [enqueue_op])

sess = tf.Session()


enqueue_threads = qr.create_threads(sess, start=True)

sess.run(enqueue_op, {data: [5]})

sess.run(output)'''