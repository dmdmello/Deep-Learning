import numpy as np
import theano as theano
import theano.tensor as T
import theano.typed_list as tlist
from theano import shared
from theano import function as func
import sys
import os
import time
from datetime import datetime
from load_text import *
from collections import deque

INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "reddit_comments500.csv")


LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "20000"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "300"))
HIDDEN_DIM1 = int(os.environ.get("HIDDEN_DIM1", "300"))
HIDDEN_DIM2 = int(os.environ.get("HIDDEN_DIM2", "300"))

print "Neural Network Specs:"
print "Hiddem Dim 1: %d" % HIDDEN_DIM1
print "Hiddem Dim 2: %d" % HIDDEN_DIM2
print "Batch Size: Single"
print "Embedding Dim : %d" %  EMBEDDING_DIM
print "Voc. Size: %d" % VOCABULARY_SIZE
print "Learning Rate: %d" % LEARNING_RATE


hidden_dim1 = HIDDEN_DIM1
hidden_dim2 = HIDDEN_DIM2
word_dim = VOCABULARY_SIZE
emb_dim = EMBEDDING_DIM

emb_matrix_path = 'embedding_matrix_gensim_300D.npy'

Vocabulary_size = word_dim
x_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, Vocabulary_size, 2000)

sys.stdout.flush()

x_test = x_train[400000:500000]
x_train = x_train[0:400000]

x = T.ivector('x')

#iterator counter
t = theano.shared(name = 't', value = np.array(0).astype('int32'))

f_t = theano.function([],[],updates=[(t, t+1)])
'''
#E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (word_dim, hidden_dim))
U1 = np.random.uniform(-np.sqrt(1./hidden_dim1), np.sqrt(1./hidden_dim1), (3, emb_dim, hidden_dim1))
U2 = np.random.uniform(-np.sqrt(1./hidden_dim2), np.sqrt(1./hidden_dim2), (3, hidden_dim1, hidden_dim2))
W1 = np.random.uniform(-np.sqrt(1./hidden_dim1), np.sqrt(1./hidden_dim1), (3, hidden_dim1, hidden_dim1))
W2 = np.random.uniform(-np.sqrt(1./hidden_dim2), np.sqrt(1./hidden_dim2), (3, hidden_dim2, hidden_dim2))
b1 = np.zeros((3, hidden_dim1))
b2 = np.zeros((3, hidden_dim2))
V = np.random.uniform(-np.sqrt(1./hidden_dim2), np.sqrt(1./hidden_dim2), (hidden_dim2, word_dim))
c = np.zeros((1, word_dim))
'''
#E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (word_dim, hidden_dim))
U1 = np.random.uniform(-np.sqrt(1./hidden_dim1), np.sqrt(1./hidden_dim1), (3, hidden_dim1, emb_dim))
U2 = np.random.uniform(-np.sqrt(1./hidden_dim2), np.sqrt(1./hidden_dim2), (3, hidden_dim2, hidden_dim1))
W1 = np.random.uniform(-np.sqrt(1./hidden_dim1), np.sqrt(1./hidden_dim1), (3, hidden_dim1, hidden_dim1))
W2 = np.random.uniform(-np.sqrt(1./hidden_dim2), np.sqrt(1./hidden_dim2), (3, hidden_dim2, hidden_dim2))
b1 = np.zeros((3, hidden_dim1))
b2 = np.zeros((3, hidden_dim2))
V = np.random.uniform(-np.sqrt(1./hidden_dim2), np.sqrt(1./hidden_dim2), (word_dim, hidden_dim2))
c = np.zeros((1, word_dim))



# Theano: Created shared variables
mU1 = theano.shared(name='mU1', value=np.zeros(U1.shape).astype(theano.config.floatX))
mU2 = theano.shared(name='mU2', value=np.zeros(U2.shape).astype(theano.config.floatX))
mW1 = theano.shared(name='mW1', value=np.zeros(W1.shape).astype(theano.config.floatX))
mW2 = theano.shared(name='mW2', value=np.zeros(W2.shape).astype(theano.config.floatX))
mb1 = theano.shared(name='mb1', value=np.zeros(b1.shape).astype(theano.config.floatX))
mb2 = theano.shared(name='mb2', value=np.zeros(b2.shape).astype(theano.config.floatX))
mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))

#E = theano.shared(name='E', value=E.astype(theano.config.floatX))
E = theano.shared(name='E', value = np.transpose(np.load(emb_matrix_path)).astype(theano.config.floatX))
U1 = theano.shared(name='U1', value=U1.astype(theano.config.floatX))
U2 = theano.shared(name='U2', value=U2.astype(theano.config.floatX))
W1 = theano.shared(name='W1', value=W1.astype(theano.config.floatX))
W2 = theano.shared(name='W2', value=W2.astype(theano.config.floatX))
b1 = theano.shared(name='b1', value=b1.astype(theano.config.floatX))
b2 = theano.shared(name='b2', value=b2.astype(theano.config.floatX))
V = theano.shared(name='V', value=V.astype(theano.config.floatX))
c = theano.shared(name='c', value=c.astype(theano.config.floatX))

y = x[1:]

def forward_prop_step(x_t, s_t1_prev, s_t2_prev):
    # This is how we calculated the hidden state in a simple RNN. No longer!
    # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))
            
    # Word embedding layer
    x_e = E[:,x_t]
            
    # GRU Layer 1
    z_t1 = T.nnet.hard_sigmoid(U1[0].dot(x_e) + W1[0].dot(s_t1_prev) + b1[0])
    r_t1 = T.nnet.hard_sigmoid(U1[1].dot(x_e) + W1[1].dot(s_t1_prev) + b1[1])
    c_t1 = T.tanh(U1[2].dot(x_e) + W1[2].dot(s_t1_prev * r_t1) + b1[2])
    s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
            
    # GRU Layer 2
    z_t2 = T.nnet.hard_sigmoid(U2[0].dot(s_t1) + W2[0].dot(s_t2_prev) + b2[0])
    r_t2 = T.nnet.hard_sigmoid(U2[1].dot(s_t1) + W2[1].dot(s_t2_prev) + b2[1])
    c_t2 = T.tanh(U2[2].dot(s_t1) + W2[2].dot(s_t2_prev * r_t2) + b2[2])
    s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev
            
    # Final output calculation
    # Theano's softmax returns a matrix with one row, we only need the row
    probs_t = T.nnet.softmax(V.dot(s_t2) + c)[0]

    return [probs_t, s_t1, s_t2]



'''

def forward_prop_step(x_t_padded, s_t1_prev, s_t2_prev):
    # Word embedding layer
    x_e = E[x_t_padded]
    
    # GRU Layer 1
    z_t1 = T.nnet.hard_sigmoid(x_e.dot(U1[0]) + s_t1_prev.dot(W1[0]) + b1[0])
    r_t1 = T.nnet.hard_sigmoid(x_e.dot(U1[1]) + s_t1_prev.dot(W1[1]) + b1[1])
    c_t1 = T.tanh(x_e.dot(U1[2]) + (s_t1_prev * r_t1).dot(W1[2]) + b1[2])
    s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
    
    # GRU Layer 2
    z_t2 = T.nnet.hard_sigmoid(s_t1.dot(U2[0]) + s_t2_prev.dot(W2[0]) + b2[0])
    r_t2 = T.nnet.hard_sigmoid(s_t1.dot(U2[1]) + s_t2_prev.dot(W2[1]) + b2[1])
    c_t2 = T.tanh(s_t1.dot(U2[2]) + (s_t2_prev * r_t2).dot(W2[2]) + b2[2])
    s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev

    # Final output calculation
    # Theano's softmax returns a matrix with one row, we only need the row
    probs_t = T.nnet.softmax(s_t2.dot(V) + c[0])

    return [probs_t, s_t1, s_t2]
'''

[probs, s, s2], updates = theano.scan(
    forward_prop_step,
    sequences=x[:-1],
    truncate_gradient=-1,
    outputs_info=[None, 
                  dict(initial=T.zeros(hidden_dim1)),
                  dict(initial=T.zeros(hidden_dim2))])

#Swaps batches and words axes
#probs_swaped = probs.swapaxes(0,1)

prediction = T.argmax(probs, axis=1)

#probs.eval({x:[1,4,5,6]})


#flat_probs = probs_swaped.reshape([-1, word_dim])
#y_flat = y.reshape([-1])


losses = T.sum(T.nnet.categorical_crossentropy(probs + 1e-16, y))
mean_losses = theano.function([x],losses/T.shape(y))

#losses.eval({x:[1,4,5,6]})


dU1 = T.grad(losses, U1)
dU2 = T.grad(losses, U2)
dW1 = T.grad(losses, W1)
dW2 = T.grad(losses, W2)
db1 = T.grad(losses, b1)
db2 = T.grad(losses, b2)
dV = T.grad(losses, V)
dc = T.grad(losses, c)

predict = theano.function([x], probs)
predict_class = theano.function([x], prediction)
bptt = theano.function([x], [dU1, dU1, dW2, dW2, db1, db1, dV, dc])
mistakes = T.neq(y, T.argmax(probs, axis = 1))
#error = theano.function([x],T.sum(T.cast(mistakes, 'float32'))/T.sum(mask))
error = theano.function([x], T.sum(T.cast(mistakes, 'float32'))/T.shape(y))

mistakes.eval({x:[1,4,5,6]})

learning_rate = T.scalar('learning_rate')
decay = T.scalar('decay')

#Adam
t_upd = t + 1

mU1_upd = decay * mU1 + (1 - decay) * dU1 ** 2
mU2_upd = decay * mU2 + (1 - decay) * dU2 ** 2
mW1_upd = decay * mW1 + (1 - decay) * dW1 ** 2
mW2_upd = decay * mW2 + (1 - decay) * dW2 ** 2
mb1_upd = decay * mb1 + (1 - decay) * db1 ** 2
mb2_upd = decay * mb2 + (1 - decay) * db2 ** 2
mV_upd = decay * mV + (1 - decay) * dV ** 2
mc_upd = decay * mc + (1 - decay) * dc ** 2

apply_grads = theano.function(
    [x, learning_rate, theano.In(decay, value= 0.9)],
    [], 
    updates=[(U1, U1 - learning_rate * dU1 / T.sqrt(mU1_upd + 1e-6)),
             (U2, U2 - learning_rate * dU2 / T.sqrt(mU2_upd + 1e-6)),
             (W1, W1 - learning_rate * dW1 / T.sqrt(mW1_upd + 1e-6)),
             (W2, W2 - learning_rate * dW2 / T.sqrt(mW2_upd + 1e-6)),
             (b1, b1 - learning_rate * db1 / T.sqrt(mb1_upd + 1e-6)),
             (b2, b2 - learning_rate * db2 / T.sqrt(mb2_upd + 1e-6)),
             (V, V - learning_rate * dV / T.sqrt(mV_upd + 1e-6)),
             (c, c - learning_rate * dc / T.sqrt(mc_upd + 1e-6)),
             (mU1, mU1_upd),
             (mU2, mU2_upd),
             (mW1, mW1_upd),
             (mW2, mW2_upd),
             (mb1, mb1_upd),
             (mb2, mb2_upd)
            ])

'''
def calculate_total_loss(X):
    return np.sum([ce_error(x) for x in X])
    
def calculate_loss(X):
    # Divide calculate_loss by the number of words
    num_words = np.sum([len(x) for x in X])
    return calculate_total_loss(X)/float(num_words)

indx = [random_indexes.popleft() for i in range(batch_size)]

    apply_grads(x_train[indx], 0.001)

ce_error(x_train[indx])

'''
sys.stdout.flush()
flag_break = False
epoch = 14
epoch_counter = 0
performance_test_hist = []
performance_train_hist = []
train_set_size = len(x_train)
test_set_size = len(x_test)
num_iterations_train = train_set_size
num_iterations_test = test_set_size

random_indexes =  deque([np.random.randint(train_set_size) 
    for i in range(int(1.3 * train_set_size)*epoch)])

random_indexes_test =  deque([np.random.randint(test_set_size) 
    for i in range(int(2.2 *test_set_size)*epoch)])


def performance_k(k, train_set = True):
    losses_ac = 0.0
    classification_error_ac = 0.0
    num_int = k
    t1 = time.time()
    for i in range(num_int):
        if (train_set):
            indx = random_indexes.popleft()
            inp = x_train[indx]
        else:
            indx = random_indexes_test.popleft()
            inp = x_test[indx]
        try:
            classification_error = error(inp)
            losses = mean_losses(inp)

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
    print "Time beetween epochs: %f milliseconds" % ((t2 - t1) * 1000.)

    return(1-(classification_error_ac/num_int), losses_ac/num_int)


for i in range(epoch):
    if flag_break:
        break

    print "--------------------------------------------------"
    print "Epoch - ",str(i)
    print "--------------------------------------------------"

    t1 = time.time()
    for j in range(0, int(num_iterations_train)):
        
        if (j % int(num_iterations_train/2) == 0):
            print"---------------------RESULTS----------------------"
            (acc_train, loss_train) = performance_k(int(0.02*num_iterations_train))
            (acc_test, loss_test) = performance_k(int(0.02*num_iterations_test), False)

            #performance_train_hist.append((acc_train, loss_train))
            #performance_test_hist.append((acc_test, loss_test))
            print"Train accuracy and losses for %d iterations:" % (int(0.1*num_iterations_train))
            print(acc_train, loss_train)
            print"Test accuracy and losses for %d iterations:" % (int(0.4*num_iterations_test))
            print(acc_test, loss_test)
            sys.stdout.flush()

        '''
        if (j % int(num_iterations_train/10) == 0):

            indx = [random_indexes.popleft() for k in range(batch_size)]
            inp = x_train[indx]

            apply_grads(inp, LEARNING_RATE)
            print "--------------------------------------------------"
            print "--------------------------------------------------"
            print "--------------------------------------------------"
            print "------------INICIO DAS AMOSTRAS-------------------"
            print "--------------------------------------------------"
            print "--------------------------------------------------"
            print "j = %d" % (j)
            print "epoch = %d" % (i)
            print "--------------------------------------------------"
            print "--------------------------------------------------"


            print(dU1.eval({x:inp}))
            print "--------------------------------------------------"
            print(dU2.eval({x:inp}))
            print "--------------------------------------------------"
            print(dW1.eval({x:inp}))
            print "--------------------------------------------------"        
            print(dW2.eval({x:inp}))
            print "--------------------------------------------------"  

            print(vU1.eval())
            print "--------------------------------------------------"
            print(vU2.eval())
            print "--------------------------------------------------"
            print(vW1.eval())
            print "--------------------------------------------------"
            print(vW2.eval())
            print "--------------------------------------------------"
            print(vV.eval())
            print "--------------------------------------------------"
            print "--------------------------------------------------"
            print "--------------------------------------------------"
            print "---------------FIM DAS AMOSTRAS-------------------"
            print "--------------------------------------------------"
            print "--------------------------------------------------"
            print "--------------------------------------------------"
        '''
        try:
            

            indx = random_indexes.popleft()
            inp = x_train[indx]

            apply_grads(inp, LEARNING_RATE)

            #if np.isnan(mean_masked_losses(x_train[indx]))):
            #    flag_break = True
            #    break   
            sys.stdout.flush()
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
    np.savez("perf_rec_GRU2ly-1hd%d-2hd%d-SingBatch-500k" % (hidden_dim1, hidden_dim2), performance_train_hist, performance_test_hist)

'''



logits.eval({x:[[4,3], [1], [3, 1, 1, 2], [3, 2, 2], [3, 4, 1, 1, 1, 2, 2]]})
error_function([[4,3], [1], [3, 1, 1, 2], [3, 2, 2], [3, 4, 1, 1, 1, 2, 2]],7)


#----------------GRAD BENCHMARK---------------------#
except_acc = 0
t1 = time.time()
for i in range(1000):
    indx = [random_indexes.popleft() for i in range(batch_size)]
    try:

        inp = x_train[indx]
        apply_grads(inp, LEARNING_RATE)

    except KeyboardInterrupt : 
        flag_break = True
        print ("KeyboardInterrupt")
        break

    except: 
        except_acc = except_acc + 1 


t2 = time.time()
print "Time beetween epochs: %f milliseconds" % ((t2 - t1) * 1000.)   




#----------------ClASSIFICATION BENCHMARK---------------------#

(acc_train, loss_train) = performance_k(1000)






time_acc = 
print "Time beetween epochs: %f milliseconds" % ((t2 - t1) * 1000.)   

x_e = E[:,[3, 1, 4, 4]
o, updates = theano.scan(
    forward_prop_step,
    sequences=x_padded.transpose(),
    outputs_info= None)


f1 = theano.function([x, max_x], [x_padded] )
f1([[4,3], [1, 3, 3, 2], [1, 2, 2]], 4)


f2 = theano.function([x, max_x], o )
f2([[4,3], [1, 3, 3, 2], [1, 2, 2]], 4)


x = T.imatrix('x')
x_vec = T.ivector('x_vec')
fe = theano.function([x_vec], E[:,x_vec])




shape_sub = shared(0)
a = T.sub(T.shape(x[0]),T.shape(x[1]))

f = func([x], a, updates={(shape_sub, a[0])})

f([[[4,3], [3,7]], 2])
f2 = T.zeros(shape_sub)

T.zeros(a[0]).eval({x:[[4,3,1, 6, 6, 7, 8], [5, 7,6], [4, 7, 1, 1]]}) 
f([[3], [3, 1,3]])[-1]
f([[3, 1, 4]])

	x1 = T.ivector('x1')
	x2 = T.ivector('x2')
	shape_sub = T.sub(T.shape(x1),T.shape(x2))
	vec = T.ivector('x1')

	zeros = T.zeros(shape_sub)

	f = theano.function([x1, x2], T.zeros(shape_sub))		

x1 = T.ivector('x1')
shape_sub = x1[0] - x1[1]
zeros = T.zeros(shape_sub)

'''
