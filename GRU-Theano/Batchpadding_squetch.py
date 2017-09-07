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

INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "reddit_comments30.csv")
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "8000"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "140"))

batch_size = 5
hidden_dim = HIDDEN_DIM
word_dim = VOCABULARY_SIZE

x_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)

#iterator counter
t = theano.shared(name = 't', value = np.array(0).astype('int32'))
x = tlist.TypedListType(T.ivector)()

#wl = T.ivector('wl')
l = tlist.length(x)

def get_shapes(index, x):

	shape_ex = T.shape(x[index])
	return shape_ex[0]

x_shapes, last_output = theano.scan(fn=get_shapes, 
							outputs_info=None, 
							non_sequences = [x],
							sequences = [T.arange(batch_size, dtype = 'int64')]
						    )

#f = theano.function([x], T.shape(x[T.argmax(x_shapes)])) 	

max_x_idx = T.argmax(x_shapes)
max_x = x_shapes[max_x_idx]

def batch_padding(index, x, max_x, x_shapes):

	#f = func([wl_t], word_length, updates = {(num_zeros, 10-word_length[0])})
	#f(wl_t)
	#max_x = x_shapes[max_x_idx]
	shape_ex = x_shapes[index]
	diff = max_x-shape_ex
	zero_vec = T.arange(diff, dtype = 'int64')

	y_t = T.concatenate([x[index], T.zeros_like(zero_vec)], axis = 0)
	return y_t


x_padded, updates = theano.scan(fn=batch_padding, 
							outputs_info=None, 
							non_sequences = [x, max_x, x_shapes],
							sequences = [T.arange(batch_size, dtype = 'int64')]
						    )

f_t = theano.function([],[],updates=[(t, t+1)])

E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (word_dim, hidden_dim))
U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, word_dim))
b = np.zeros((6, hidden_dim))
c = np.zeros((1, word_dim))

# Theano: Created shared variables
mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))

vE = theano.shared(name='vE', value=np.zeros(E.shape).astype(theano.config.floatX))
vU = theano.shared(name='vU', value=np.zeros(U.shape).astype(theano.config.floatX))
vV = theano.shared(name='vV', value=np.zeros(V.shape).astype(theano.config.floatX))
vW = theano.shared(name='vW', value=np.zeros(W.shape).astype(theano.config.floatX))
vb = theano.shared(name='vb', value=np.zeros(b.shape).astype(theano.config.floatX))
vc = theano.shared(name='vc', value=np.zeros(c.shape).astype(theano.config.floatX))

E = theano.shared(name='E', value=E.astype(theano.config.floatX))
U = theano.shared(name='U', value=U.astype(theano.config.floatX))
W = theano.shared(name='W', value=W.astype(theano.config.floatX))
V = theano.shared(name='V', value=V.astype(theano.config.floatX))
b = theano.shared(name='b', value=b.astype(theano.config.floatX))
c = theano.shared(name='c', value=c.astype(theano.config.floatX))


y = x_padded[:,1:]

def forward_prop_step(x_t_padded, s_t1_prev, s_t2_prev):
    # Word embedding layer
    x_e = E[x_t_padded]
    
    # GRU Layer 1
    z_t1 = T.nnet.hard_sigmoid(x_e.dot(U[0]) + s_t1_prev.dot(W[0]) + b[[0]])
    r_t1 = T.nnet.hard_sigmoid(x_e.dot(U[1]) + s_t1_prev.dot(W[1]) + b[[1]])
    c_t1 = T.tanh(x_e.dot(U[2]) + (s_t1_prev * r_t1).dot(W[2]) + b[[2]])
    s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
    
    # GRU Layer 2
    z_t2 = T.nnet.hard_sigmoid(s_t1.dot(U[3]) + s_t2_prev.dot(W[3]) + b[[3]])
    r_t2 = T.nnet.hard_sigmoid(s_t1.dot(U[4]) + s_t2_prev.dot(W[4]) + b[[4]])
    c_t2 = T.tanh(s_t1.dot(U[5]) + (s_t2_prev * r_t2).dot(W[5]) + b[[5]])
    s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev

    # Final output calculation
    # Theano's softmax returns a matrix with one row, we only need the row
    probs_t = T.nnet.softmax(s_t2.dot(V) + c[[0]])

    return [probs_t, s_t1, s_t2]

[probs, s, s2], updates = theano.scan(
    forward_prop_step,
    sequences=x_padded[:,0:-1].transpose(),
    truncate_gradient=-1,
    outputs_info=[None, 
                  dict(initial=T.zeros([batch_size, hidden_dim])),
                  dict(initial=T.zeros([batch_size, hidden_dim]))])

#Swaps batches and words axes
probs_swaped = probs.swapaxes(0,1)

prediction = T.argmax(probs_swaped, axis=2)

flat_probs = probs_swaped.reshape([-1, word_dim])
y_flat = y.reshape([-1])


losses = T.nnet.categorical_crossentropy(flat_probs + 1e-16, y_flat)

mask = T.sgn(y_flat)
masked_losses = T.sum(mask * losses)
mean_masked_losses = theano.function([x], masked_losses/T.sum(mask))



dE = T.grad(masked_losses, E)
dU = T.grad(masked_losses, U)
dW = T.grad(masked_losses, W)
db = T.grad(masked_losses, b)
dV = T.grad(masked_losses, V)
dc = T.grad(masked_losses, c)

predict = theano.function([x], probs)
predict_class = theano.function([x], prediction)
bptt = theano.function([x], [dE, dU, dW, db, dV, dc])
mistakes = T.neq(y_flat, mask*T.argmax(flat_probs, axis = 1))
error = theano.function([x],T.sum(T.cast(mistakes, 'float32'))/T.sum(mask))

learning_rate = T.scalar('learning_rate')
beta1 = T.scalar('beta1')
beta2 = T.scalar('beta2')
epsilon = T.scalar('epsilon')

#Adam
t_upd = t + 1

mE_upd = beta1 * mE + (1 - beta1) * dE
mU_upd = beta1 * mU + (1 - beta1) * dU
mW_upd = beta1 * mW + (1 - beta1) * dW
mV_upd = beta1 * mV + (1 - beta1) * dV
mb_upd = beta1 * mb + (1 - beta1) * db
mc_upd = beta1 * mc + (1 - beta1) * dc

vE_upd = beta2 * vE + (1 - beta2) * dE ** 2
vU_upd = beta2 * vU + (1 - beta2) * dU ** 2
vW_upd = beta2 * vW + (1 - beta2) * dW ** 2
vV_upd = beta2 * vV + (1 - beta2) * dV ** 2
vb_upd = beta2 * vb + (1 - beta2) * db ** 2
vc_upd = beta2 * vc + (1 - beta2) * dc ** 2

learning_rate_upd = learning_rate * T.cast(T.sqrt((1 - beta2 ** t_upd) / (1 - beta1 ** t_upd)), dtype='float32')

apply_grads = theano.function(
    [x, learning_rate, theano.In(beta1, value= 0.9), theano.In(beta2, value= 0.99), 
    theano.In(epsilon, value= 1e-16)],
    [], 
    updates=[(E, E - learning_rate_upd * mE_upd / (T.sqrt(vE_upd) + epsilon)),
             (U, U - learning_rate_upd * mU_upd / (T.sqrt(vU_upd) + epsilon)),
             (W, W - learning_rate_upd * mW_upd / (T.sqrt(vW_upd) + epsilon)),
             (V, V - learning_rate_upd * mV_upd / (T.sqrt(vV_upd) + epsilon)),
             (b, b - learning_rate_upd * mb_upd / (T.sqrt(vb_upd) + epsilon)),
             (c, c - learning_rate_upd * mc_upd / (T.sqrt(vc_upd) + epsilon)),
             (mE, mE_upd),
             (mU, mU_upd),
             (mW, mW_upd),
             (mV, mV_upd),
             (mb, mb_upd),
             (mc, mc_upd),
             (vE, vE_upd),
             (vU, vU_upd),
             (vW, vW_upd),
             (vV, vV_upd),
             (vb, vb_upd),
             (vc, vc_upd),
             (t, t_upd)
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
flag_break = False
epoch = 20

performance_test_hist = []
performance_train_hist = []
train_set_size = len(x_train)
test_set_size = 100000
num_iterations_train = train_set_size/batch_size
num_iterations_test = test_set_size/batch_size

random_indexes =  deque([np.random.randint(train_set_size) 
    for i in range((train_set_size)*epoch)])


def performance_k(k):
    losses_ac = 0.0
    classification_error_ac = 0.0
    num_int = k
    for i in range(num_int):
        t1 = time.time()
        indx = [random_indexes.popleft() for i in range(batch_size)]
        inp = x_train[indx]
        try:
            classification_error = error(inp)
            losses = mean_masked_losses(inp)

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


for i in range(epoch):
    if flag_break:
        break

    print "--------------------------------------------------"
    print "Epoch - ",str(i)
    print "--------------------------------------------------"

    t1 = time.time()
    for j in range(0, int(num_iterations_train)):
        
        if (j % int(num_iterations_train/3) == 0):
            (acc_train, loss_train) = performance_k(int(0.1*num_iterations_train))
            #(acc_test, loss_test) = performance_k(sess, int(0.4*num_iterations_test), True, test_q.dequeue_batch)

            #performance_train_hist.append((acc_train, loss_train))
            #performance_test_hist.append((acc_test, loss_test))

            print"---------------------RESULTS----------------------"
            print"Train accuracy and losses for %d iterations:" % (int(0.1*num_iterations_train))
            print(acc_train, loss_train)
            print"Test accuracy and losses for %d iterations:" % (int(0.4*num_iterations_test))
            #print(acc_test, loss_test)

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

            indx = [random_indexes.popleft() for i in range(batch_size)]
            apply_grads(x_train[indx], 0.001)

            #if np.isnan(mean_masked_losses(x_train[indx]))):
            #    flag_break = True
            #    break   

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






logits.eval({x:[[4,3], [1], [3, 1, 1, 2], [3, 2, 2], [3, 4, 1, 1, 1, 2, 2]]})
error_function([[4,3], [1], [3, 1, 1, 2], [3, 2, 2], [3, 4, 1, 1, 1, 2, 2]],7)

t1 = time.time()
for i in range(10):
    random_indexes.popleft()
t2 = time.time()
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