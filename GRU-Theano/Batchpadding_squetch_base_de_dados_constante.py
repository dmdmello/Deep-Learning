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

batch_size = 5
hidden_dim = 40
word_dim = VOCABULARY_SIZE


x_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)

#iterator counter
t = theano.shared(name = 't', value = 0)

#batch_counter = theano.shared(name = 'batch_counter', value = 0)

#x = T.ivector('x')
#max_x  = T.iscalar('max_x')
x = tlist.TypedListType(T.ivector)()

#wl = T.ivector('wl')
l = tlist.length(x)
'''
def batch_padding(index, x_t, max_x):

	#f = func([wl_t], word_length, updates = {(num_zeros, 10-word_length[0])})
	#f(wl_t)

	shape_ex = T.shape(x_t[index])
	zero_vec = T.arange(max_x-shape_ex[0], dtype = 'int64')


	y_t = T.concatenate([x_t[index], T.zeros_like(zero_vec)], axis = 0)
	return y_t
'''

def get_shapes(index, x, t):

	shape_ex = T.shape(x[index + t*batch_size])
	return shape_ex[0]

x_shapes, last_output = theano.scan(fn=get_shapes, 
							outputs_info=None, 
							non_sequences = [x, t],
							sequences = [T.arange(batch_size, dtype = 'int64')]
						    )

#f = theano.function([x], T.shape(x[T.argmax(x_shapes)])) 	

max_x_idx = T.argmax(x_shapes)
max_x = x_shapes[max_x_idx]

def batch_padding(index, x, max_x, x_shapes, t):

	#f = func([wl_t], word_length, updates = {(num_zeros, 10-word_length[0])})
	#f(wl_t)
	#max_x = x_shapes[max_x_idx]
	shape_ex = x_shapes[index]
	diff = max_x-shape_ex
	zero_vec = T.arange(diff, dtype = 'int64')

	y_t = T.concatenate([x[index + t*batch_size], T.zeros_like(zero_vec)], axis = 0)
	return y_t


x_padded, updates = theano.scan(fn=batch_padding, 
							outputs_info=None, 
							non_sequences = [x, max_x, x_shapes, t],
							sequences = [T.arange(batch_size, dtype = 'int64')]
						    )

f1 = theano.function([x], x_padded)
f1([[4,3], [3,7,1,5], [4,6],  [3,7,1,5],  [3,7,1,5], [6, 6, 1, 2, 4, 6, 7], [4,3], [4,3], [4,3], [1,7]])

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
    logits_t = T.nnet.softmax(s_t2.dot(V) + c[[0]])

    return [logits_t, s_t1, s_t2]

[logits, s, s2], updates = theano.scan(
    forward_prop_step,
    sequences=x_padded[:,0:-1].transpose(),
    truncate_gradient=-1,
    outputs_info=[None, 
                  dict(initial=T.zeros([batch_size, hidden_dim])),
                  dict(initial=T.zeros([batch_size, hidden_dim]))])

#Swaps batches and words axes
logits_swaped = logits.swapaxes(0,1)

prediction = T.argmax(logits_swaped, axis=2)

logits_flat = logits_swaped.reshape([-1, word_dim])
y_flat = y.reshape([-1])

losses = T.nnet.categorical_crossentropy(logits_flat + 1e-16, y_flat)

mask = T.sgn(y_flat)
masked_losses = mask * losses
mean_masked_losses = T.sum(masked_losses)/T.sum(mask)

cost = mean_masked_losses

dE = T.grad(cost, E)
dU = T.grad(cost, U)
dW = T.grad(cost, W)
db = T.grad(cost, b)
dV = T.grad(cost, V)
dc = T.grad(cost, c)


predict = theano.function([x], logits)
predict_class = theano.function([x], prediction)
ce_error = theano.function([x], cost)
bptt = theano.function([x], [dE, dU, dW, db, dV, dc])

bptt([[4,3], [1], [3, 1, 1, 2], [3, 2, 2], [3, 4, 1, 1, 1, 2, 2]])


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

learning_rate_upd = learning_rate * T.sqrt((1 - beta2 ** t_upd) / (1 - beta1 ** t_upd))

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


apply_grads([[4,3], [1], [3, 1, 1, 2], [3, 2, 2], [3, 4, 1, 1, 1, 2, 2]], 0.001)

E.eval()
U.eval()
W.eval()
V.eval()

def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=20, nepoch_prev=0, decay=0.9,
    callback_every=10000, callback=None):
    num_examples_seen = 0
    for epoch in range(nepoch):
        num_examples_seen = 0
        print "Epoch = %d" % (epoch + nepoch_prev)
        # For each training example...
        for i in np.random.permutation(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate, decay)
            num_examples_seen += 1
            # Optionally do callback
            if (callback and callback_every and num_examples_seen % callback_every == 0):
                callback(model, num_examples_seen, epoch)            
    return model



performance_test_hist = []
performance_train_hist = []
train_set_size = 300000
test_set_size = 100000
num_iterations_train = train_set_size/batch_size
num_iterations_test = test_set_size/batch_size

flag_break = False
epoch = 20

random_indexes =  deque([np.random.randint(test_set_size+train_set_size) 
	for i in range((test_set_size+train_set_size)*epoch)])



for i in range(epoch):
    if flag_break:
        break

    print "--------------------------------------------------"
    print "Epoch - ",str(i)
    print "--------------------------------------------------"

    t1 = time.time()
    for j in range(0, int(num_iterations_train)):
        
        if (j % int(num_iterations_train/3) == 0):
            #(acc_train, loss_train) = performance_k(sess, int(0.1*num_iterations_train))
            #(acc_test, loss_test) = performance_k(sess, int(0.4*num_iterations_test), True, test_q.dequeue_batch)

            #performance_train_hist.append((acc_train, loss_train))
            #performance_test_hist.append((acc_test, loss_test))

            print"---------------------RESULTS----------------------"
            print"Train accuracy and losses for %d iterations:" % (int(0.1*num_iterations_train))
           # print(acc_train, loss_train)
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
            apply_grads(x)


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



logits.eval({x:[[4,3], [1], [3, 1, 1, 2], [3, 2, 2], [3, 4, 1, 1, 1, 2, 2]]})
error_function([[4,3], [1], [3, 1, 1, 2], [3, 2, 2], [3, 4, 1, 1, 1, 2, 2]],7)




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