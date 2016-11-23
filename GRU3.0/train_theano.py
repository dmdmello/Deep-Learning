import sys
import os
import time
import numpy as np
from utils import *
from datetime import datetime
from gru_theano import GRUTheano

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

path_load = 'pretrained.npz'


if not MODEL_OUTPUT_FILE:
  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
  MODEL_OUTPUT_FILE = "GRU-%s-%s-%s-%s.dat" % (ts, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

# Load data
x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)

print "LOADORNOT = %s" % (LOADORNOT) 
if (LOADORNOT == 'True'):
  # Load parameters of pre-trained model
  model = load_model_parameters_theano(path_load)
else: 
  # Build model
  model = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)

# Print SGD step time
t1 = time.time()
model.sgd_step(x_train[10], y_train[10], LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)
sys.stdout.flush()

# We do this every few examples to understand what's going on
def sgd_callback(model, num_examples_seen, epoch):
  dt = datetime.now().isoformat()
  loss = model.calculate_loss(x_train[:10000], y_train[:10000])
  print("\n%s (%d)" % (dt, num_examples_seen))
  print("--------------------------------------------------")
  print("Loss for first 10K: %f" % loss)
  if num_examples_seen == 100000:
    try:
      loss = model.calculate_loss(x_train, y_train)
    except: 
      print "Unhandled Exception!"
  print("--------------------------------------------------")
  print("Loss for ALL: %f" % loss)
  generate_sentences(model, 50, index_to_word, word_to_index)
  save_model_parameters_theano(model, epoch+nepoch_prev, MODEL_OUTPUT_FILE)
  print("\n")
  sys.stdout.flush()

#for epoch in range(NEPOCH):
#  print "Epoch = %d" %epoch

#print "LOADORNOT = %s" % (LOADORNOT) 
#if (LOADORNOT == 'True'):
#  nepoch_prev=np.load(path_load)["epoch"]
#else: 
#  nepoch_prev=0
nepoch_prev=0


print "Last number of epochs was %d in the loaded parameters, which will be subtracted from the selected value NEPOCH = %d, which now equals %d - %d = %d" % (nepoch_prev, NEPOCH, NEPOCH , nepoch_prev, NEPOCH - nepoch_prev)
nepoch_remain = NEPOCH - nepoch_prev

train_with_sgd(model, x_train, y_train, learning_rate=LEARNING_RATE, nepoch=nepoch_remain, nepoch_prev=nepoch_prev, decay=0.9, callback_every=PRINT_EVERY, callback=sgd_callback)

