import tensorflow as tf
import numpy as np
import sys
import os
import time
from load_text import *
from datetime import datetime
from random import shuffle
import gensim, logging

VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "8000"))
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "reddit_comments500.csv")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "50"))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

x_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)
x_train_list = x_train.tolist()
x_train_list_string = [[index_to_word[word] for word in sent] for sent in x_train_list]

model = gensim.models.Word2Vec(x_train_list_string, min_count=20, size=EMBEDDING_DIM)



embedding_matrix = np.zeros((len(word_to_index), EMBEDDING_DIM))
for word, i in word_to_index.items():
    try: 
    	embedding_vector =  model[word]
    except KeyError: 
        embedding_vector = np.zeros(EMBEDDING_DIM)

    except:
    	print('Error')

    if embedding_vector is not None:
    	# words not found in embedding index will be all-zeros.
    	embedding_matrix[i] = embedding_vector

np.save('embedding_matrix_gensim_%dD' % (EMBEDDING_DIM), embedding_matrix)