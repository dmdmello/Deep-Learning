import tensorflow as tf
import numpy as np
import sys
import os
import time
from utils import *
from datetime import datetime
from random import shuffle


VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "8000"))
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "reddit_comments.csv")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "100"))

embeddings_index = {}
f = open(os.path.join('../', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)

embedding_matrix = np.zeros((len(word_to_index), EMBEDDING_DIM+1))
for word, i in word_to_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = np.append(embedding_vector, 0)


#Assigning values for the extra dimensions in the SENTENCE_END and SENCENTE_START tokens

embedding_matrix[7998][-1] = -100
embedding_matrix[7999][-1] = 100


#Unknown Token becomes the mean value of the rarest words available on the embedding
for i in range(13,40):
	embedding_matrix[1] = embedding_matrix[1]+ embedding_matrix[i]
embedding_matrix[1] = embedding_matrix[1]/(len(range(13,40)))

np.savez(embedding_matrix, 'embedding_matrix_WIKI_100D')