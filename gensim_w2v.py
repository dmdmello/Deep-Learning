import gensim, logging
import numpy as np
import sys
import os
import time
from utils import *
from datetime import datetime
from random import shuffle
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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


x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)

x_train_list = x_train.tolist()
y_train_list = y_train.tolist()


x_train_list_string = [[index_to_word[word] for word in sent] for sent in x_train_list]

model = gensim.models.Word2Vec(x_train_list_string, min_count=20, size=100)