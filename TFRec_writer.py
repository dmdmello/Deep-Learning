import tensorflow as tf
import numpy as np
import sys
import os
import time
from load_text import *
from datetime import datetime
from random import shuffle
from collections import deque
tf.reset_default_graph()


PATH_TF_REC_TRAIN = os.environ.get("PATH_TF_REC_TRAIN", "TFRec2/TFRecordfile500k_20kDim")
PATH_TF_REC_TEST = os.environ.get("PATH_TF_REC_TEST", "TFRec2/TFRecordfile500k_20kDim_test")
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "20000"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "reddit_comments500.csv")


path_TFRecord_train = PATH_TF_REC_TRAIN
path_TFRecord_test = PATH_TF_REC_TEST

# Load data
x_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)
x_train_list = x_train.tolist()

#conversão dados do reddit para o formato numpy
x_train_numpy = [np.array([x_train_list[i]]).transpose() for i in range(len(x_train_list))]

train_size = int(len(x_train_numpy)*0.80)

#---------------TFRecord-Format-----------------------
sequences_train = x_train_numpy[0:train_size]
sequences_test = x_train_numpy[train_size:len(x_train_numpy)]

#label_sequences = x_train_list[0:]
#label_sequences_test = x_train_list[:]

def make_example(sequence):
    # The object we return
    ex = tf.train.SequenceExample()
    # A non-sequential feature of our example
    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list["tokens"]
    for token in sequence:
        fl_tokens.feature.add().int64_list.value.append(token)
    return ex

# Write all examples into a TFRecords file

#------------------------------TRAINING SET WRITER-----------------------------
writer_train = tf.python_io.TFRecordWriter(path_TFRecord_train)
for sequence in sequences_train:
    ex = make_example(sequence)
    writer_train.write(ex.SerializeToString())
writer_train.close()

#--------------------------------TEST SET WRITER--------------------------------
writer_test = tf.python_io.TFRecordWriter(path_TFRecord_test)
for sequence in sequences_test:
    ex = make_example(sequence)
    writer_test.write(ex.SerializeToString())
writer_test.close()