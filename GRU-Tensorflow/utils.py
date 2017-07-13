import csv
import itertools
import numpy as np
import nltk
import time
import sys
import operator
import io
import array
from datetime import datetime
from GRUTensorflow import GRUTensorflow

SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"

def load_data(filename="data/reddit-comments-2015-08.csv", vocabulary_size=2000, min_sent_characters=0):

    word_to_index = []
    index_to_word = []

    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading CSV file...")
    with open(filename, 'rt') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode("utf-8").lower()) for x in reader])
        # Filter sentences
        sentences = [s for s in sentences if len(s) >= min_sent_characters]
        sentences = [s for s in sentences if "http" not in s]
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (SENTENCE_START_TOKEN, x, SENTENCE_END_TOKEN) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocabulary_size-2]
    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
    index_to_word = ["<MASK/>", UNKNOWN_TOKEN] + [x[0] for x in sorted_vocab]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    return X_train, y_train, word_to_index, index_to_word


def train_with_sgd(model, X_train, y_train, embedding_path, learning_rate=0.001, nepoch=20, nepoch_prev=0, decay=0.9,
    callback_every=10000, callback=None):
    model.embedding_init(embedding_path)
    num_examples_seen = 0
    for epoch in range(nepoch):
        num_examples_seen = 0
        print "Epoch = %d" % (epoch + nepoch_prev)
        # For each training example...
        for i in np.random.permutation(len(y_train)):
            # One SGD stepx_train_numpy[10:11], y_train_list[10:11]
            model.sgd_step(X_train[i], y_train[i])
            num_examples_seen += 1
            # Optionally do callback
            if (callback and callback_every and num_examples_seen % callback_every == 0):
                callback(model, num_examples_seen, epoch)            
    return model

def save_model_parameters_tensorflow(model, epoch, outfile):

    ''' np.savez(outfile,
        E=  model.eval_parameter(model.E),
        W = model.eval_parameter(model.W),
        V=  model.eval_parameter(model.V),
        b = model.eval_parameter(model.b),    
        c=  model.eval_parameter(model.c),
        epoch = epoch)
    '''    #sess = model.sess)

    np.savez(outfile, epoch = epoch)
    model.save_parameters("tmp/"+outfile)
    print "Saved model parameters to %s." % outfile

def load_model_parameters_tensorflow(path, hidden_dim, word_dim, modelClass=GRUTensorflow):
    npzfile = np.load(path + '.npz')
    #E, W, V, b, c = npzfile["E"], npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"]
    #hidden_dim, word_dim = E.shape[1], E.shape[0]
    print "Building model model from %s with hidden_dim=%d word_dim=%d" % (path, hidden_dim, word_dim)
    sys.stdout.flush()
    model = modelClass(word_dim, hidden_dim=hidden_dim)
    model.restore_parameters('tmp/'+path)
    
    '''model.assign_parameter(E, model.E)
    model.assign_parameter(W, model.W)
    model.assign_parameter(V, model.V)
    model.assign_parameter(c, model.c)
    model.assign_parameter(b, model.b)
    '''
    return model 


def print_sentence(s, index_to_word):
    sentence_str = [index_to_word[x] for x in s[1:-1]]
    try:
        print(" ".join(sentence_str))
    except UnicodeEncodeError:
        print "UnicodeEncodeError!"
    except:
        print "Unhandled Exception!"
    sys.stdout.flush()

def generate_sentence(model, index_to_word, word_to_index, min_length=5):
    # We start the sentence with the start token
    #new_sentence = [[word_to_index[SENTENCE_START_TOKEN]]]
    new_sentence = [word_to_index[SENTENCE_START_TOKEN]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[SENTENCE_END_TOKEN]:
        #next_word_probs = model.predict([new_sentence])[-1]
        next_word_probs = model.predict(new_sentence)[-1]
        aux = 0
        while aux == 0:
            try:
                samples = np.random.multinomial(1, next_word_probs)
                aux = 1;
            except ValueError:
                print "Value Error!"
                next_word_probs = np.array(next_word_probs)*0.99
            except:
                print "Unhandled Exception!!"
            sys.stdout.flush()
        sampled_word = np.argmax(samples)
        #new_sentence = np.append(new_sentence, [[sampled_word]], axis = 0)
        new_sentence.append(sampled_word)
        # Seomtimes we get stuck if the sentence becomes too long, e.g. "........" :(
        # And: We don't want sentences with UNKNOWN_TOKEN's
        if len(new_sentence) > 30 or sampled_word == word_to_index[UNKNOWN_TOKEN]:
            return new_sentence
    if len(new_sentence) < min_length:
        return None
    #return new_sentence.tolist()
    return new_sentence

def generate_sentences(model, n, index_to_word, word_to_index):
    for i in range(n):
        sent = None
        while not sent:
            sent = generate_sentence(model, index_to_word, word_to_index)
        print_sentence(sent, index_to_word)
