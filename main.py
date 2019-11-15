from __future__ import print_function
import sys, os
import numpy as np
from helpers import *
import pandas as pd
import tensorflow as tf
import time
from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords
import gensim
from symspellpy.symspellpy import SymSpell
from models import *
from methods import *

# Get number of cores to use
if 'NUM_THREADSPROCESSES' in os.environ:
    ncpu = os.environ['NUM_THREADSPROCESSES']
    ncpu = int(ncpu)
    print('ncpu = ', ncpu, flush=True)

else:
    ncpu = 2
    print("By default, ncpu = ", ncpu)


tf.config.threading.set_intra_op_parallelism_threads(ncpu)
tf.config.threading.set_inter_op_parallelism_threads(ncpu)
print(tf.config.threading.get_intra_op_parallelism_threads(),
tf.config.threading.get_inter_op_parallelism_threads())

path_w2v = 'w2v_models/'
name_w2v = 'w2v_s400_i5_w6_mc5'
word_vector = gensim.models.KeyedVectors.load( path_w2v + name_w2v)

# Load processed data
path_pr = "Processed_Data/"
lem_data = np.load(path_pr + 'lem_data10000.npy')
labels = np.load(path_pr + 'labels_train10000.npy')

lem_data_test = np.load(path_pr + 'lem_data_test100.npy')

# Remove words that are not in vocabulary
find_words_not_in_vocab(word_vector, lem_data)
find_words_not_in_vocab(word_vector, lem_data_test)

# Max length of tweet (after removed not in vocab words)
len_max_tweet = np.max([len(tweet) for tweet in lem_data])
len_max_tweet = np.max((len_max_tweet, np.max([len(tweet) for tweet in lem_data_test])))

x = convert_w2v(word_vector, lem_data, len_max_tweet)
x_test_ai = convert_w2v(word_vector, lem_data_test, len_max_tweet)

# Define neural network parameters
filters, kernel_size, batch_size = 400, 5, 32
epochs, hidden_dims = 10, 250

model = build_model(filters, kernel_size, hidden_dims)

# np stand for not padded
x_train, y_train, x_test, y_test = split_data(x, labels, ratio=0.9)

t_tf = time.time()


model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs, validation_data=(x_test, y_test))

y_pred = np.ndarray.flatten(model.predict_classes(x_test_ai, batch_size=batch_size))

path_csv = 'Subs/'
csv_name = 'sub_' + name_w2v + 'tf_e' + str(epochs) + '_f' + str(filters) + '_bs' + str(batch_size) \
           + '_hd' + str(hidden_dims) + '_ks' + str(kernel_size)

create_csv_submission(y_pred, path_csv + csv_name + '.csv')

print("Time to train network = ", (time.time()-t_tf)/60, "min")
