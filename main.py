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
from multiprocessing import Pool


def convert_w2v_p(tweet):
    if len(tweet) == 0:
            x = np.zeros((len_max_tweet, size_w2v), dtype=np.float32)
    else:
        vec = word_vector[tweet]
        x = np.transpose(sequence.pad_sequences(vec.T, maxlen=len_max_tweet, dtype=np.float32))

    return x


def find_words_not_in_vocab_p(tweet):
    corr_tweet = [w for w in tweet if w not in words_not_in_vocab]
    return corr_tweet


# Get number of cores to use
if 'NUM_THREADSPROCESSES' in os.environ:
    ncpu = os.environ['NUM_THREADSPROCESSES']
    ncpu = int(ncpu)
    print('ncpu = ', ncpu, flush=True)

else:
    ncpu = 4
    print("By default, ncpu = ", ncpu)

# Setting rigth number of threads for tensorflow
tf.config.threading.set_intra_op_parallelism_threads(ncpu)
tf.config.threading.set_inter_op_parallelism_threads(ncpu)
print(tf.config.threading.get_intra_op_parallelism_threads(),
tf.config.threading.get_inter_op_parallelism_threads(), flush=True)

path_w2v = 'w2v_models/'
name_w2v = 'w2v_s400_i5_w6_mc5'
word_vector = gensim.models.KeyedVectors.load(path_w2v + name_w2v)

# Load processed data
print("Loading Data ...", flush=True)
path_pr = "Processed_Data/"
lem_data = np.load(path_pr + 'lem_data_nf.npy', allow_pickle=True)
labels = np.load(path_pr + 'labels_train_nf.npy', allow_pickle=True)

# If labels are -1 instead of 0
labels = np.where(labels == -1, 0, labels)

# To train without the full set
n_train = -1

if n_train > 0:
    lem_data = lem_data[:n_train]
    labels = labels[:n_train]

lem_data_test = np.load(path_pr + 'lem_data_test.npy', allow_pickle=True)

print("Removing word not in vocab", flush=True)
# Remove words that are not in vocabulary
t_rm = time.time()


# Take all the words in the vocabulary
gensim_words = set(word_vector.vocab)
print(len(gensim_words))
# Take all the words in dataset
all_words = set([word for tweet in lem_data for word in tweet])
print(len(all_words))
all_words = all_words.union(set([word for tweet in lem_data_test for word in tweet]))
print(len(all_words))

words_not_in_vocab = set(all_words - gensim_words)

# Remove words not in vocab using multiprocessing
with Pool(ncpu) as p:
    final_text = p.map(find_words_not_in_vocab_p, lem_data)
    final_text_test = p.map(find_words_not_in_vocab_p, lem_data_test)

print("Time to remove words:", time.time()-t_rm, "s")

print("Computing len_max_tweet", flush=True)

# Max length of tweet (after removed not in vocab words)
len_max_tweet = np.max([len(tweet) for tweet in final_text])
len_max_tweet = np.max((len_max_tweet, np.max([len(tweet) for tweet in final_text_test])))

size_w2v = word_vector.vector_size

print("Start to convert word to vector", flush=True)
t_w2v = time.time()

# Convert words to vec using multiprocessing
with Pool(ncpu, maxtasksperchild=5000) as p:
    x = p.map(convert_w2v_p, final_text)
    x_test_ai = p.map(convert_w2v_p, final_text_test)

x = np.asarray(x)
x_test_ai = np.asarray(x_test_ai)
print(x.shape, x_test_ai.shape)

print("Time to convert words into vec:", time.time()-t_w2v, "s")


# Define neural network parameters
filters, kernel_size, batch_size = 400, 5, 150
epochs, hidden_dims = 2, 250

model = build_model(filters, kernel_size, hidden_dims)

x_train, y_train, x_test, y_test = split_data(x, labels, ratio=0.8)

t_tf = time.time()


model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs, validation_data=(x_test, y_test))

y_pred = np.ndarray.flatten(model.predict_classes(x_test_ai, batch_size=batch_size))

# Replace for submission
y_pred = np.where(y_pred == 0, -1, y_pred)

path_csv = 'Subs/'
csv_name = 'sub_' + name_w2v + 'tf_e' + str(epochs) + '_f' + str(filters) + '_bs' + str(batch_size) \
           + '_hd' + str(hidden_dims) + '_ks' + str(kernel_size)

create_csv_submission(y_pred, path_csv + csv_name + '.csv')
print("Output name:", csv_name)
print("Time to train network = ", (time.time()-t_tf)/60, "min")
