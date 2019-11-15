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
from multiprocessing import Pool
from symspellpy.symspellpy import SymSpell
from models import *
from methods import *


if 'NUM_THREADSPROCESSES' in os.environ:
    ncpu = os.environ['NUM_THREADSPROCESSES']
    ncpu = int(ncpu)
    print('ncpu = ', ncpu, flush=True)

else:
    ncpu = 4
    print("By default, ncpu = ", ncpu)


t_start = time.time()
n = 5000
n_test = 1000
print("n = ", n)
path_neg = 'data/twitter-datasets/train_neg.txt'
path_pos = 'data/twitter-datasets/train_pos.txt'
path_test = 'data/twitter-datasets/test_data.txt'

data_neg = open_file(path_neg)
data_pos = open_file(path_pos)
test_data = open_file(path_test)
t1 = time.time()

data, labels = remove_not_unique_tweet_training(data_pos, data_neg)
test_data = remove_not_unique_tweet_test(test_data)
#test_data = test_data[:n_test]


data, labels = np.asarray(data), np.asarray(labels)
test_data = np.asarray(test_data[:-1])

perm_tot = np.random.permutation(labels.shape[0])
data = data[perm_tot]
labels = labels[perm_tot]

#data_tmp = data[:n]
data_tmp = data

# Correct spelling
print("Start spelling correction", flush=True)

t_s = time.time()

with Pool(ncpu) as p:
    data_tmp = p.map(correct_spelling_p, data_tmp)
    test_data = p.map(correct_spelling_p, test_data)

print("Duration:", time.time() - t_s, "s", flush=True)


data = [tweet.replace('\'', '') for tweet in data_tmp]
data = [text_to_word_sequence(text, filters='!"$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n01234556789\'') for text in data]

test_data = [tweet.replace('\'', '') for tweet in test_data]
test_data = [text_to_word_sequence(text, filters='!"$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n01234556789\'')
             for text in test_data]


data = np.asarray(data)
test_data = np.asarray(test_data)

stop_words = list(stopwords.words('english'))
stop_words.append('u')
stop_words.append('ur')

print("Start Lemmatization", flush=True)
t_s = time.time()


with Pool(ncpu) as p:
    lem_data = p.map(lemmatize_p, data)
    lem_data_test = p.map(lemmatize_p, test_data)


print("Time to lemmatize", time.time()-t_s, "s")

# Max length of tweet
len_max_tweet = np.max([len(tweet) for tweet in lem_data])
len_max_tweet = np.max((len_max_tweet, np.max([len(tweet) for tweet in lem_data_test])))

lem_data = np.asarray(lem_data)
lem_data_test = np.asarray(lem_data_test)

np.save('lem_data', lem_data)
np.save('lem_data_test', lem_data_test)

print("Time for preprocessing: ", time.time() - t1, "s")

print("Start training Word2Vec", flush=True)

# Define gensim model
size_w2v = 250
iter_w2v = 5
#lem_data_tot = np.concatenate((lem_data, lem_data_test), axis=0)
model_gs = gensim.models.Word2Vec(lem_data, size=size_w2v, window=6, min_count=5, iter=iter_w2v, workers=ncpu)


find_words_not_in_vocab(model_gs, lem_data)
find_words_not_in_vocab(model_gs, lem_data_test)

print("W2V trained, word not in vocab removed", flush=True)

# Convert words to vectors
x = convert_w2v(model_gs, lem_data, size_w2v, len_max_tweet)
x_test_real = convert_w2v(model_gs, lem_data_test, size_w2v, len_max_tweet)


# Define neural network parameters
filters, kernel_size, batch_size = 400, 5, 150
epochs, hidden_dims = 5, 250

model = build_model(filters, kernel_size, hidden_dims)

# np stand for not padded
x_train, y_train, x_test, y_test = split_data(x, labels, ratio=1)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs, verbose=1, workers=ncpu, use_multiprocessing=True)

y_pred = np.ndarray.flatten(model.predict_classes(x_test_real, batch_size=batch_size))
create_csv_submission(y_pred, 'csv_workers_mp_test_e_'+str(epochs)+'_10_w2v_train_'+str(size_w2v)+'_iter_'+str(iter_w2v)+'_c_'+str(filters)+'_2.csv')

print("Final duration = ", (time.time()-t_start)/60, "min")
