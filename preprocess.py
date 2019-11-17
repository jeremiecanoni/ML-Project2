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
from methods import *


if 'NUM_THREADSPROCESSES' in os.environ:
    ncpu = os.environ['NUM_THREADSPROCESSES']
    ncpu = int(ncpu)
    print('ncpu = ', ncpu, flush=True)

else:
    ncpu = 2
    print("By default, ncpu = ", ncpu)


t_start = time.time()
n = 10000
n_test = 100
print("n = ", n)
path_neg = 'data/twitter-datasets/train_neg_full.txt'
path_pos = 'data/twitter-datasets/train_pos_full.txt'
path_test = 'data/twitter-datasets/test_data.txt'

data_neg = open_file(path_neg)
data_pos = open_file(path_pos)
test_data = open_file(path_test)
t1 = time.time()

data, labels = remove_not_unique_tweet_training(data_pos, data_neg)
test_data = remove_not_unique_tweet_test(test_data)
#test_data = test_data[:n_test]


data, labels = np.asarray(data), np.asarray(labels)
test_data = np.asarray(test_data)

perm_tot = np.random.permutation(labels.shape[0])
data = data[perm_tot]
labels = labels[perm_tot]

data_tmp = data
#data_tmp = data[:n]
#labels = labels[:n]


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

lem_data = np.asarray(lem_data)
lem_data_test = np.asarray(lem_data_test)

np.save('Processed_Data/lem_data_f', lem_data)
np.save('Processed_Data/lem_data_test', lem_data_test)
np.save('Processed_Data/labels_train_f', labels)

print("Time for preprocessing: ", time.time() - t1, "s")
