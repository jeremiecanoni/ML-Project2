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
    ncpu = 3
    print("By default, ncpu = ", ncpu)

# Setting rigth number of threads for tensorflow
tf.config.threading.set_intra_op_parallelism_threads(ncpu)
tf.config.threading.set_inter_op_parallelism_threads(ncpu)
print(tf.config.threading.get_intra_op_parallelism_threads(),
tf.config.threading.get_inter_op_parallelism_threads(), flush=True)


x = np.load('converted_tweets/x_train_nf_1_b.npy')
labels = np.load('converted_tweets/labels_nf_1_b.npy')
x_test_ai = np.load('converted_tweets/x_test.npy')

print("x.shape", x.shape)

# Define neural network parameters
filters, kernel_size, batch_size = 500, 5, 150
epochs, hidden_dims = 2, 250

model = build_model(filters, kernel_size, hidden_dims)

x, y_train, x_test, y_test = split_data(x, labels, ratio=0.8)

t_tf = time.time()


model.fit(x, y_train, batch_size=batch_size,
          epochs=epochs, validation_data=(x_test, y_test))

y_pred = np.ndarray.flatten(model.predict_classes(x_test_ai, batch_size=batch_size))

# Replace for submission
y_pred = np.where(y_pred == 0, -1, y_pred)

path_csv = 'Subs/'
csv_name = 'sub_' + 'tf_e' + str(epochs) + '_f' + str(filters) + '_bs' + str(batch_size) \
           + '_hd' + str(hidden_dims) + '_ks' + str(kernel_size)

create_csv_submission(y_pred, path_csv + csv_name + '.csv')
print("Output name:", csv_name)
print("Time to train network = ", (time.time()-t_tf)/60, "min")
