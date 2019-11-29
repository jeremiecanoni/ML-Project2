from __future__ import print_function
import sys, os
import numpy as np
from helpers import *
import time
from keras.preprocessing.text import Tokenizer
import gensim
from models import *
from methods import *
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf


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
print("Tensorflow num threads:",
      tf.config.threading.get_intra_op_parallelism_threads(),
      tf.config.threading.get_inter_op_parallelism_threads(), flush=True)


# Loading processed data
full = 'f'

text_data = np.load('Processed_Data/data_train_pr_' + full + '_sl3.npy', allow_pickle=True)
labels = np.load('Processed_Data/labels_train_' + full + '_sl3.npy', allow_pickle=True)
text_data_test = np.load('Processed_Data/data_test_pr_sl3.npy', allow_pickle=True)

perm = np.random.permutation(text_data.shape[0])
text_data = text_data[perm]
labels = labels[perm]

# If we don't want to take full dataset
n_train = -1

if n_train > 0:
    text_data = text_data[:n_train]
    labels = labels[:n_train]

# Take all dataset to train gensim word2vec
text_data_tot = np.concatenate((text_data, text_data_test), axis=0)

t1 = time.time()

# Define gensim model
size_w2v = 200
iter_w2v = 6
window_w2v = 6
min_count = 5


print("\nGensim_Parameters:")
print("Size embedding:", size_w2v)
print("Num Iters:", iter_w2v)
print("Window size:", window_w2v)
print("Min count:", min_count)

# Name to save the model afterwards
path_w2v = 'w2v_models/'
name_w2v = 'w2v_f_s' + str(size_w2v) + '_i' + str(iter_w2v) + '_w' + str(window_w2v) + '_mc' + str(min_count)

model_gs = gensim.models.Word2Vec(text_data_tot, size=size_w2v, window=window_w2v, min_count=min_count, iter=iter_w2v,
                                  workers=ncpu)
word_vector = model_gs.wv

word_vector.save(path_w2v+name_w2v)

print("Total time to train gensim", time.time() - t1, "s", flush=True)
print("Saved as:", path_w2v+name_w2v, end="\n")

#path_w2v = 'w2v_models/'
#name_w2v = 'w2v_f_s250_i6_w6_mc5_w14'
#word_vector = gensim.models.KeyedVectors.load(path_w2v + name_w2v)

# Go from gensim to keras embedding, create num_data (vector with idx for each word)
# Choose if keep training embedding layer
k_emb = word_vector.get_keras_embedding(train_embeddings=True)
vocabulary = {word: vector.index for word, vector in word_vector.vocab.items()}
tk = Tokenizer(num_words=len(vocabulary))
tk.word_index = vocabulary
num_data = np.asarray((pad_sequences(tk.texts_to_sequences(text_data), padding='post')))
num_data_test = np.asarray((pad_sequences(tk.texts_to_sequences(text_data_test), padding='post')))

# Parameters for model
filters, kernel_size, batch_size = 250, 5, 100
epochs, hidden_dims, learning_rate = 8, 250, 0.001

model = build_model_emb(k_emb, filters, kernel_size, hidden_dims, num_data.shape[1], learning_rate)

weights = np.asarray(model.layers[0].get_weights())
x_train, x_test, y_train, y_test = train_test_split(num_data, labels, train_size=0.9, random_state=42)

checkpoint = ModelCheckpoint("best_model.hdf5", monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='auto', period=1)

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks=[checkpoint])


best_model = load_model('best_model.hdf5')

y_pred = np.ndarray.flatten(best_model.predict_classes(num_data_test, batch_size=batch_size))

# Replace for submission
y_pred = np.where(y_pred == 0, -1, y_pred)

path_csv = 'Subs/'
csv_name = 'sub_' + full + '_' + name_w2v + '_fil' + str(filters) + '_bs' + str(batch_size) \
           + '_hd' + str(hidden_dims) + '_ks' + str(kernel_size) + '_lr' + str(learning_rate) \
           + "_sl3"

create_csv_submission(y_pred, path_csv + csv_name + '.csv')
print("Output name:", csv_name + '.csv')

