import gensim
import os
import sys
import numpy as np
import time


if 'NUM_THREADSPROCESSES' in os.environ:
    ncpu = os.environ['NUM_THREADSPROCESSES']
    ncpu = int(ncpu)
    print('ncpu = ', ncpu, flush=True)

else:
    ncpu = 3
    print("By default, ncpu = ", ncpu)


t1 = time.time()

lem_data = np.load('Processed_Data/data_train_pr_nf.npy', allow_pickle=True)

lem_data_test = np.load('Processed_Data/data_test_pr.npy', allow_pickle=True)
lem_data_tot = np.concatenate((lem_data, lem_data_test), axis=0)
print(lem_data_tot.shape)

perm = np.random.permutation(lem_data_tot.shape[0])
lem_data_tot = lem_data_tot[perm]

# Define gensim model
size_w2v = 400
iter_w2v = 6
window = 6
min_count = 5

# Name to save the model afterwards
path = 'w2v_models/'
name = 'w2v_nf_s' + str(size_w2v) + '_i' + str(iter_w2v) + '_w' + str(window) + '_mc' + str(min_count)

model_gs = gensim.models.Word2Vec(lem_data, size=size_w2v, window=window, min_count=min_count, iter=iter_w2v,
                                  workers=ncpu)
word_vector = model_gs.wv

word_vector.save(path+name)

print("Total time to train", time.time() - t1, "s")
print("Name is:", name)
