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
    ncpu = 2
    print("By default, ncpu = ", ncpu)


t1 = time.time()

lem_data = np.load('lem_data.npy')
print(lem_data.shape)

# Define gensim model
size_w2v = 400
iter_w2v = 5
window = 6
min_count = 5

# Name to save the model afterwards
path = 'w2v_models/'
name = 'w2v_s' + str(size_w2v) + '_i' + str(iter_w2v) + '_w' + str(window) + '_mc' + str(min_count)

model_gs = gensim.models.Word2Vec(lem_data, size=size_w2v, window=window, min_count=min_count, iter=iter_w2v, workers=ncpu)

word_vector = model_gs.wv

word_vector.save(path+name)

print("Total time to train", time.time() - t1, "s")