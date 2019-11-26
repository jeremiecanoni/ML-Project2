import time
from helpers import *
from models import *
from methods import *
from multiprocessing import Pool
import sys, os
import gensim


# Do test also ?
test = True


# Get number of cores to use
if 'NUM_THREADSPROCESSES' in os.environ:
    ncpu = os.environ['NUM_THREADSPROCESSES']
    ncpu = int(ncpu)
    print('ncpu = ', ncpu, flush=True)

else:
    ncpu = 3
    print("By default, ncpu = ", ncpu)


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


path_save = "converted_tweets/"

path_w2v = 'w2v_models/'
name_w2v = 'w2v_nf_s400_i5_w6_mc5'
word_vector = gensim.models.KeyedVectors.load(path_w2v + name_w2v)

# Load processed data
print("Loading Data ...", flush=True)
path_pr = "Processed_Data/"
data_train = np.load(path_pr + 'data_train_pr_nf.npy', allow_pickle=True)
labels = np.load(path_pr + 'labels_train_nf.npy', allow_pickle=True)
data_test = np.load(path_pr + 'data_test_pr.npy', allow_pickle=True)

# To train without the full set
n_train = -1

if n_train > 0:
    lem_data = data_train[:n_train]
    labels = labels[:n_train]

print("Removing word not in vocab", flush=True)
# Remove words that are not in vocabulary
t_rm = time.time()

# Take all the words in the vocabulary
gensim_words = set(word_vector.vocab)
print("Number of words in vocab", len(gensim_words))

# Take all the words in dataset
all_words = set([word for tweet in data_train for word in tweet])
all_words = all_words.union(set([word for tweet in data_test for word in tweet]))
print("Number of words (train and test):", len(all_words))

words_not_in_vocab = set(all_words - gensim_words)

# Remove words not in vocab using multiprocessing
with Pool(ncpu) as p:
    final_text = p.map(find_words_not_in_vocab_p, data_train)
    final_text_test = p.map(find_words_not_in_vocab_p, data_test)

print("Time to remove words:", time.time()-t_rm, "s")

print("Computing len_max_tweet", flush=True)

# Max length of tweet (after removed not in vocab words)
len_max_tweet = np.max([len(tweet) for tweet in final_text])
len_max_tweet = np.max((len_max_tweet, np.max([len(tweet) for tweet in final_text_test])))
print("len_max_tweet:", len_max_tweet)

final_text = np.asarray(final_text)
final_text_test = np.asarray(final_text_test)


n_tweet_train = final_text.shape[0]
n_batch = 10
size_per_batch = int(np.floor(n_tweet_train/n_batch))

print("Num of batches:", n_batch)

size_w2v = word_vector.vector_size
print("Start to convert word to vector", flush=True)
t_w2v = time.time()

for b in range(n_batch):
    print("BATCH #", b+1)
    if b < n_batch-1:
        final_text_b = final_text[b*size_per_batch: (b+1)*size_per_batch]
        labels_b = labels[b*size_per_batch: (b+1)*size_per_batch]
    else:
        final_text_b = final_text[b*size_per_batch:]
        labels_b = labels[b*size_per_batch:]



    # Convert words to vec using multiprocessing
    with Pool(ncpu, maxtasksperchild=5000) as p:
        x = p.map(convert_w2v_p, final_text_b)

    x = np.asarray(x)

    np.save(path_save + 'x_train_batch_' + str(b), x)
    np.save(path_save + 'labels_batch_' + str(b), labels_b)
    #np.save(path_save + 'x_train_nf_1_b', x)
    #np.save(path_save + 'labels_nf_1_b', labels_b)

    x = None


print("Time to convert words into vec (train):", time.time()-t_w2v, "s")


if test:
    with Pool(ncpu, maxtasksperchild=5000) as p:
        x_test_ai = p.map(convert_w2v_p, final_text_test)

    x_test_ai = np.asarray(x_test_ai)

    np.save(path_save + 'x_test', x_test_ai)


