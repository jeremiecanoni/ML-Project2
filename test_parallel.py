from __future__ import print_function
import numpy as np
from helpers import *
import pandas as pd
import tensorflow as tf
import time
from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
import gensim
from multiprocessing import Pool
import re
from keras.preprocessing import sequence

import pkg_resources
from symspellpy.symspellpy import SymSpell
from models import *

ncpu = 2
print("ncpu = ", ncpu)


def correct_spelling_p(tweet):
    # lookup suggestions for multi-word input strings (supports compound
    # splitting & merging)
    # max edit distance per lookup (per single word, not per whole input string)
    # Remove numbers from strings
    for i in range(10):
        tweet = tweet.replace(str(i), '')

    # Replace multiple spaces by one space
    tweet = re.sub(' +',' ', tweet)
    result = sym_spell.lookup_compound(tweet, max_edit_distance=2)
    return result[0].term


def correct_spelling_hashtag(word_dataset):
    sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt")
    # term_index is the column of the term and count_index is the
    # column of the term frequency
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    for id_tweet, tweet in enumerate(word_dataset):
        for id_word, word in enumerate(tweet):
            if word.startswith("#"):
                result = sym_spell.word_segmentation(word)
                parsed_hashtag = text_to_word_sequence(result.corrected_string,
                                                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n01234556789\'')

                word_dataset[id_tweet][id_word:id_word] = parsed_hashtag


def open_file(path):
    f = open(path, "r", encoding="utf-8")
    data = f.read().split('\n')
    f.close()
    return data


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def remove_not_unique_tweet_training(data_pos, data_neg):
    # Load the data into a pandas dataframe
    df_neg = pd.DataFrame(data_neg)
    df_pos = pd.DataFrame(data_pos)

    # Discard all tweets appearing twice in the set.
    df_neg = pd.DataFrame(pd.unique(df_neg[0]).T, columns=['tweet'])
    df_neg['sentiment'] = 0

    df_pos = pd.DataFrame(pd.unique(df_pos[0]).T, columns=['tweet'])
    df_pos['sentiment'] = 1

    df = pd.concat([df_neg, df_pos])

    # convert data into a list of tweets
    out_data = df['tweet'].astype(str).values.tolist()

    labels = np.asarray(df['sentiment'].astype(int).values.tolist())

    return out_data, labels


def remove_not_unique_tweet_test(data):
    # Load the data into a pandas dataframe
    df = pd.DataFrame(data)

    # Discard all tweets appearing twice in the set.
    df = pd.DataFrame(pd.unique(df[0]).T, columns=['tweet'])

    # convert data into a list of tweets
    out_data = df['tweet'].astype(str).values.tolist()

    return out_data


def find_words_not_in_vocab(model_gs, words_dataset):
    # Take all the words in the vocabulary
    gensim_words = set(model_gs.wv.vocab)
    # Take all the words in dataset
    all_words = set([word for tweet in words_dataset for word in tweet])

    # Subtract both datasets to get the words in the vocab
    words_not_in_vocab = list(all_words - gensim_words)

    for idx, tweet in enumerate(words_dataset):
        words_dataset[idx] = [w for w in tweet if w not in words_not_in_vocab]


def lemmatize_p(tweet):
    # filtered_data[idx] = [w for w in tweet if not w in stop_words]
    lem_data= [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tweet]

    return lem_data


n = 5000
n_test = 100
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
test_data = test_data[:n_test]


data, labels = np.asarray(data), np.asarray(labels)
test_data = np.asarray(test_data)

perm_tot = np.random.permutation(labels.shape[0])
data = data[perm_tot]
labels = labels[perm_tot]

data_tmp = data[:n]

# Correct spelling
print("Start spelling correction")
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

t_s = time.time()


with Pool(ncpu) as p:
    data_tmp = p.map(correct_spelling_p, data_tmp)
    test_data = p.map(correct_spelling_p, test_data)

print("Duration:", time.time() - t_s, "s")


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

print("Start Lemmatization")
t_s = time.time()

lemmatizer = WordNetLemmatizer()

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

print("Start training Word2Vec")

# Define gensim model
size_w2v = 150

lem_data_tot = np.concatenate((lem_data, lem_data_test), axis=0)
model_gs = gensim.models.Word2Vec(lem_data_tot, size=size_w2v, window=6, min_count=5, iter=10, workers=ncpu)

find_words_not_in_vocab(model_gs, lem_data)
find_words_not_in_vocab(model_gs, lem_data_test)

# Convert words to vectors
x = convert_w2v(model_gs, lem_data, size_w2v, len_max_tweet)
x_test_real = convert_w2v(model_gs, lem_data_test, size_w2v, len_max_tweet)


# Define neural network parameters
filters, kernel_size, batch_size = 250, 5, 32
epochs, hidden_dims = 10, 250

model = build_model(filters, kernel_size, hidden_dims)

# np stand for not padded
x_train, y_train, x_test, y_test = split_data(x, labels[:n], ratio=1)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs)

y_pred = np.ndarray.flatten(model.predict_classes(x_test_real, batch_size=32))
create_csv_submission(y_pred, 'csv_test1.csv')
