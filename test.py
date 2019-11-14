from __future__ import print_function
import numpy as np
from helpers import *
import pandas as pd
# import textblob as tb
import tensorflow as tf
import time
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
import gensim

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D

import pkg_resources
from symspellpy.symspellpy import SymSpell

from models import *


def correct_spelling(word_dataset):
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt")
    bigram_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
    # term_index is the column of the term and count_index is the
    # column of the term frequency
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
    # lookup suggestions for multi-word input strings (supports compound
    # splitting & merging)
    for idx, tweet in enumerate(word_dataset):
        # max edit distance per lookup (per single word, not per whole input string)
        result = sym_spell.lookup_compound(tweet, max_edit_distance=2)

        word_dataset[idx] = result[0].term


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



def lemmatize(data):
    lem_data = [None] * len(data)
    lemmatizer = WordNetLemmatizer()

    for idx, tweet in enumerate(data):
        # filtered_data[idx] = [w for w in tweet if not w in stop_words]
        lem_data[idx] = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tweet]

    return np.asarray(lem_data)


n = 10000

path_neg = 'data/twitter-datasets/train_neg.txt'
path_pos = 'data/twitter-datasets/train_pos.txt'

data_neg = open_file(path_neg)
data_pos = open_file(path_pos)

t1 = time.time()

data, labels = remove_not_unique_tweet_training(data_pos, data_neg)

data, labels = np.asarray(data), np.asarray(labels)

perm_tot = np.random.permutation(labels.shape[0])
data = data[perm_tot]
labels = labels[perm_tot]

data_tmp = data[:n]

# Correct spelling
correct_spelling(data_tmp)

data = [tweet.replace('\'', '') for tweet in data_tmp]
data = [text_to_word_sequence(text, filters='!"$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n01234556789\'') for text in data]

data = np.asarray(data)

stop_words = list(stopwords.words('english'))
stop_words.append('u')
stop_words.append('ur')

lem_data = lemmatize(data[:n])

# Max length of tweet
len_max_tweet = np.max([len(tweet) for tweet in lem_data])

print("Time for preprocessing: ", time.time() - t1, "s")

print("Start training Word2Vec")

# Define gensim model
size_w2v = 100
model_gs = gensim.models.Word2Vec(lem_data, size=size_w2v, window=6, min_count=5, iter=10, workers=1)

find_words_not_in_vocab(model_gs, lem_data)

# Convert words to vectors
x = convert_w2v(model_gs, lem_data, size_w2v, len_max_tweet)

# Pad tweets
for idx_t, tweet in enumerate(lem_data):
    if len(tweet)==0: tweet = ['empty']
    vec = model_gs.wv[tweet]
    x[idx_t, :, :] = np.transpose(sequence.pad_sequences(vec.T, maxlen=len_max_tweet, dtype=np.float32))


# Define neural network parameters
filters, kernel_size, batch_size = 250, 5, 32
epochs, hidden_dims = 20, 250

model = build_model(filters, kernel_size, hidden_dims)

# np stand for not padded
x_train, y_train, x_test, y_test = split_data(x, labels[:n], ratio=0.8)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
