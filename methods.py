import nltk
from nltk.corpus import wordnet
import gensim
import re
from keras.preprocessing import sequence
import cython
import pkg_resources
import pandas as pd
import numpy as np
from symspellpy.symspellpy import SymSpell
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import text_to_word_sequence


# Load appropriate things
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

lemmatizer = WordNetLemmatizer()

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


def find_words_not_in_vocab(model_wv, words_dataset):
    # Take all the words in the vocabulary
    gensim_words = set(model_wv.vocab)
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
