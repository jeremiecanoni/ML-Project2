import numpy as np
import pandas as pd
import time
import nltk
import re
from symspellpy.symspellpy import SymSpell
from nltk.stem import WordNetLemmatizer
import pkg_resources
from nltk.corpus import wordnet
from multiprocessing import Pool
import os, sys
from keras.preprocessing.text import text_to_word_sequence
import string
from helpers import *

# Setting number of cpu if on a cluster
if 'NUM_THREADSPROCESSES' in os.environ:
    ncpu = os.environ['NUM_THREADSPROCESSES']
    ncpu = int(ncpu)
    print('ncpu = ', ncpu, flush=True)

# Otherwise set to default number of cpu
else:
    ncpu = 2
    print("By default, ncpu = ", ncpu)

# f for full dataset, nf for small dataset (not full)
full = 'f'

if full=='f':
    path_pos = 'data/twitter-datasets/train_pos_full.txt'
    path_neg = 'data/twitter-datasets/train_neg_full.txt'

elif full=='nf':
    path_pos = 'data/twitter-datasets/train_pos.txt'
    path_neg = 'data/twitter-datasets/train_neg.txt'

else:
    raise ValueError("Not valid full, should be 'f' or 'nf'")

path_test = 'data/twitter-datasets/test_data.txt'

# Read all files
data_neg = read_file(path_neg)

data_pos = read_file(path_pos)

data_test = read_file(path_test)

# Convert into DataFrame
df_neg = pd.DataFrame(data_neg)
df_pos = pd.DataFrame(data_pos)
df_test = pd.DataFrame(data_test, columns=['tweet'])


# Remove copies of same tweet
df_neg = pd.DataFrame(pd.unique(df_neg[0]).T, columns=['tweet'])
df_neg['sentiment'] = 0
print(df_neg.shape)

df_pos = pd.DataFrame(pd.unique(df_pos[0]).T, columns=['tweet'])
df_pos['sentiment'] = 1
print(df_pos.shape)

df = pd.concat([df_neg, df_pos])

# Load all dictionnary used for spelling correction, instantiate SymSpell object
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")

sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

# Adding some expression to dictionnary
dic_yes = 159595214
list_add_dic = ['lol', 'haha', 'tv', 'xoxo', 'lmao', 'omg', 'url', 'jk', 'rt']

for word in list_add_dic:
    sym_spell.words[word] = dic_yes

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Little list of stop word  to remove from tweet
stop_list = ['user', 'url', 'a', 'an', 'the', 'and', 'of', 'at', 'by']


def correct_spelling_p(tweet):
    """
    Return tweet with punctuation removed, spelling errors corrected and some expression replaced by a known expression
    :param tweet: string containing the whole tweet
    :return: Partially processed tweet
    """
    # Replace some known expression by one in the dictionnary
    # PUT SPACE BEFORE AND AFTER
    to_rpl = [' im ', '<3', ' u ', ' ur ', ' youre ', ' wanna ']
    rpl_with = [" i'm ", ' love ', ' you ', ' your ', " you're ", ' want to ']
    for old, new in zip(to_rpl, rpl_with):
        tweet = re.sub(old, new, tweet)

    tweet = ' '.join(text_to_word_sequence(tweet, filters='!#"$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789', lower=True))

    # Replace multiple spaces by one space
    tweet = re.sub(' +', ' ', tweet)

    # Correct the spelling errors
    result = sym_spell.lookup_compound(tweet, max_edit_distance=2)
    return result[0].term


def stop_clean(tweet):
    """
    Reomve from tweet word present in the stop lists defined above
    :param tweet: string containing tweet
    :return: modified tweet
    """
    # Remove the stop words of the tweet
    tweet = ' '.join(word for word in tweet.split() if word not in stop_list)
    return tweet


def lemmatize_tweet(tweet):
    """
    Lemmatize a tweet using nltk wordnet lemmatizer
    :param tweet: string containing tweet
    :return: tweet with lemmatized words
    """
    tweet = tweet.split()
    for idx, word in enumerate(tweet):
        tweet[idx] = lemmatizer.lemmatize(word, get_wordnet_pos(word))
    tweet = ' '.join(tweet)
    return tweet


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def cleaning(tweet):
    """
    Apply the full processing to a tweet sequentially
    :param tweet: string containing tweet
    :return: processed tweet
    """
    tweet = correct_spelling_p(tweet)
    tweet = stop_clean(tweet)
    tweet = lemmatize_tweet(tweet)
    tweet = re.sub("'", "", tweet)
    return tweet


def cleaning_df(df):
    """

    :param df: Dataframe containing tweets
    :return: Dataframe containing processed tweet
    """
    return df['tweet'].apply(cleaning)


def parallelize_dataframe(df, func, n_cores=2):
    """
    Performs processing of tweets on a dataframe using multiprocessing
    :param df: Dataframe containing all the tweets
    :param func: cleaning_df function
    :param n_cores: number of core on which multiprocessing is launched
    :return: dataframe with processed tweets
    """
    # Split dataframe to be able to launch multiprocessing pool on n_cores
    df_split = np.array_split(df, n_cores)

    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


t1 = time.time()
print("Start cleaning train:")
df_processed_train = parallelize_dataframe(df, cleaning_df, n_cores=ncpu)

print("Start cleaning test:")
df_processed_test = parallelize_dataframe(df_test, cleaning_df, n_cores=ncpu)

print(time.time()-t1)

# Convert to array of list of words
data_train_pr = np.asarray([text_to_word_sequence(text, filters='') for text in df_processed_train])
data_test_pr = np.asarray([text_to_word_sequence(text, filters='') for text in df_processed_test])

# Save file
path_save = "Processed_Data/"
np.save(path_save + "labels_train_" + full + "_sl5", df['sentiment'].values)
np.save(path_save + "data_train_pr_" + full + "_sl5", data_train_pr)
np.save(path_save + "data_test_pr" + "_sl5", data_test_pr)

df_processed_train.to_csv(path_save + "data_train_pr_" + full + "_sl4.txt")
df_processed_test.to_csv(path_save + "data_test_pr_" + full + "_sl4.txt")





