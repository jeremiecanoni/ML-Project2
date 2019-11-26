import numpy as np
import pandas as pd
import time
import nltk
from nltk.corpus import stopwords
import re
from symspellpy.symspellpy import SymSpell
from nltk.stem import WordNetLemmatizer
import pkg_resources
from nltk.corpus import wordnet
from multiprocessing import Pool
import os, sys
from keras.preprocessing.text import text_to_word_sequence


if 'NUM_THREADSPROCESSES' in os.environ:
    ncpu = os.environ['NUM_THREADSPROCESSES']
    ncpu = int(ncpu)
    print('ncpu = ', ncpu, flush=True)

else:
    ncpu = 2
    print("By default, ncpu = ", ncpu)

full = 'nf'

if full=='f':
    path_pos = 'data/twitter-datasets/train_pos_full.txt'
    path_neg = 'data/twitter-datasets/train_neg_full.txt'

elif full=='nf':
    path_pos = 'data/twitter-datasets/train_pos.txt'
    path_neg = 'data/twitter-datasets/train_neg.txt'

else:
    raise "Not valid full, should be 'f' or 'nf'"

path_test = 'data/twitter-datasets/test_data.txt'

# Read all files
f = open(path_neg, "r")
data_neg = f.read().split('\n')
f.close()

f = open(path_pos, "r")
data_pos = f.read().split('\n')
f = f.close()


f = open(path_test)
data_test = f.read().split('\n')
f.close()

df_neg = pd.DataFrame(data_neg)
df_pos = pd.DataFrame(data_pos)
df_test = pd.DataFrame(data_test, columns=['tweet'])

# Remove last empty row
df_neg.drop(df_neg.tail(1).index, inplace=True)
df_pos.drop(df_pos.tail(1).index, inplace=True)
df_test.drop(df_test.tail(1).index, inplace=True)


df_neg = pd.DataFrame(pd.unique(df_neg[0]).T, columns=['tweet'])
df_neg['sentiment'] = 0
print(df_neg.shape)

df_pos = pd.DataFrame(pd.unique(df_pos[0]).T, columns=['tweet'])
df_pos['sentiment'] = 1
print(df_pos.shape)


df = pd.concat([df_neg, df_pos])

df = df[0:500]
df_test = df_test[0:500]


sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

lemmatizer = WordNetLemmatizer()

stop_words = list(stopwords.words('english'))
stop_words.remove('not') #Remove not from the list
stop_words.append('u') #Add u to the list
stop_words.append('ur') #Add 'ur' to the list
stop_words.append("i'm")
stop_words.append("im")
stop_words.append('user')
stop_words.append('urls')

stop_list = stop_words


def correct_spelling_p(tweet):
    # lookup suggestions for multi-word input strings (supports compound
    # splitting & merging)
    # max edit distance per lookup (per single word, not per whole input string)
    
    tweet = re.sub(r'\d+|\\','',tweet) #Remove any numbers, '\' from the tweets
    # Replace multiple spaces by one space
    tweet = re.sub(' +',' ', tweet)
    result = sym_spell.lookup_compound(tweet, max_edit_distance=2)
    return result[0].term


def stop_clean(tweet):
    tweet = tweet.lower() #To remove
    tweet = re.sub(r"\w+n't\s?",'not ',tweet) #Replace all word finishing by n't by not
    to_not = ['havent','doesnt','cant','dont','shouldnt','arent','couldnt',"didnt","hadnt","mightnt","mustnt","neednt","wasnt","wont","wouldnt"]
    for word in to_not:
        tweet = re.sub(r'\b' + word + r'\b', 'not', tweet)
    tweet = ' '.join(word for word in tweet.split() if word not in stop_list) #Remove the stop words of the tweet
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
    tweet = correct_spelling_p(tweet)
    tweet = stop_clean(tweet)
    if tweet != "": #If the tweet is an empty string, lemmatizing throws an error
        tweet = lemmatizer.lemmatize(tweet, get_wordnet_pos(tweet))
    return tweet


def cleaning_df(df):
    return df['tweet'].apply(cleaning)


def parallelize_dataframe(df, func, n_cores=2):
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

# Convert to list of list of words
data_train_pr = np.asarray([text_to_word_sequence(text, filters='') for text in df_processed_train])
data_test_pr = np.asarray([text_to_word_sequence(text, filters='') for text in df_processed_test])


path_save = "Processed_Data/"
np.save(path_save + "labels_train_" + full, df['sentiment'].values)
np.save(path_save + "data_train_pr_" + full, data_train_pr)
np.save(path_save + "data_test_pr", data_test_pr)


