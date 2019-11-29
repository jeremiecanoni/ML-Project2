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


stop_words_2 = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'of', 'at', 'by', 'then', 'once', 'here', 'there', 's', 't', 'can', 'don', "don't", 'should', "should've", 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'u', 'ur', "i'm", 'im', 'user', 'urls']
not_remove = ['when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'what', 'which', 'who', 'whom', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'just', 'now']

stop_words_3 = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'of', 'at', 'by', 'then', 'once', 'here', 'there', 's', 't', 'd', 'll', 'm', 'o', 're', 've', 'y', 'u', 'ur', "i'm", 'im', 'user', 'urls']


# Choose which stop list to use
stop_list = stop_words_3


def correct_spelling_p(tweet):
    # lookup suggestions for multi-word input strings (supports compound
    # splitting & merging)
    # max edit distance per lookup (per single word, not per whole input string)
    
    tweet = re.sub(r'\d+|\\', '', tweet) #Remove any numbers, '\' from the tweets
    # Replace multiple spaces by one space
    tweet = re.sub(' +', ' ', tweet)
    result = sym_spell.lookup_compound(tweet, max_edit_distance=2)
    return result[0].term


def stop_clean(tweet):
    tweet = tweet.lower() #To remove
    """tweet = re.sub(r"\w+n't\s?", 'not ', tweet) #Replace all word finishing by n't by not
    to_not = ['havent', 'doesnt', 'cant', 'dont', 'shouldnt', 'arent', 'couldnt', "didnt", "hadnt", "mightnt",
              "mustnt", "neednt", "wasnt", "wont", "wouldnt", 'neednt', 'isnt', 'werent']
    for word in to_not:
        tweet = re.sub(r'\b' + word + r'\b', 'not', tweet)
    """
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
np.save(path_save + "labels_train_" + full + "_sl3", df['sentiment'].values)
np.save(path_save + "data_train_pr_" + full + "_sl3", data_train_pr)
np.save(path_save + "data_test_pr" + "_sl3", data_test_pr)


