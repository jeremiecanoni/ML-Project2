import numpy as np
import csv
import pandas as pd
from tensorflow.keras.preprocessing.text import text_to_word_sequence


def get_raw_data(path_g, full):
    """
    Generate raw data without punctuation

    :param path_g: Path on the Google Drive
    :param full: string, 'f' for full dataset and 'nf' for non-full dataset
    """

    if full == 'f':
        path_pos = path_g + 'data/twitter-datasets/train_pos_full.txt'
        path_neg = path_g + 'data/twitter-datasets/train_neg_full.txt'

    elif full == 'nf':
        path_pos = path_g + 'data/twitter-datasets/train_pos.txt'
        path_neg = path_g + 'data/twitter-datasets/train_neg.txt'

    else:
        raise ValueError("Not valid full, should be 'f' or 'nf'")

    path_test = path_g + 'data/twitter-datasets/test_data.txt'

    # Read all files
    data_neg = read_file(path_neg)

    data_pos = read_file(path_pos)

    data_test = read_file(path_test)

    df_neg = pd.DataFrame(data_neg)
    df_pos = pd.DataFrame(data_pos)
    df_test = pd.DataFrame(data_test, columns=['tweet'])

    df_neg = pd.DataFrame(pd.unique(df_neg[0]).T, columns=['tweet'])
    df_neg['sentiment'] = 0
    print(df_neg.shape)

    df_pos = pd.DataFrame(pd.unique(df_pos[0]).T, columns=['tweet'])
    df_pos['sentiment'] = 1
    print(df_pos.shape)

    df = pd.concat([df_neg, df_pos])
    text_data = df['tweet'].values
    text_data_test = df_test['tweet'].values

    for idx, tweet in enumerate(text_data):
        text_data[idx] = text_to_word_sequence(tweet, filters='#"$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789',
                                               lower=True)

    for idx, tweet in enumerate(text_data_test):
        text_data_test[idx] = text_to_word_sequence(tweet, filters='#"$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n123456789',
                                                    lower=True)

    labels = df['sentiment'].values

    return text_data, labels, text_data_test



def create_csv_submission(y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """

    ids = np.arange(0, y_pred.shape[0]) + 1
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1), 'Prediction':int(r2)})


def read_file(path):
    f = open(path, "r", encoding="utf-8")
    data = f.read().split('\n')
    f.close()
    # Remove last line if empty
    if data[-1] == '':
        data = data[:-1]
    return data
