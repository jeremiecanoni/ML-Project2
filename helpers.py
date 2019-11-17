import numpy as np
import csv


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing

    """
    np.random.seed(seed)
    n = len(x)
    perm = np.random.permutation(n)

    # Split the data based on the given ratio
    x, y = x[perm], y[perm]

    x_train, y_train = x[:int(np.floor(ratio * n))], y[:int(np.floor(ratio * n))]
    x_test, y_test = x[int(np.ceil(ratio * n)):], y[int(np.ceil(ratio * n)):]

    return x_train, y_train, x_test, y_test


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


def open_file(path):
    f = open(path, "r", encoding="utf-8")
    data = f.read().split('\n')
    f.close()
    # Remove last line if empty
    if data[-1] == '':
        data = data[:-1]
    return data
