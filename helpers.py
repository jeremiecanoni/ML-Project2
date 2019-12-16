import numpy as np
import csv


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
