import numpy as np


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