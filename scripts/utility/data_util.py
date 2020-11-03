"""
Data utility methods to make life easier.
"""
import os

import numpy as np


def get_data(dataset, data_dir='data', continuous=False):
    """
    Returns a train and test set from the desired dataset.
    """
    in_dir = os.path.join(data_dir, dataset)

    if continuous:
        in_dir = os.path.join(in_dir, 'continuous')

    assert os.path.exists(in_dir)

    train = np.load(os.path.join(data_dir, dataset, 'train.npy'))
    test = np.load(os.path.join(data_dir, dataset, 'test.npy'))
    assert np.all(np.unique(train) == np.array([0, 1]))
    assert np.all(np.unique(test) == np.array([0, 1]))

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, X_test, y_train, y_test
