"""
Utility methods to make life easier.
"""
import os

import numpy as np


def get_data(dataset, seed, data_dir='data', n_samples=100,
             n_attributes=4, test_frac=0.2):
    """
    Returns a train and test set from the desired dataset.
    """
    if dataset == 'synthetic':
        return _create_synthetic_data(seed, n_samples=n_samples,
                                      n_attributes=n_attributes,
                                      test_frac=test_frac)
    return _load_data(dataset, data_dir=data_dir)


def _create_synthetic_data(seed, n_samples=100, n_attributes=4,
                           test_frac=0.2):
    np.random.seed(seed)
    X = np.random.randint(2, size=(n_samples, n_attributes))

    np.random.seed(seed)
    y = np.random.randint(2, size=n_samples)

    n_train = X.shape[0] - int(X.shape[0] * test_frac)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, X_test, y_train, y_test


def _load_data(dataset, data_dir='data'):
    """
    Load the binary dataset.
    """
    assert os.path.exists(os.path.join(data_dir, dataset))

    train = np.load(os.path.join(data_dir, dataset, 'train.npy'))
    test = np.load(os.path.join(data_dir, dataset, 'test.npy'))
    assert np.all(np.unique(train) == np.array([0, 1]))
    assert np.all(np.unique(test) == np.array([0, 1]))

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, X_test, y_train, y_test
