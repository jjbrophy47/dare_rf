"""
Preprocess dataset.
"""
import os
import sys
import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def get_logger(filename=''):
    """
    Return a logger object to easily save textual output.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    log_handler = logging.FileHandler(filename, mode='w')
    formatter = logging.Formatter('%(message)s')

    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(log_handler)

    return logger


def main(random_state=1, test_size=0.2, n_instances=1000000, out_dir='continuous'):

    # create logger
    logger = get_logger('log.txt')

    # columns to use
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    # data dtypes for each column
    dtypes = {c: np.float32 for c in cols}
    dtypes[0] = np.uint8

    # retrieve dataset
    logger.info('reading in dataset...')
    df = pd.read_csv('day_0', sep='\t', header=None, usecols=cols, dtype=dtypes, nrows=n_instances)
    logger.info('{}'.format(df))

    # get numpy array
    X = df.values
    df = None

    # impute missing values with the mean
    logger.info('imputing missing values with the mean...')
    assert np.isnan(X[:, 0]).sum() == 0
    col_mean = np.nanmean(X, axis=0)
    nan_indices = np.where(np.isnan(X))
    X[nan_indices] = np.take(col_mean, nan_indices[1])

    # move the label column in X to the last column
    logger.info('moving label column to the last column...')
    y = X[:, 0].copy()
    np.delete(X, 0, 1)
    X = np.hstack([X, y])

    # split into train and test
    logger.info('splitting into train and test sets...')
    indices = np.arange(X.shape[0])
    n_train_samples = int(len(indices) * (1 - test_size))

    np.random.seed(random_state)
    train_indices = np.random.choice(indices, size=n_train_samples, replace=False)
    test_indices = np.setdiff1d(indices, train_indices)

    train = X[train_indices]
    test = X[test_indices]

    logger.info('train.shape: {}, label sum: {}'.format(train.shape, train[:, -1].sum()))
    logger.info('test.shape: {}, label sum: {}'.format(test.shape, test[:, -1].sum()))

    # save to numpy format
    logger.info('saving...')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'train.npy'), train)
    np.save(os.path.join(out_dir, 'test.npy'), test)


if __name__ == '__main__':
    main()
