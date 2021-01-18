"""
Preprocess dataset.
"""
import os
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


def dataset_specific(random_state, test_size,
                     n_instances, max_feature_vals,
                     logger):

    # retrieve dataset
    df = pd.read_csv('train', nrows=n_instances)
    logger.info('{}'.format(df))

    # crete month, day, and hour columns
    # df['month'] = df['hour'].apply(lambda x: int(str(x)[2:4]))
    # df['day'] = df['hour'].apply(lambda x: int(str(x)[4:6]))
    # df['hour'] = df['hour'].apply(lambda x: int(str(x)[6:8]))

    # check data type and no. unique values for each column
    # for c in df.columns:
    #     logger.info('{}, {}, {}'.format(c, df[c].dtype, len(df[c].unique())))

    # remove select columns
    logger.info('removing columns...')
    remove_cols = ['id', 'site_id', 'site_domain', 'app_id',
                   'device_id', 'device_ip', 'device_model']

    # remove columns that have too many feature values
    for c in df.columns:
        n_feature_vals = len(df[c].unique())
        if n_feature_vals > max_feature_vals:
            remove_cols.append(c)

    if len(remove_cols) > 0:
        df = df.drop(columns=remove_cols)

    # remove nan rows
    nan_rows = df[df.isnull().any(axis=1)]
    df = df.dropna()
    logger.info('nan rows: {}'.format(len(nan_rows)))

    # split into train and test
    indices = np.arange(len(df))
    n_train_samples = int(len(indices) * (1 - test_size))

    np.random.seed(random_state)
    train_indices = np.random.choice(indices, size=n_train_samples, replace=False)
    test_indices = np.setdiff1d(indices, train_indices)

    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    # categorize attributes
    columns = list(df.columns)
    label = ['click']
    numeric = ['month', 'day', 'hour']
    categorical = list(set(columns) - set(numeric) - set(label))
    logger.info('label: {}'.format(label))
    logger.info('numeric: {}'.format(numeric))
    logger.info('categorical: {}'.format(categorical))

    return train_df, test_df, label, numeric, categorical


def main(random_state=1, test_size=0.2, n_instances=20000000,
         max_feature_vals=250, out_dir='continuous'):

    logger = get_logger('log.txt')

    train_df, test_df, label, numeric, categorical = dataset_specific(random_state=random_state,
                                                                      test_size=test_size,
                                                                      n_instances=n_instances,
                                                                      max_feature_vals=max_feature_vals,
                                                                      logger=logger)

    # binarize inputs
    ct = ColumnTransformer([('kbd', 'passthrough', numeric),
                            ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical)])
    train = ct.fit_transform(train_df)
    test = ct.transform(test_df)

    # binarize outputs
    le = LabelEncoder()
    train_label = le.fit_transform(train_df[label].to_numpy().ravel()).reshape(-1, 1)
    test_label = le.transform(test_df[label].to_numpy().ravel()).reshape(-1, 1)

    # combine binarized data
    train = np.hstack([train, train_label]).astype(np.int32)
    test = np.hstack([test, test_label]).astype(np.int32)

    logger.info('train.shape: {}, label sum: {}'.format(train.shape, train[:, -1].sum()))
    logger.info('test.shape: {}, label sum: {}'.format(test.shape, test[:, -1].sum()))

    # save to numpy format
    logger.info('saving...')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'train.npy'), train)
    np.save(os.path.join(out_dir, 'test.npy'), test)


if __name__ == '__main__':
    main()
