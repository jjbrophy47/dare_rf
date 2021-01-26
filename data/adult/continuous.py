"""
Preprocesses dataset but keep continuous variables.
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def dataset_specific(random_state, test_size):
    """
    Put dataset specific processing here.
    """

    # categorize attributes
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country', 'label']

    # retrieve dataset
    train_df = pd.read_csv('adult.data', header=None, names=columns)
    test_df = pd.read_csv('adult.test', header=None, names=columns)

    # remove select columns
    remove_cols = ['education-num']
    if len(remove_cols) > 0:
        train_df = train_df.drop(columns=remove_cols)
        test_df = test_df.drop(columns=remove_cols)
        columns = [x for x in columns if x not in remove_cols]

    # remove nan rows
    train_nan_rows = train_df[train_df.isnull().any(axis=1)]
    test_nan_rows = test_df[test_df.isnull().any(axis=1)]
    print('train nan rows: {}'.format(len(train_nan_rows)))
    print('test nan rows: {}'.format(len(test_nan_rows)))
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # fix label columns
    test_df['label'] = test_df['label'].apply(lambda x: x.replace('.', ''))

    # categorize attributes
    label = ['label']
    numeric = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical = list(set(columns) - set(numeric) - set(label))

    return train_df, test_df, label, numeric, categorical


def main(random_state=1, test_size=0.2, out_dir='continuous'):

    train_df, test_df, label, numeric, categorical = dataset_specific(random_state=random_state,
                                                                      test_size=test_size)

    # encode categorical inputs
    ct = ColumnTransformer([('kbd', 'passthrough', numeric),
                            ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical)])
    train = ct.fit_transform(train_df)
    test = ct.transform(test_df)

    # binarize outputs
    le = LabelEncoder()
    train_label = le.fit_transform(train_df[label].to_numpy().ravel()).reshape(-1, 1)
    test_label = le.transform(test_df[label].to_numpy().ravel()).reshape(-1, 1)

    # add labels
    train = np.hstack([train, train_label]).astype(np.float32)
    test = np.hstack([test, test_label]).astype(np.float32)

    print('\ntrain:\n{}, dtype: {}'.format(train, train.dtype))
    print('train.shape: {}, label sum: {}'.format(train.shape, train[:, -1].sum()))

    print('\ntest:\n{}, dtype: {}'.format(test, test.dtype))
    print('test.shape: {}, label sum: {}'.format(test.shape, test[:, -1].sum()))

    # save to numpy format
    print('saving...')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'train.npy'), train)
    np.save(os.path.join(out_dir, 'test.npy'), test)


if __name__ == '__main__':
    main()
