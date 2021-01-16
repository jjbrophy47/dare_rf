"""
Preprocess dataset.
"""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def dataset_specific(random_state, test_size):

    columns = ['age', 'workclass', 'industry_code', 'occupation_code', 'education',
               'wage_per_hour', 'enrolled_in_edu', 'marital_status', 'major_industry_code', 'major_occupation_code',
               'race', 'hispanic_origin', 'sex', 'union_member', 'unemployment_reason',
               'employment', 'capital_gain', 'capital_loss', 'dividends', 'tax_staus',
               'prev_region', 'prev_state', 'household_stat', 'household_summary', 'weight',
               'migration_msa', 'migration_reg', 'migration_reg_move', '1year_house', 'prev_sunbelt',
               'n_persons_employer', 'parents', 'father_birth', 'mother_birth', 'self_birth',
               'citizenship', 'income', 'business', 'taxable_income', 'veterans_admin',
               'veterans_benfits', 'label']

    # retrieve dataset
    train_df = pd.read_csv('census-income.data', header=None, names=columns)
    test_df = pd.read_csv('census-income.test', header=None, names=columns)

    # remove select columns
    remove_cols = ['industry_code', 'occupation_code', 'n_persons_employer',
                   'veterans_admin', 'veterans_benfits']
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

    # categorize attributes
    label = ['label']
    numeric = ['age', 'wage_per_hour', 'capital_gain', 'capital_loss', 'dividends', 'weight']
    categorical = list(set(columns) - set(numeric) - set(label))
    print('label', label)
    print('numeric', numeric)
    print('categorical', categorical)

    return train_df, test_df, label, numeric, categorical


def main(random_state=1, test_size=0.2, n_bins=5):

    train_df, test_df, label, numeric, categorical = dataset_specific(random_state=random_state,
                                                                      test_size=test_size)

    # binarize inputs
    ct = ColumnTransformer([('kbd', KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense'), numeric),
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

    print('train.shape: {}, label sum: {}'.format(train.shape, train[:, -1].sum()))
    print('test.shape: {}, label sum: {}'.format(test.shape, test[:, -1].sum()))

    # save to numpy format
    print('saving...')
    np.save('train.npy', train)
    np.save('test.npy', test)


if __name__ == '__main__':
    main()
