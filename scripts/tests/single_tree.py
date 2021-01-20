import os
import sys
import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.metrics import roc_auc_score

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
import dart

def load_data(dataset):

    if dataset == 'iris':
        data = load_iris()
        X = data['data']
        y = data['target']

        # make into binary classification dataset
        indices = np.where(y != 2)[0]
        X = X[indices]
        y = y[indices]

    elif dataset == 'boston':
        data = load_boston()
        X = data['data']
        y = data['target']

        # make into binary classification dataset
        y = np.where(y < np.mean(y), 0, 1)

    return X, y


def main(args):

    X, y = load_data(args.dataset)

    print(X.shape)

    # train decision tree
    seed = 1
    model = dart.Tree(topd=0, k=10, max_depth=20, random_state=seed)
    model = model.fit(X, y)

    # predict
    y_proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_proba)
    print('AUC: {:.3f}'.format(auc))

    # delete training data
    if args.delete:
        delete_indices = np.random.default_rng(seed=seed).choice(X.shape[0], size=10, replace=False)
        print('instances to delete: {}'.format(delete_indices))

        for delete_ndx in delete_indices:
            print('\ninstance to delete, {}'.format(delete_ndx))
            model.delete(delete_ndx)

        types, depths = model.get_removal_types_depths()
        print('types: {}'.format(types))
        print('depths: {}'.format(depths))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to use for the experiment.')
    parser.add_argument('--delete', action='store_true', help='whether to deletion or not.')
    args = parser.parse_args()
    main(args)
