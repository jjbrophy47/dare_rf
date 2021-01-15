import os
import sys
import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston

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
        print(data)
        X = data['data']
        y = data['target']

        # make into binary classification dataset
        y = np.where(y < np.mean(y), 0, 1)

    return X, y


def main(args):

    X, y = load_data(args.dataset)

    print(X)

    # train decision tree
    model = dart.Tree(topd=0, k=5, max_depth=3, random_state=1)
    model = model.fit(X, y)

    # predict
    print(X[0])
    print(model.predict_proba(X[[0]]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to use for the experiment.')
    args = parser.parse_args()
    main(args)
