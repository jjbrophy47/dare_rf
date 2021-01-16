import os
import sys
import time
import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
import dart
from utility import data_util

def load_data(dataset, data_dir):

    if dataset == 'iris':
        data = load_iris()
        X = data['data']
        y = data['target']

        # make into binary classification dataset
        indices = np.where(y != 2)[0]
        X = X[indices]
        y = y[indices]

        X_train, X_test, y_train, y_test = X, X, y, y

    elif dataset == 'boston':
        data = load_boston()
        X = data['data']
        y = data['target']

        # make into binary classification dataset
        y = np.where(y < np.mean(y), 0, 1)

        X_train, X_test, y_train, y_test = X, X, y, y

    else:
        X_train, X_test, y_train, y_test = data_util.get_data(dataset, data_dir)

        X_train = X_train[:,:50]
        X_test = X_test[:,:50]

    return X_train, X_test, y_train, y_test


def main(args):

    # get data
    X_train, X_test, y_train, y_test = load_data(args.dataset, args.data_dir)
    print(X_train, X_train.shape)

    # train
    n_estimators = 100
    max_depth = 10
    random_state = 1

    if args.model == 'dart':
        model = dart.Forest(topd=0, k=5, n_estimators=n_estimators,
                            max_depth=max_depth, random_state=random_state)

    elif args.model == 'sklearn':
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=random_state)

    start = time.time()
    model = model.fit(X_train, y_train)
    end = time.time() - start

    print('train time: {:.3f}s'.format(end))

    # predict
    y_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_proba, axis=1)

    # evaluate
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    print('ACC: {:.3f}, AUC: {:.3f}'.format(acc, auc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to use for the experiment.')
    parser.add_argument('--model', type=str, default='dart', help='dart or sklearn')
    args = parser.parse_args()
    main(args)
