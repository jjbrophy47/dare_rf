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

    # train
    k = 100
    n_estimators = 10
    max_depth = 20
    seed = 1
    n_delete = 269

    if args.model == 'dart':
        model = dart.Forest(topd=0, k=k, n_estimators=n_estimators,
                            max_depth=max_depth, random_state=seed)

    elif args.model == 'sklearn':
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=seed)

    start = time.time()
    model = model.fit(X_train, y_train)
    train_time = time.time() - start

    print('train time: {:.3f}s'.format(train_time))

    # predict
    y_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_proba, axis=1)

    # evaluate
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    print('ACC: {:.3f}, AUC: {:.3f}'.format(acc, auc))

    # delete training data
    cum_delete_time = 0
    if args.delete and not args.simulate:
        delete_indices = np.random.default_rng(seed=seed).choice(X_train.shape[0], size=n_delete, replace=False)
        print('instances to delete: {}'.format(delete_indices))

        for delete_ndx in delete_indices:
            start = time.time()
            model.delete(delete_ndx)
            delete_time = time.time() - start
            cum_delete_time += delete_time
            print('\ndeleted instance, {}: {:.3f}s'.format(delete_ndx, delete_time))

        types, depths = model.get_removal_types_depths()
        print('types: {}'.format(types))
        print('depths: {}'.format(depths))

        avg_delete_time = cum_delete_time / len(delete_indices)
        print('train time: {:.3f}s'.format(train_time))
        print('avg. delete time: {:.3f}s'.format(avg_delete_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to use for the experiment.')
    parser.add_argument('--model', type=str, default='dart', help='dart or sklearn')
    parser.add_argument('--delete', action='store_true', help='whether to deletion or not.')
    parser.add_argument('--simulate', action='store_true', help='whether to simulate deletions or not.')
    args = parser.parse_args()
    main(args)
