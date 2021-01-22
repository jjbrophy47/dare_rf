import os
import sys
import time
import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.metrics import roc_auc_score

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

    X_train, X_test, y_train, y_test = load_data(args.dataset, args.data_dir)

    print(X_train.shape)

    # settings
    topd = 20
    k = 100
    max_depth = 20
    seed = 1
    n_delete = 20

    # train decision tree
    model = dart.Tree(topd=topd, k=k, max_depth=max_depth, random_state=seed)
    model = model.fit(X_train, y_train)

    # predict
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print('AUC: {:.3f}'.format(auc))

    # delete training data
    if args.delete and not args.simulate:
        delete_indices = np.random.default_rng(seed=seed).choice(X_train.shape[0], size=n_delete, replace=False)
        print('instances to delete: {}'.format(delete_indices))

        for delete_ndx in delete_indices:
            print('\ninstance to delete, {}'.format(delete_ndx))
            model.delete(delete_ndx)

        types, depths = model.get_removal_types_depths()
        print('types: {}'.format(types))
        print('depths: {}'.format(depths))

    # simulate the deletion of each instance
    elif args.delete and args.simulate:
        delete_indices = np.random.default_rng(seed=seed).choice(X_train.shape[0], size=n_delete, replace=False)
        print('instances to delete: {}'.format(delete_indices))

        # cumulative time
        cum_delete_time = 0
        cum_sim_time = 0

        # simulate and delete each sample
        for delete_ndx in delete_indices:

            # simulate the deletion
            start = time.time()
            n_samples_to_retrain = model.sim_delete(delete_ndx)
            sim_time = time.time() - start
            cum_sim_time += sim_time
            print('\nsimulated instance, {}: {:.3f}s, no. samples: {:,}'.format(
                  delete_ndx, sim_time, n_samples_to_retrain))

            # delete
            start = time.time()
            model.delete(delete_ndx)
            delete_time = time.time() - start
            cum_delete_time += delete_time
            print('deleted instance, {}: {:.3f}s'.format(delete_ndx, delete_time))

        types, depths = model.get_removal_types_depths()
        print('types: {}'.format(types))
        print('depths: {}'.format(depths))

        avg_sim_time = cum_sim_time / len(delete_indices)
        avg_delete_time = cum_delete_time / len(delete_indices)

        print('avg. sim. time: {:.5f}s'.format(avg_sim_time))
        print('avg. delete time: {:.5f}s'.format(avg_delete_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', type=str, default='iris', help='dataset to use for the experiment.')
    parser.add_argument('--delete', action='store_true', help='whether to deletion or not.')
    parser.add_argument('--simulate', action='store_true', help='whether to simulate deletions or not.')
    args = parser.parse_args()
    main(args)
