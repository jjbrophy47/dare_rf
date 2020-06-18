"""
Tests the CeDAR tree implementation.
"""
import os
import sys
import time
import argparse
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

import cedar
from experiments.utility import data_util


def main(args):

    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, data_dir=args.data_dir)

    start = time.time()
    model = cedar.tree(epsilon=args.epsilon,
                       lmbda=args.lmbda,
                       criterion=args.criterion,
                       max_depth=args.max_depth,
                       random_state=args.rs,
                       cedar_type=args.cedar_type)
    model = model.fit(X_train, y_train)
    print('\n[CeDAR] build time: {:.7f}s'.format(time.time() - start))

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    print('auc: {:.3f}, acc: {:.3f}'.format(auc, acc))

    if not args.add:
        np.random.seed(args.rs)
        delete_indices = np.random.choice(X_train.shape[0], size=args.n_update, replace=False)
        print('\ndeleting {} instances'.format(args.n_update))

        if args.batch:
            t1 = time.time()
            model.delete(delete_indices)
            print('delete time: {:.7f}s'.format(time.time() - t1))

        else:
            for i in range(len(delete_indices)):
                t1 = time.time()
                model.delete(delete_indices[i])
                print('delete time: {:.7f}s'.format(time.time() - t1))

        print(model.get_removal_statistics())
        proba = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)
        print('auc: {:.3f}, acc: {:.3f}'.format(auc, acc))

    else:
        np.random.seed(args.rs)
        add_indices = np.random.choice(X_train.shape[0], size=args.n_update, replace=False)
        X_add, y_add = X_train[add_indices], y_train[add_indices]

        if args.batch:
            t1 = time.time()
            model.add(X_add, y_add)
            print('add time: {:.7f}s'.format(time.time() - t1))

        else:
            for i in range(X_add.shape[0]):
                t1 = time.time()
                model.add(X_add[[i]], y_add[[i]])
                print('add time: {:.7f}s'.format(time.time() - t1))

        print(model.get_add_statistics())
        proba = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)
        print('auc: {:.3f}, acc: {:.3f}'.format(auc, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # experiment settings
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', default='spect', help='dataset to use for the experiment.')
    parser.add_argument('--rs', type=int, default=1, help='seed to enhance reproducibility.')

    # model hyperparameters
    parser.add_argument('--cedar_type', default='1', help='dataset to use for the experiment.')
    parser.add_argument('--epsilon', type=float, default=1.0, help='setting for certified adversarial ordering.')
    parser.add_argument('--lmbda', type=float, default=0.1, help='noise hyperparameter.')
    parser.add_argument('--max_depth', type=int, default=3, help='maximum depth of the tree.')
    parser.add_argument('--criterion', type=str, default='gini', help='splitting criterion.')

    # update settings
    parser.add_argument('--add', action='store_true', default=False, help='add instances.')
    parser.add_argument('--batch', action='store_true', default=False, help='update in batches.')
    parser.add_argument('--n_update', type=int, default=10, help='no. instances to add/delete.')

    args = parser.parse_args()
    main(args)
