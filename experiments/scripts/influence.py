"""
This experiment compares the time it takes to compute the effect
of each training instance on a test instance for different methods.
BABC: Binary Attributes Binary Classification.
"""
import os
import sys
import time
import argparse

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
from mulan.trees.babc_tree_d import BABC_Tree_D
from utility import data_util


def _refit_method(tree_original, X_train, y_train, x_test, y_test, indices, max_depth=4):

    yhat_original = tree_original.predict_proba(x_test)[:, 1][0]

    result = {'influence': [], 'refit_time': []}
    for i, ndx in enumerate(indices):

        if (i + 1) % 100 == 0:
            print(i)

        X_new, y_new = np.delete(X_train, ndx, axis=0), np.delete(y_train, ndx)

        start = time.time()
        tree = BABC_Tree_D(max_depth=max_depth).fit(X_new, y_new)
        result['refit_time'].append(time.time() - start)

        influence = tree.predict_proba(x_test)[:, 1][0] - yhat_original
        if y_test == 0:
            influence *= -1.0
        result['influence'].append(influence)

    result['refit_time_sum'] = np.sum(result['refit_time'])
    return result


def _delete_method(tree_original, X_train, y_train, x_test, y_test, indices, max_depth=4):

    yhat_original = tree_original.predict_proba(x_test)[:, 1][0]

    result = {'influence': [], 'copy_time': [], 'delete_time': [], 'delete_type': []}
    for i, ndx in enumerate(indices):

        if (i + 1) % 1000 == 0:
            print(i)

        start = time.time()
        tree = tree_original.copy()
        result['copy_time'].append(time.time() - start)

        start = time.time()
        result['delete_type'].append(tree.delete(int(ndx)))
        result['delete_time'].append(time.time() - start)

        influence = tree.predict_proba(x_test)[:, 1][0] - yhat_original
        if y_test == 0:
            influence *= -1.0
        result['influence'].append(influence)

    result['copy_time_sum'] = np.sum(result['copy_time'])
    result['delete_time_sum'] = np.sum(result['delete_time'])
    return result


def _fit_delete_refit(X, y, delete_ndx, refit=False, max_depth=4):
    """
    This method first a tree, efficiently deletes the target instance,
    and refits a new tree without the target instance, and
    returns the times for each of these events.
    """
    result = {}

    t1 = BABC_Tree_D(max_depth=args.max_depth).fit(X, y)
    start = time.time()
    result['delete_type'] = t1.delete(delete_ndx)
    result['delete'] = time.time() - start

    if refit:
        X_new = np.delete(X, delete_ndx, axis=0)
        y_new = np.delete(y, delete_ndx)

        start = time.time()
        t2 = BABC_Tree_D(max_depth=max_depth).fit(X_new, y_new)
        result['refit'] = time.time() - start
        result['refit_to_delete_ratio'] = result['refit'] / result['delete']
        assert t1.equals(t2)

    return result


def _display_results(results, args, n_samples, n_attributes, n_remove, out_dir='output/bbac/influence'):
    """
    Plot the average time of deletion for each deletion type.
    """

    # display total time to compute influence on target training samples
    total_times = []
    methods = []

    if not args.no_refit:
        refit_time = results['refit']['total_time']
        total_times.append(refit_time)
        methods.append('refit')

    delete_time = results['delete']['total_time']
    total_times.append(delete_time)
    methods.append('delete')

    print('\ntotal times:')
    if not args.no_refit:
        print('refit total time: {:.3f}'.format(refit_time))
    print('delete total time: {:.3f}'.format(delete_time))

    f = plt.figure(figsize=(14, 4))
    indices = np.arange(len(total_times))
    ax0 = f.add_subplot(131)
    ax0.bar(indices, total_times)
    ax0.set_xticks(indices)
    ax0.set_xticklabels(methods)
    ax0.set_ylabel('cumulative time (s)')
    ax0.set_title('total time')

    # breakdown of delete times
    copy_time_sum = results['delete']['copy_time_sum']
    delete_time_sum = results['delete']['delete_time_sum']
    prediction_time_sum = delete_time - copy_time_sum - delete_time_sum
    names = ['copy', 'delete', 'prediction']

    print('\ndelete time breakdown:')
    print('copy time sum: {:.3f}'.format(copy_time_sum))
    print('delete time sum: {:.3f}'.format(delete_time_sum))
    print('prediction time sum: {:.3f}'.format(prediction_time_sum))

    indices = np.arange(3)
    ax1 = f.add_subplot(132)
    ax1.bar(indices, [copy_time_sum, delete_time_sum, prediction_time_sum])
    ax1.set_xticks(indices)
    ax1.set_xticklabels(names)
    ax1.set_ylabel('cumulative time (s)')
    ax1.set_title('delete times')

    # breakdown of delete types
    ax2 = f.add_subplot(133)
    ax2.set_title('delete types')
    sns.countplot(x=results['delete']['delete_type'], ax=ax2)

    title_str = 'Influence - dataset: {}, samples: {}, attributes: {}, removed: {}\n'
    f.suptitle(title_str.format(args.dataset, n_samples, n_attributes, n_remove))
    f.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_dir = os.path.join(out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 's{}_a{}_r{}.pdf'.format(n_samples, n_attributes, n_remove)),
                bbox_inches='tight')


def main(args):

    # obtain dataset
    if args.dataset == 'synthetic':

        np.random.seed(args.seed)
        X = np.random.randint(2, size=(args.n_samples, args.n_attributes))

        np.random.seed(args.seed)
        y = np.random.randint(2, size=args.n_samples)

        X_train, y_train = X[:-1], y[:-1]
        x_test, y_test = X[[-1]], y[-1]

    else:
        X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, convert=False)
        x_test, y_test = X_test[[-1]], y_test[-1]

    # retrieve the indices to remove
    n_samples = X_train.shape[0]
    n_attributes = X_train.shape[1]
    n_remove = int(args.remove_frac * n_samples) if args.n_remove is None else args.n_remove
    np.random.seed(args.seed)
    indices_to_delete = np.random.choice(np.arange(n_samples), size=n_remove, replace=False)
    print('n_samples: {}, n_attributes: {}'.format(n_samples, n_attributes))

    tree_original = BABC_Tree_D(max_depth=args.max_depth).fit(X_train, y_train)

    result = {}
    if not args.no_refit:
        print('refit method...')
        start = time.time()
        result['refit'] = _refit_method(tree_original, X_train, y_train, x_test, y_test,
                                        indices_to_delete, max_depth=args.max_depth)
        result['refit']['total_time'] = time.time() - start

    print('delete method...')
    start = time.time()
    result['delete'] = _delete_method(tree_original, X_train, y_train, x_test, y_test,
                                      indices_to_delete, max_depth=args.max_depth)
    result['delete']['total_time'] = time.time() - start

    # make sure influences match
    if not args.no_refit:
        assert np.all(result['refit']['influence'] == result['delete']['influence'])

    # save results
    _display_results(result, args, n_samples, n_attributes, n_remove)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')
    parser.add_argument('--n_samples', type=int, default=10, help='number of samples to generate.')
    parser.add_argument('--n_attributes', type=int, default=4, help='number of attributes to generate.')
    parser.add_argument('--remove_frac', type=float, default=0.1, help='fraction of instances to delete.')
    parser.add_argument('--n_remove', type=int, default=None, help='number of instances to delete.')
    parser.add_argument('--seed', type=int, default=423, help='seed to populate the data.')
    parser.add_argument('--max_depth', type=int, default=4, help='maximum depth of the tree.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    parser.add_argument('--no_refit', action='store_true', default=False, help='do not record refits.')
    args = parser.parse_args()
    print(args)
    main(args)
