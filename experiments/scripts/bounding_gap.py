"""
Analysis of the GINI splitting criterion between two attributes.
"""
import os
import sys
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
from utility import data_util


def _plot(ax, gini_gains_best, gini_gains_worst, best_ndx, worst_ndx, ltype=''):
    instances = np.arange(len(gini_gains_best))
    ax.plot(instances, gini_gains_best, label='X{}'.format(best_ndx), marker='.')
    ax.plot(instances, gini_gains_worst, label='X{}'.format(worst_ndx), marker='.')
    ax.set_title('{} lower bound to switch'.format(ltype))
    ax.set_xlabel('# instances deleted')
    ax.set_ylabel('gini gain')
    ax.legend()
    return ax


def _lower_bound(ax, r_best, r_worst, best_ndx, worst_ndx, type_counts=None, ltype=''):

    # theoretical lower bound on the number and type of instances given a starting dataset
    gini_gains_best = []
    gini_gains_worst = []
    converged = True

    while r_best['gini_gain'] >= r_worst['gini_gain']:
        r_best, r_worst, _, _, _ = _find_delta(r_best, r_worst, type_counts)
        if r_best is None or r_worst is None:
            converged = False
            print('[{}}] failed to find min # of instances to change split'.format(ltype.capitalize()))
            break
        gini_gains_best.append(r_best['gini_gain'])
        gini_gains_worst.append(r_worst['gini_gain'])

    if converged:
        _plot(ax, gini_gains_best, gini_gains_worst, best_ndx, worst_ndx, ltype='"Greedy" {}'.format(ltype))


def _random_bound(ax, X, y, attr1_ndx, attr2_ndx, r_best, r_worst, best_ndx, worst_ndx):

    # actual instances in the dataset that lead to a switch in attributes
    gini_gains_best = []
    gini_gains_worst = []
    converged = True

    random_indices = np.random.choice(np.arange(len(X)), size=X.shape[0], replace=False)
    for ndx in random_indices[:-1]:
        xa, xb, yi = X[ndx][attr1_ndx], X[ndx][attr2_ndx], y[ndx]

        if not _check_compatibility(r_best, xa, yi) or not _check_compatibility(r_worst, xb, yi):
            print('[Random] failed to find min # of instances to change split')
            converged = False
            break
        else:
            r_best, r_worst = _decrement2(r_best, xa, yi), _decrement2(r_worst, xb, yi)
            gini_gains_best.append(r_best['gini_gain'])
            gini_gains_worst.append(r_worst['gini_gain'])

        if r_worst['gini_gain'] > r_best['gini_gain']:
            break

    if converged:
        _plot(ax, gini_gains_best, gini_gains_worst, best_ndx, worst_ndx, ltype='random')


def _find_best_attributes(X, y):
    """
    Find the first and second best attributes to split on.
    """
    best_gini_gain = 1e-8
    second_best_gini_gain = 1e-8
    best_feature_ndx = None
    second_best_feature_ndx = None

    # iterate through each feature
    for i in range(X.shape[1]):

        gini_gain = _compute_gini_gain(X[:, i], y)['gini_gain']

        # keep the best attribute
        if gini_gain > best_gini_gain:
            best_gini_gain = gini_gain
            best_feature_ndx = i

        # keep the second best attribute
        elif gini_gain > second_best_gini_gain:
            second_best_gini_gain = gini_gain
            second_best_feature_ndx = i

    return best_feature_ndx, second_best_feature_ndx


def _check_compatibility(attr, x, y):
    """
    Makes sure no illegal deletions are being made.
    """

    # check attribute
    if attr['n_count'] < 2:
        result = False

    elif x == 1:
        if y == 1:
            result = attr['left_pos'] > 0
        else:
            result = attr['left_count'] - attr['left_pos'] > 0

    elif x == 0:
        if y == 1:
            result = attr['right_pos'] > 0
        else:
            result = attr['right_count'] - attr['right_pos'] > 0

    return result


def _find_delta(r_best, r_worst, type_counts=None):
    """
    Update both attributes by deleting each instance type and seeing
    which narrows the gini gap between them the most.
    """

    best_bin_str = None
    best_r_best = None
    best_r_worst = None
    best_xa = None
    best_xb = None
    best_y = None
    best_gini_gap = 1e7

    for bin_str in ['000', '001', '010', '011', '100', '101', '110', '111']:
        xa, xb, y = (int(i) for i in bin_str)
        if not _check_compatibility(r_best, xa, y) or not _check_compatibility(r_worst, xb, y):
            continue
        temp_r_best = _decrement2(r_best, xa, y)
        temp_r_worst = _decrement2(r_worst, xb, y)

        gini_gap = temp_r_best['gini_gain'] - temp_r_worst['gini_gain']
        if gini_gap < best_gini_gap:

            # empirical instances
            if type_counts is not None:
                if type_counts[bin_str] > 0:
                    best_bin_str = bin_str
                    best_r_best = temp_r_best
                    best_r_worst = temp_r_worst
                    best_gini_gap = gini_gap
                    best_xa = xa
                    best_xb = xb
                    best_y = y

            # theoretical instances
            else:
                best_bin_str = bin_str
                best_r_best = temp_r_best
                best_r_worst = temp_r_worst
                best_gini_gap = gini_gap
                best_xa = xa
                best_xb = xb
                best_y = y

    if best_r_best is not None and type_counts is not None:
        type_counts[best_bin_str] -= 1

    return best_r_best, best_r_worst, best_xa, best_xb, best_y


def _type_counts(x_a, x_b, y):
    """
    Counts the number of instance types between two attributes and a class label.
    """
    result = {'000': 0, '001': 0, '010': 0, '011': 0,
              '100': 0, '101': 0, '110': 0, '111': 0}
    for i in range(len(x_a)):
        result[str(x_a[i]) + str(x_b[i]) + str(y[i])] += 1
    return result


def _decrement2(result, x, y):
    """
    Decrements an instance from either or both branches and recomputes the gini gain.
    """

    left_pos, left_neg = 0, 0
    right_pos, right_neg = 0, 0

    if x == 1:
        if y == 1:
            left_pos += 1
        else:
            left_neg += 1
    else:
        if y == 1:
            right_pos += 1
        else:
            right_neg += 1

    result = _decrement(result, left_pos, left_neg, right_pos, right_neg)
    return result


def _decrement(result, left_pos=0, left_neg=0, right_pos=0, right_neg=0):
    """
    Decrements an instance from either or both branches and recomputes the gini gain.
    """

    result = result.copy()

    result['n_pos'] -= (left_pos + right_pos)
    result['n_count'] -= (left_pos + left_neg + right_pos + right_neg)
    result['left_pos'] -= left_pos
    result['left_count'] -= (left_pos + left_neg)
    result['right_pos'] -= right_pos
    result['right_count'] -= (right_pos + right_neg)
    result['gini_gain'] = _compute_gini_metadata(result['n_pos'], result['n_count'], result['left_pos'],
                                                 result['left_count'], result['right_pos'], result['right_count'])
    return result


def _compute_gini_metadata(n_pos, n_count, left_pos, left_count, right_pos, right_count):
    """
    Compute the metadata necessary to compute the gini gain given the node and
    bracnh counts.
    """
    # print(n_pos, n_count, left_pos, left_count, right_pos, right_count)

    n_pos_prob = n_pos / n_count
    n_pos_prob_sq = n_pos_prob ** 2
    n_neg_prob = 1 - n_pos_prob
    n_neg_prob_sq = n_neg_prob ** 2
    n_gini = 1 - n_pos_prob_sq - n_neg_prob_sq

    left_weight = left_count / n_count
    left_pos_prob = 0 if left_pos == 0 else left_pos / left_count
    left_pos_prob_sq = left_pos_prob ** 2
    left_neg_prob = 1 - left_pos_prob
    left_neg_prob_sq = left_neg_prob ** 2
    left_gini = 1 - left_pos_prob_sq - left_neg_prob_sq
    left_weighted_gini = left_weight * left_gini

    right_weight = right_count / n_count
    right_pos_prob = 0 if right_pos == 0 else right_pos / right_count
    right_pos_prob_sq = right_pos_prob ** 2
    right_neg_prob = 1 - right_pos_prob
    right_neg_prob_sq = right_neg_prob ** 2
    right_gini = 1 - right_pos_prob_sq - right_neg_prob_sq
    right_weighted_gini = right_weight * right_gini

    attr_gini = left_weighted_gini + right_weighted_gini
    gini_gain = n_gini - attr_gini

    return gini_gain


def _compute_gini_gain(x, y):
    """
    Computes the gini gain of this attribute and returns
    the metadata used in the computation.
    """
    n_pos = np.sum(y)
    n_count = len(y)

    left_indices = np.where(x == 1)[0]
    right_indices = np.where(x == 0)[0]

    left_pos = np.sum(y[left_indices])
    left_count = len(left_indices)

    right_pos = np.sum(y[right_indices])
    right_count = len(right_indices)

    gini_gain = _compute_gini_metadata(n_pos, n_count, left_pos, left_count, right_pos, right_count)

    result = {}
    result['n_pos'] = n_pos
    result['n_count'] = n_count
    result['left_pos'] = left_pos
    result['left_count'] = left_count
    result['right_pos'] = right_pos
    result['right_count'] = right_count
    result['gini_gain'] = gini_gain

    return result


def main(args):

    # create data
    if args.dataset == 'synthetic':
        start = time.time()
        np.random.seed(args.seed)
        X = np.random.randint(2, size=(args.n_samples, args.n_attributes))
        np.random.seed(args.seed)
        y = np.random.randint(2, size=args.n_samples)
        print('creation time: {:.3f}s'.format(time.time() - start))

    else:
        X, _, y, _ = data_util.get_data(args.dataset)

    print('training instances: {}, features: {}'.format(X.shape[0], X.shape[1]))

    # choose two attributes to analyze
    if args.random_attributes:
        np.random.seed(args.seed)
        attr1_ndx, attr2_ndx = np.random.choice(np.arange(X.shape[1]), size=2, replace=False)
    else:
        attr1_ndx, attr2_ndx = _find_best_attributes(X, y)

    print('1st and 2nd attribute indices: {}, {}'.format(attr1_ndx, attr2_ndx))

    # extract their respective arrays
    x_a = X[:, attr1_ndx]
    x_b = X[:, attr2_ndx]

    # compute statistics about each one
    r_a = _compute_gini_gain(x_a, y)
    r_b = _compute_gini_gain(x_b, y)

    # find which one to split on
    best_ndx = attr1_ndx if r_a['gini_gain'] >= r_b['gini_gain'] else attr2_ndx
    worst_ndx = attr2_ndx if r_a['gini_gain'] >= r_b['gini_gain'] else attr1_ndx
    r_best = r_a if r_a['gini_gain'] >= r_b['gini_gain'] else r_b
    r_worst = r_b if r_a['gini_gain'] >= r_b['gini_gain'] else r_a
    Xs = {best_ndx: x_a, worst_ndx: x_b}
    type_counts = _type_counts(Xs[best_ndx], Xs[worst_ndx], y)

    # plot results
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    # theoretical and empirical lower bounds
    _lower_bound(axs[0], r_best.copy(), r_worst.copy(), best_ndx, worst_ndx, ltype='theoretical')
    _lower_bound(axs[1], r_best.copy(), r_worst.copy(), best_ndx, worst_ndx, type_counts, ltype='empirical')

    # lower bound using random
    _random_bound(axs[2], X, y, attr1_ndx, attr2_ndx, r_best.copy(), r_worst.copy(), best_ndx, worst_ndx)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='synthetic', help='dataset to use for the experiment.')
    parser.add_argument('--n_samples', type=int, default=10, help='number of samples to generate.')
    parser.add_argument('--n_attributes', type=int, default=4, help='number of attributes to generate.')
    parser.add_argument('--random_attributes', action='store_true', default=False, help='Uses two random attributes.')
    parser.add_argument('--seed', type=int, default=423, help='seed to populate the data.')
    args = parser.parse_args()
    print(args)
    main(args)
