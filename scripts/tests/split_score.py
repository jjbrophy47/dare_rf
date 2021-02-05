"""
Computes split score for each threshold of each feature,
and plots a distribution of the top k thresholds for each feature.
"""
import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')
sys.path.insert(0, here + '/../')
from utility import data_util
from utility import print_util


class Threshold:
    def __init__(self, v, n, n_pos, n_left, n_left_pos):
        self.v = v
        self.n = n
        self.n_pos = n_pos
        self.n_left = n_left
        self.n_left_pos = n_left_pos

def get_thresholds(x_arr, y_arr):
    """
    Find all candidate threshold values and return a list of Threshold objects.
    """

    # sort values and labels
    indices = np.argsort(x_arr)
    x = x_arr[indices]
    y = y_arr[indices]

    # get unique values
    vals = np.unique(x)

    # compute node statistics
    n = len(x)
    n_pos = np.sum(y)

    # return variable
    thresholds = []

    # find each valid threshold between every adjacent pair of attribute values
    for i in tqdm(range(1, vals.shape[0])):
        v1 = vals[i-1]
        v2 = vals[i]

        v1_indices = np.where(x == v1)
        v2_indices = np.where(x == v2)

        v1_ratio = np.sum(y[v1_indices]) / len(v1_indices)
        v2_ratio = np.sum(y[v2_indices]) / len(v2_indices)

        valid = (v1_ratio != v2_ratio) or (v1_ratio > 0 and v2_ratio < 1.0)

        if valid:
            left_indices = np.where(x <= v1)[0]
            n_left = len(left_indices)
            n_left_pos = np.sum(y[left_indices])
            T = Threshold(v=v1, n=n, n_pos=n_pos, n_left=n_left, n_left_pos=n_left_pos)
            thresholds.append(T)

    return thresholds

def compute_scores(C):
    """
    Compute split criterion for each valid threshold.
    """
    results = []

    # compute score for each threshold
    for T in tqdm(C):
        score = compute_gini_index(T)
        results.append((T, score))

    return results


def compute_gini_index(T):
    """
    Compute Gini index for this threshold.
    """

    # get statistics to compute Gini index
    n = T.n
    n_pos = T.n_pos
    n_left = T.n_left
    n_left_pos = T.n_left_pos
    n_right = n - n_left
    n_right_pos = n_pos - n_left_pos

    if n_left > 0:
        weight = n_left / n
        pos_prob = n_left_pos / n_left
        neg_prob = 1 - pos_prob
        index = 1 - (pos_prob * pos_prob) - (neg_prob * neg_prob)
        left_weighted_index = weight * index

    if n_right > 0:
        weight = n_right / n
        pos_prob = n_right_pos / n_right
        neg_prob = 1 - pos_prob
        index = 1 - (pos_prob * pos_prob) - (neg_prob * neg_prob)
        right_weighted_index = weight * index

    return left_weighted_index + right_weighted_index


def main(args):

    # create output directory
    out_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    # create logger
    logger_fp = os.path.join(out_dir, 'log.txt')
    logger = print_util.get_logger(logger_fp)
    logger.info('{}'.format(args))
    logger.info('\ntimestamp: {}'.format(datetime.now()))

    # get dataset
    X_train, X_test, y_train, y_test = data_util.get_data(args.dataset, args.data_dir)
    logger.info('X_train.shape: {}'.format(X_train.shape))

    # collect top threshold scores
    top_scores = []

    # get best threshold(s) for each feature
    for i in range(X_train.shape[1]):
        vals = np.unique(X_train[:, i])
        C = get_thresholds(X_train[:, i], y_train)
        S = compute_scores(C)
        logger.info('\n[FEATURE {}] no. unique: {:,}, no. valid thresholds: {:,}'.format(i, len(vals), len(C)))

        # sort thresholds based on score
        S = sorted(S, key=lambda x: x[1])

        # display split score for each threshold
        for T, s in S[:args.k]:
            logger.info('  threshold value: {:.5f}, score: {:.5f}'.format(T.v, s))
            top_scores.append(s)

    # plot distribution of top threshold scores
    ax = sns.distplot(top_scores, rug=True, hist=False)
    ax.set_title('{}: Scores for Top {} Threshold(s) / Feature'.format(args.dataset.title(), args.k))
    ax.set_xlabel('Gini index')
    ax.set_ylabel('Density')
    plt.savefig(os.path.join(out_dir, 'k_{}.pdf'.format(args.k)), bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--out_dir', type=str, default='output/split_score/', help='output directory.')
    parser.add_argument('--dataset', type=str, default='surgical', help='dataset to use for the experiment.')
    parser.add_argument('--k', type=int, default=1, help='no. top thresholds to analyze.')
    args = parser.parse_args()
    main(args)
