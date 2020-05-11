"""
Utility methods for displaying data.
"""
import os
import sys
import logging

import numpy as np
from scipy.stats import sem


def get_logger(filename=''):
    """
    Return a logger object to easily save textual output.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    log_handler = logging.FileHandler(filename, mode='w')
    formatter = logging.Formatter('%(message)s')

    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(log_handler)

    return logger


def remove_logger(logger):
    """
    Remove handlers from logger.
    """
    logger.handlers = []


def get_results(args, dataset, adversary, method):
    """
    Get results for a single method.
    """
    r = {}
    for rs in args.rs:
        fp = os.path.join(args.in_dir, dataset,
                          args.model_type, args.criterion,
                          adversary,
                          'rs{}'.format(rs),
                          '{}.npy'.format(method))
        r[rs] = np.load(fp, allow_pickle=True)[()]
    return r


def get_max_features(args, r, name):
    """
    Get mean values.
    """
    max_features = r[args.rs[0]][name]
    max_features = 'sqrt' if not max_features else max_features
    return max_features


def get_mean(args, r, name):
    """
    Get mean values.
    """
    res_list = []
    min_n_vals = 1e10

    for rs in args.rs:
        vals = r[rs][name]
        min_n_vals = min(min_n_vals, len(vals))
        res_list.append(vals)

    # only keep the minimum number of values
    res_list = [x[:min_n_vals] for x in res_list]

    res = np.vstack(res_list)
    res_mean = res.mean(axis=0)
    res_sem = sem(res, axis=0)
    return res_mean, res_sem


def get_mean1d(args, r, name, as_int=False):
    """
    Get mean values.
    """
    res_list = []
    for rs in args.rs:
        res_list.append(r[rs][name])
    res_arr = np.array(res_list)
    res_mean = res_arr.mean()
    res_sem = sem(res_arr)

    if as_int:
        res_mean = int(res_mean)
        if not np.isnan(res_sem):
            res_sem = int(res_sem)

    return res_mean, res_sem


def get_mean_retrainings(args, r, delete=True):
    """
    Get mean number of retrainings.
    """
    res_list = []

    for rs in args.rs:
        types = r[rs]['type']
        if delete:
            n_retrains = len(np.where(types >= 2)[0])
        res_list.append(n_retrains)

    res_arr = np.array(res_list)
    res_mean = res_arr.mean()
    res_sem = sem(res_arr)
    return int(res_mean), int(res_sem)


def get_mean_retrain_depth(args, r):
    """
    Get mean number of retrainings.
    """
    res_list = []
    for rs in args.rs:
        types = r[rs]['type']
        depths = r[rs]['depth']
        retrain_indices = np.where(types >= 2)[0]

        if len(retrain_indices) > 0:
            retrain_depths = depths[retrain_indices]
            res_list.append(retrain_depths.mean())

    if len(res_list) == 0:
        res_mean = -1
        res_sem = -1

    else:
        res_arr = np.array(res_list)
        res_mean = res_arr.mean()
        res_sem = sem(res_arr) if len(res_list) > 1 else -1

    return int(res_mean), int(res_sem)


def get_mean_amortize(args, r):
    """
    Get mean amortized runtime.
    """
    res_list = []
    for rs in args.rs:
        amortize = r[rs]['time'].mean()
        res_list.append(amortize)
    res_arr = np.array(res_list)
    res_mean = res_arr.mean()
    res_sem = sem(res_arr)
    return res_mean, res_sem


def get_mean_completions(args, r, n_trees):
    """
    Get mean number of successful deletions.
    """
    res_list = []
    for rs in args.rs:
        n_deletions = r[rs]['type'].shape[0]
        res_list.append(n_deletions)

    res_arr = np.array(res_list)
    res_mean = res_arr.mean() / n_trees
    res_sem = sem(res_arr)
    return int(res_mean), int(res_sem)
