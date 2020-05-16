"""
Adversarial utilities.
"""
import math

import numpy as np


def order_samples(X, y, n_samples=None, criterion='gini',
                  seed=None, verbose=0, logger=None):
    """
    Given a dataset with labels, find the ordering that
    causes the most retrainings at the root node when
    deleting samples; brute-force greedy method.
    """

    # start out with a being the better feature
    a_is_better = True
    retrains = 0
    ordering = np.empty(n_samples)

    # return only a fraction of the training data
    if n_samples is not None:
        assert n_samples <= len(X)
    else:
        n_samples = len(X)

    # find best two attributes
    ndx_a, ndx_b, meta_a, meta_b = _find_best_attributes(X, y, criterion)
    if logger and verbose > 1:
        logger.info('1st: x{}, 2nd: x{}'.format(ndx_a, ndx_b))

    # get instance counts based on the two attributes and the label
    counts, indices = _type_counts(X[:, ndx_a], X[:, ndx_b], y)
    if logger and verbose > 1:
        logger.info(counts)

    for i in range(n_samples):

        # brute force find instance type that reduces the score gap the most
        meta_a, meta_b, bin_str, score_gap = _find_instance(meta_a, meta_b,
                                                            counts, a_is_better,
                                                            criterion)
        if logger and verbose > 1:
            logger.info('{}, {}, {}'.format(i, bin_str, score_gap))

        # attributes have switched position!
        if score_gap < 0:
            a_is_better = not a_is_better
            retrains += 1

        # put that instance in the ordering
        np.random.seed(seed)
        ndx = np.random.choice(indices[bin_str])
        ordering[i] = ndx

        # remove that instance from the counts and indices
        counts[bin_str] -= 1
        indices[bin_str] = indices[bin_str][indices[bin_str] != ndx]

    if logger:
        logger.info('estimated retrains: {:,}'.format(retrains))
    ordering = ordering.astype(np.int32)
    return ordering


def _find_best_attributes(X, y, criterion):
    """
    Find the first and second best attributes to split on.
    """
    si_a, si_b = 2, 2
    ndx_a, ndx_b = None, None
    meta_a, meta_b = None, None

    # iterate through each feature
    for i in range(X.shape[1]):

        meta = _get_score(X[:, i], y, criterion)
        score = meta['score']

        # keep the best attribute
        if score < si_a:
            si_b = si_a
            ndx_b = ndx_a
            meta_b = meta_a

            si_a = score
            ndx_a = i
            meta_a = meta

        # keep the second best attribute
        elif score < si_b:
            si_b = score
            ndx_b = i
            meta_b = meta

    return ndx_a, ndx_b, meta_a, meta_b


def _get_score(x, y, criterion):
    """
    Computes the split score of this attribute and returns
    the metadata used in the computation.
    """

    # compute counts from this attribute
    n_pos = np.sum(y)
    n_count = len(y)

    left_indices = np.where(x == 1)[0]
    right_indices = np.setdiff1d(np.arange(len(x)), left_indices)

    left_pos = np.sum(y[left_indices])
    left_count = len(left_indices)

    right_pos = np.sum(y[right_indices])
    right_count = len(right_indices)

    if criterion == 'gini':
        score = _compute_gini_index(n_pos, n_count, left_pos, left_count, right_pos, right_count)

    elif criterion == 'entropy':
        score = _compute_entropy(n_pos, n_count, left_pos, left_count, right_pos, right_count)

    else:
        raise ValueError('criterion unknown: {}'.format(criterion))

    # save the metadata for this attribute
    result = {}
    result['n_pos'] = n_pos
    result['n_count'] = n_count
    result['left_pos'] = left_pos
    result['left_count'] = left_count
    result['right_pos'] = right_pos
    result['right_count'] = right_count
    result['score'] = score

    return result


def _compute_gini_index(n_pos, n_count, left_pos, left_count,
                        right_pos, right_count):
    """
    Computes the Gini Index given the appropriate statistics.
    """

    # use counts to compute gini index
    left_weight = left_count / n_count
    left_pos_prob = 0 if left_pos == 0 else left_pos / left_count
    left_neg_prob = 1 - left_pos_prob
    left_gini = 1 - (left_pos_prob ** 2) - (left_neg_prob ** 2)

    right_weight = right_count / n_count
    right_pos_prob = 0 if right_pos == 0 else right_pos / right_count
    right_neg_prob = 1 - right_pos_prob
    right_gini = 1 - (right_pos_prob ** 2) - (right_neg_prob ** 2)

    return left_weight * left_gini + right_weight * right_gini


def _compute_entropy(n_pos, n_count, left_pos, left_count,
                     right_pos, right_count):
    """
    Computes the conditional entropy given the appropriate statistics.
    """

    left_weighted_entropy = 0
    right_weighted_entropy = 0

    if left_count > 0:
        weight = left_count / n_count
        pos_prob = left_pos / left_count
        neg_prob = 1 - pos_prob

        entropy = 0
        if pos_prob > 0:
            entropy -= pos_prob * math.log(pos_prob, 2)
        if neg_prob > 0:
            entropy -= neg_prob * math.log(neg_prob, 2)

        left_weighted_entropy = weight * entropy

    if right_count > 0:
        weight = right_count / n_count
        pos_prob = right_pos / right_count
        neg_prob = 1 - pos_prob

        entropy = 0
        if pos_prob > 0:
            entropy -= pos_prob * math.log(pos_prob, 2)
        if neg_prob > 0:
            entropy -= neg_prob * math.log(neg_prob, 2)

        right_weighted_entropy = weight * entropy

    return left_weighted_entropy + right_weighted_entropy


def _type_counts(xa, xb, y):
    """
    Counts the number of instance types between two
    attributes and a class label.
    """
    counts = {'000': 0, '001': 0, '010': 0, '011': 0,
              '100': 0, '101': 0, '110': 0, '111': 0}
    indices = {'000': 0, '001': 0, '010': 0, '011': 0,
               '100': 0, '101': 0, '110': 0, '111': 0}

    for i in range(len(xa)):
        counts[str(xa[i]) + str(xb[i]) + str(y[i])] += 1

    for k in indices.keys():
        v1, v2, v3 = int(k[0]), int(k[1]), int(k[2])
        indices[k] = np.where((xa == v1) & (xb == v2) & (y == v3))[0]

    return counts, indices


def _find_instance(meta_a, meta_b, counts, a_is_better, criterion):
    """
    Find which instance type narrows the score gap
    the most after deletion using a brute-force method.
    """

    best_bin_str = None
    best_a, best_b = None, None
    best_score_gap = 1e7

    bin_strs = ['000', '001', '010', '011', '100', '101', '110', '111']
    for bin_str in bin_strs:
        v1, v2, v3 = (int(i) for i in bin_str)

        if not _check(meta_a, v1, v3) or not _check(meta_b, v2, v3):
            continue

        temp_a = _decrement(meta_a, v1, v3, criterion)
        temp_b = _decrement(meta_b, v2, v3, criterion)

        if a_is_better:
            score_gap = temp_b['score'] - temp_a['score']
        else:
            score_gap = temp_a['score'] - temp_b['score']

        if score_gap < best_score_gap:

            # empirical instances
            if counts[bin_str] > 0:
                best_bin_str = bin_str
                best_a, best_b = temp_a, temp_b
                best_score_gap = score_gap

    return best_a, best_b, best_bin_str, best_score_gap


def _check(meta, x, y):
    """
    Makes sure no illegal deletions are being made.
    """

    # check attribute
    if meta['n_count'] < 2:
        result = False

    elif x == 1:
        if y == 1:
            result = meta['left_pos'] > 0
        else:
            result = meta['left_count'] - meta['left_pos'] > 0

    elif x == 0:
        if y == 1:
            result = meta['right_pos'] > 0
        else:
            result = meta['right_count'] - meta['right_pos'] > 0

    return result


def _decrement(meta, x, y, criterion):
    """
    Decrements an instance from either or both branches
    and recomputes the split score.
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

    score = _recompute(meta, criterion, left_pos, left_neg,
                       right_pos, right_neg)
    return score


def _recompute(meta, criterion, left_pos=0, left_neg=0, right_pos=0, right_neg=0):
    """
    Decrements an instance from either or both branches and recomputes
    the split score.
    """

    meta = meta.copy()

    meta['n_pos'] -= (left_pos + right_pos)
    meta['n_count'] -= (left_pos + left_neg + right_pos + right_neg)
    meta['left_pos'] -= left_pos
    meta['left_count'] -= (left_pos + left_neg)
    meta['right_pos'] -= right_pos
    meta['right_count'] -= (right_pos + right_neg)

    if criterion == 'gini':
        meta['score'] = _compute_gini_index(meta['n_pos'],
                                            meta['n_count'],
                                            meta['left_pos'],
                                            meta['left_count'],
                                            meta['right_pos'],
                                            meta['right_count'])
    elif criterion == 'entropy':
        meta['score'] = _compute_entropy(meta['n_pos'],
                                         meta['n_count'],
                                         meta['left_pos'],
                                         meta['left_count'],
                                         meta['right_pos'],
                                         meta['right_count'])
    else:
        raise ValueError('unknown criterion: {}'.format(criterion))

    return meta
