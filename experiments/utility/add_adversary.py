"""
Addition adversary that attacks the root, choosing
samples from a given set of samples.
"""
import numpy as np


def constrained_add_adversary(X_train, y_train, X_add, y_add, seed=None, verbose=0, logger=None):
    """
    Given a dataset with labels, find the ordering that causes the most
    retrainings at the root node; brute-force greedy method.
    """

    # start out with a being the better feature
    a_is_better = True
    retrains = 0
    ordering = np.empty(X_add.shape[0])

    # find best two attributes
    ndx_a, ndx_b, meta_a, meta_b = _find_best_attributes(X_train, y_train)
    if logger and verbose > 1:
        logger.info('1st: x{}, 2nd: x{}'.format(ndx_a, ndx_b))

    # get instance counts based on the two attributes and the label
    counts, indices = _type_counts(X_add[:, ndx_a], X_add[:, ndx_b], y_add)
    if logger and verbose > 1:
        logger.info(counts)

    for i in range(X_add.shape[0]):

        # brute force check which instance type reduces the gini index gap the most
        meta_a, meta_b, bin_str, index_gap = _find_instance(meta_a, meta_b, counts, a_is_better)
        if logger and verbose > 1:
            logger.info('{}, {}, {}'.format(i, bin_str, index_gap))

        # attributes have switched position!
        if index_gap < 0:
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


def _find_best_attributes(X, y):
    """
    Find the first and second best attributes to split on.
    """
    gi_a, gi_b = 1, 1
    ndx_a, ndx_b = None, None
    meta_a, meta_b = None, None

    # iterate through each feature
    for i in range(X.shape[1]):

        meta = _get_gini_index(X[:, i], y)
        gini_index = meta['gini_index']

        # keep the best attribute
        if gini_index < gi_a:
            gi_b = gi_a
            ndx_b = ndx_a
            meta_b = meta_a

            gi_a = gini_index
            ndx_a = i
            meta_a = meta

        # keep the second best attribute
        elif gini_index < gi_b:
            gi_b = gini_index
            ndx_b = i
            meta_b = meta

    return ndx_a, ndx_b, meta_a, meta_b


def _get_gini_index(x, y):
    """
    Computes the gini index of this attribute and returns
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

    gini_index = _compute_gini_index(n_pos, n_count, left_pos, left_count, right_pos, right_count)

    # save the metadata for this attribute
    result = {}
    result['n_pos'] = n_pos
    result['n_count'] = n_count
    result['left_pos'] = left_pos
    result['left_count'] = left_count
    result['right_pos'] = right_pos
    result['right_count'] = right_count
    result['gini_index'] = gini_index

    return result


def _compute_gini_index(n_pos, n_count, left_pos, left_count, right_pos, right_count):

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


def _type_counts(xa, xb, y):
    """
    Counts the number of instance types between two attributes and a class label.
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


def _find_instance(meta_a, meta_b, counts, a_is_better):
    """
    Find which instance type narrows the gini index gap the most after deletion
    using a brute-force method.
    """

    best_bin_str = None
    best_a, best_b = None, None
    best_index_gap = 1e7

    for bin_str in ['000', '001', '010', '011', '100', '101', '110', '111']:
        v1, v2, v3 = (int(i) for i in bin_str)

        temp_a = _increment(meta_a, v1, v3)
        temp_b = _increment(meta_b, v2, v3)

        if a_is_better:
            index_gap = temp_b['gini_index'] - temp_a['gini_index']
        else:
            index_gap = temp_a['gini_index'] - temp_b['gini_index']

        if index_gap < best_index_gap:

            # empirical instances
            if counts[bin_str] > 0:
                best_bin_str = bin_str
                best_a, best_b = temp_a, temp_b
                best_index_gap = index_gap

    return best_a, best_b, best_bin_str, best_index_gap


def _increment(meta, x, y):
    """
    Increments an instance to either or both branches and recomputes the gini gain.
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

    return _recompute(meta, left_pos, left_neg, right_pos, right_neg)


def _recompute(meta, left_pos=0, left_neg=0, right_pos=0, right_neg=0):
    """
    Increments an instance to either or both branches and recomputes the gini gain.
    """

    meta = meta.copy()

    meta['n_pos'] += (left_pos + right_pos)
    meta['n_count'] += (left_pos + left_neg + right_pos + right_neg)
    meta['left_pos'] += left_pos
    meta['left_count'] += (left_pos + left_neg)
    meta['right_pos'] += right_pos
    meta['right_count'] += (right_pos + right_neg)
    meta['gini_index'] = _compute_gini_index(meta['n_pos'], meta['n_count'], meta['left_pos'],
                                             meta['left_count'], meta['right_pos'], meta['right_count'])
    return meta
