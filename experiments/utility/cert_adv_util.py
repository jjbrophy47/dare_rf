"""
Adversarial utilities against a certified model.
"""
import numpy as np


# TODO: make this work for addition too
def certified_adversary(X, y, epsilon, lmbda, gamma, n_samples=None, seed=None, verbose=0, logger=None):
    """
    Given a dataset with labels, find the ordering that causes the most
    retrainings at the root node; brute-force greedy method.
    """

    # results
    retrains = 0
    ordering = np.empty(n_samples)

    # store data as dicts
    X, y = _numpy_to_dict(X, y)

    # return only a fraction of the training data
    if n_samples is not None:
        assert n_samples <= len(X)
    else:
        n_samples = len(X)

    # generate probability distribution
    metas, gini_indices = _get_gini_indices(X, y)
    p, pk = _probability_distribution(metas, lmbda=lmbda, gamma=gamma)

    # pick a random attribute to target
    np.random.seed(seed)
    ndx = np.random.choice(list(pk.keys()))
    counts, indices = _type_counts(X, y, ndx)
    if logger and verbose > 0:
        logger.info('chosen: x{}, gini index: {:.3f}'.format(ndx, gini_indices[ndx]))
        logger.info(counts)

    # iteratively find the ordering that causes the most retrains
    for i in range(n_samples):

        # brute force check which instance type creates the biggest change in distribution
        result = _find_instance(metas, ndx, counts, lmbda, gamma)

        # hanging branch, retrain
        if type(result) == tuple:
            bin_str, metas = result
            metas = _update_metas(metas, bin_str)
            new_p, new_pk = _probability_distribution(metas, lmbda=lmbda, gamma=gamma)
            retrains += 1
            p = new_p
            if logger and verbose > 0:
                logger.info('hanging branch, retrain')

        # delete instance and recheck
        else:
            bin_str = result
            metas = _update_metas(metas, bin_str)
            new_p, new_pk = _probability_distribution(metas, lmbda=lmbda, gamma=gamma)

            # check if retraning is necessary
            ratio = new_p / p
            if np.any(ratio > np.exp(epsilon)) or np.any(ratio < np.exp(-epsilon)):
                retrains += 1
                p = new_p
                diff_ndx = np.where(ratio > np.exp(epsilon))[0]
                if logger and verbose > 0:
                    logger.info('differing distributions, retrain, {}, {}'.format(diff_ndx, ratio[diff_ndx]))

            # show progress
            else:
                if logger and verbose > 0:
                    logger.info('{}, {}, {}, {}'.format(i, bin_str, metas[ndx]['gini_index'], ratio[ndx]))

        # put that instance in the ordering
        np.random.seed(seed)
        delete_ndx = np.random.choice(indices[bin_str])
        ordering[i] = delete_ndx

        # remove that instance from the counts, indices, and data
        counts[bin_str] -= 1
        indices[bin_str] = indices[bin_str][indices[bin_str] != delete_ndx]
        del X[delete_ndx]
        del y[delete_ndx]

        # choose a new attribute to focus on
        if len(new_p) != len(p):
            np.random.seed(seed)
            ndx = np.random.choice(list(new_pk.keys()))
            counts, indices = _type_counts(X, y, ndx)
            if logger and verbose > 0:
                logger.info('new attribute: x{}'.format(ndx))

    if logger and verbose > 0:
        logger.info('retrains: {}'.format(retrains))
    ordering = ordering.astype(np.int64)
    ordering = ordering[:i]
    return ordering


def _find_instance(metas, ndx, counts, lmbda, gamma):
    """
    Find which instance raises/lowers the probability gap the most.
    """

    best_bin_str = None
    best_ratio = 0

    p, _ = _probability_distribution(metas, lmbda=lmbda, gamma=gamma)
    for bin_str in ['00', '01', '10', '11']:
        v1, v2 = (int(i) for i in bin_str)

        invalid = _check_metas(metas, v1, v2)
        if len(invalid) > 0 and counts[bin_str] > 0:
            for i in invalid:
                del metas[i]
            return bin_str, metas

        temp_metas = _update_metas(metas, bin_str)
        temp_p, temp_pk = _probability_distribution(temp_metas, lmbda=lmbda, gamma=gamma)
        ratio = temp_p / p

        if ratio[temp_pk[ndx]] > best_ratio and counts[bin_str] > 0:
            best_ratio = ratio[temp_pk[ndx]]
            best_bin_str = bin_str

    return best_bin_str


def _get_gini_indices(Xd, yd):
    """
    Compute the gini index for each attribute.
    """
    metas = {}
    gini_indices = {}

    X, y, keys = _get_numpy_data(Xd, yd)

    for i in range(X.shape[1]):
        meta = _get_gini_index(X[:, i], y)

        if meta['left_count'] > 0 and meta['right_count'] > 0:
            metas[i] = meta
            gini_indices[i] = meta['gini_index']

    return metas, gini_indices


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


def _type_counts(Xd, yd, ndx):
    """
    Counts the number of instance types between two an attribute and a label.
    """
    counts = {'00': 0, '01': 0, '10': 0, '11': 0}
    indices = {'00': 0, '01': 0, '10': 0, '11': 0}

    X, y, keys = _get_numpy_data(Xd, yd)
    x = X[:, ndx]

    for i in range(len(x)):
        counts[str(x[i]) + str(y[i])] += 1

    for k in indices.keys():
        v1, v2 = int(k[0]), int(k[1])
        indices[k] = keys[np.where((x == v1) & (y == v2))]

    return counts, indices


def _max_gini_index(y):
    """
    Find the maximum gini index given the number of classes.
    """
    c = len(np.unique(y))
    return 1 - c * np.square(1 / c)


def _check_metas(metas, x, y):
    """
    Check each meta for hanging branches.
    """
    invalid = []
    for i in metas.keys():
        if not _check(metas[i], x, y):
            invalid.append(i)
    return invalid


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


def _update_metas(metas, bin_str):
    """
    Updates all attributes' metadata given the instance type to delete.
    """
    new_metas = {}
    v1, v2 = (int(i) for i in bin_str)
    for i in metas.keys():
        new_metas[i] = _decrement(metas[i], v1, v2)
    return new_metas


def _probability_distribution(metas, lmbda, gamma):
    """
    Creates a probability distribution over the attributes given
    their gini index scores.
    """
    gini_indices = np.array([metas[i]['gini_index'] for i in metas])
    p = np.exp(-(lmbda * gini_indices) / (5 * gamma))
    p /= p.sum()
    keys = {a_ndx: i for i, a_ndx in enumerate(metas.keys())}
    return p, keys


def _decrement(meta, x, y):
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

    return _recompute(meta, left_pos, left_neg, right_pos, right_neg)


def _recompute(meta, left_pos=0, left_neg=0, right_pos=0, right_neg=0):
    """
    Decrements an instance from either or both branches and recomputes the gini gain.
    """

    meta = meta.copy()

    meta['n_pos'] -= (left_pos + right_pos)
    meta['n_count'] -= (left_pos + left_neg + right_pos + right_neg)
    meta['left_pos'] -= left_pos
    meta['left_count'] -= (left_pos + left_neg)
    meta['right_pos'] -= right_pos
    meta['right_count'] -= (right_pos + right_neg)
    meta['gini_index'] = _compute_gini_index(meta['n_pos'], meta['n_count'], meta['left_pos'],
                                             meta['left_count'], meta['right_pos'], meta['right_count'])
    return meta


def _get_numpy_data(Xd, yd):
    """
    Collects the data from the dicts as specified by indices,
    then puts them into numpy arrays.
    """
    assert len(Xd) == len(yd)
    n_samples = len(Xd)
    n_features = len(Xd[next(iter(Xd))])
    X = np.zeros((n_samples, n_features), np.int32)
    y = np.zeros(n_samples, np.int32)
    keys = np.zeros(n_samples, np.int32)

    for i, ndx in enumerate(Xd.keys()):
        X[i] = Xd[ndx]
        y[i] = yd[ndx]
        keys[i] = ndx

    return X, y, keys


def _numpy_to_dict(X, y):
    """
    Converts numpy data into dicts.
    """
    Xd, yd = {}, {}
    for i in range(X.shape[0]):
        Xd[i] = X[i]
        yd[i] = y[i]
    return Xd, yd
