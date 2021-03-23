"""
Utility methods to make epxeriments easier.
"""
import time

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score

MAX_INT = 2147483647


def performance(model, X_test, y_test, logger=None,
                name='', do_print=False):
    """
    Returns AUROC and accuracy scores.
    """
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    ap = average_precision_score(y_test, proba)

    score_str = '[{}] auc: {:.3f}, acc: {:.3f}, ap: {:.3f}'

    if logger:
        logger.info(score_str.format(name, auc, acc, ap))
    elif do_print:
        print(score_str.format(name, auc, acc))

    return auc, acc, ap


def get_params(dataset, criterion='gini'):
    """
    Return selected hyperparameters for DARE models
    using error tolerances of 0%, 0.1%, 0.25%, 0.5%, and 1.0%.
    """
    # no. trees, max. depth, k, drmax for 0%, 0.1%, 0.25%, 0.5%, and 1.0%

    # gini
    gini = {}
    gini['surgical'] = [100, 20, 25, 0, 0, 1, 2, 4]
    gini['vaccine'] = [50, 20, 5, 0, 5, 7, 11, 14]
    gini['adult'] = [50, 20, 5, 0, 10, 13, 14, 16]
    gini['bank_marketing'] = [100, 20, 25, 0, 6, 9, 12, 14]
    gini['flight_delays'] = [250, 20, 25, 0, 1, 3, 5, 10]
    gini['diabetes'] = [250, 20, 5, 0, 7, 10, 12, 15]
    gini['no_show'] = [250, 20, 10, 0, 1, 3, 6, 10]
    gini['olympics'] = [250, 20, 5, 0, 0, 1, 2, 3]
    gini['census'] = [100, 20, 25, 0, 6, 9, 12, 16]
    gini['credit_card'] = [250, 20, 5, 0, 5, 8, 14, 17]
    gini['ctr'] = [100, 10, 50, 0, 2, 3, 4, 6]
    gini['twitter'] = [100, 20, 5, 0, 2, 4, 7, 11]
    gini['synthetic'] = [50, 20, 10, 0, 0, 2, 3, 5]
    gini['higgs'] = [50, 20, 10, 0, 1, 3, 6, 9]

    # entropy
    entropy = {}
    entropy['surgical'] = [100, 20, 50, 0, 1, 1, 2, 4]
    entropy['vaccine'] = [250, 20, 5, 0, 6, 9, 11, 15]
    entropy['adult'] = [50, 20, 5, 0, 9, 12, 14, 15]
    entropy['bank_marketing'] = [100, 10, 10, 0, 1, 1, 3, 4]
    entropy['flight_delays'] = [250, 20, 50, 0, 1, 3, 5, 10]
    entropy['diabetes'] = [100, 20, 5, 0, 4, 10, 11, 14]
    entropy['no_show'] = [250, 20, 10, 0, 1, 3, 6, 9]
    entropy['olympics'] = [250, 20, 5, 0, 0, 1, 2, 4]
    entropy['census'] = [100, 20, 25, 0, 5, 8, 11, 15]
    entropy['credit_card'] = [250, 10, 25, 0, 1, 2, 3, 4]
    entropy['ctr'] = [100, 10, 25, 0, 2, 3, 4, 6]
    entropy['twitter'] = [100, 20, 5, 0, 3, 5, 8, 11]
    entropy['synthetic'] = [50, 20, 10, 0, 1, 2, 3, 6]
    entropy['higgs'] = [50, 20, 10, 0, 0, 2, 5, 8]

    # select params
    params = gini[dataset] if criterion == 'gini' else entropy[dataset]

    return params


def explain(model, X_train, y_train, X_test, y_test=None):
    """
    Generate an instance-attribution explanation for each test
    instance, returning a matrix of shape=(no. train, X.shape[0]).

    Entry i, j in the matrix represents the effect training
    sample i has on the prediction of test instance j.

    If the labels of the test instances are given (y), then positive
    numbers in the matrix correspond to training samples that contribute
    towards the predicted label, and vice versa if negative.
    Otherwise, the value in each cell is simply the difference from the
    original prediction.
    """
    assert X_train.shape[1] == X_test.shape[1]
    if y_test is not None:
        assert y_test.ndim == 1

    # setup containers
    initial_proba = model.predict_proba(X_test)[:, 1]
    impact = np.zeros(shape=(X_train.shape[0], X_test.shape[0]))

    # measure the effect of each training sample
    for i in tqdm(range(X_train.shape[0])):
        model.delete(i)
        proba = model.predict_proba(X_test)[:, 1]
        impact[i] = initial_proba - proba

        # flip contribution for predictions whose label=1
        if y_test is not None:
            for j in range(X_test.shape[0]):
                if y_test[j] == 1:
                    impact[i][j] *= -1

        model.add(X_train[[i]], y_train[[i]])

    return impact


def explain_lite(model, X_train, y_train, X_test, y_test=None,
                 use_abs=False, print_cnt=100, logger=None):
    """
    Generate an instance-attribution explanation for each test
    instance, and then sum over the test instances,
    returning a matrix of shape=(X.shape[0],).

    Entry i in the vector represents the sum effect training
    sample i has on the prediction of the test instances.

    If the labels of the test instances are given (y), then positive
    numbers in the matrix correspond to training samples that contribute
    towards the predicted label, and vice versa if negative.
    Otherwise, the value in each cell is simply the difference from the
    original prediction.

    If using absolute (abs), then take the sum over the absolute
    contributions.
    """
    start = time.time()

    assert X_train.shape[1] == X_test.shape[1]
    if y_test is not None:
        assert y_test.ndim == 1

    # setup containers
    initial_proba = model.predict_proba(X_test)[:, 1]
    impact = np.zeros(shape=(X_train.shape[0],))

    # measure the effect of each training sample
    for i in tqdm(range(X_train.shape[0])):
        model.delete(i)
        proba = model.predict_proba(X_test)[:, 1]
        diff = initial_proba - proba

        # flip contribution for predictions whose label=1
        if y_test is not None:
            for j in range(X_test.shape[0]):
                if y_test[j] == 1:
                    diff[j] *= -1

        impact[i] = np.sum(diff) if use_abs else np.sum(np.abs(diff))

        model.add(X_train[[i]], y_train[[i]])

        if logger and i % print_cnt == 0:
            elapsed = time.time() - start
            logger.info('[Influence on sample {}] cum time: {:.3f}s'.format(i, elapsed))

    return impact


def get_random_state(seed):
    """
    Get a random number from the whole range of large integer values.
    """
    np.random.seed(seed)
    return np.random.randint(MAX_INT)
