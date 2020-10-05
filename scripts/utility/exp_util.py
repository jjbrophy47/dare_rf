"""
Utility methods to make epxeriments easier.
"""
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


def explain_lite(model, X_train, y_train, X_test, y_test=None, use_abs=False):
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

    return impact


def get_random_state(seed):
    """
    Get a random number from the whole range of large integer values.
    """
    np.random.seed(seed)
    return np.random.randint(MAX_INT)
