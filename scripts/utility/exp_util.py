"""
Utility methods to make epxeriments easier.
"""
import numpy as np
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


def get_random_state(seed):
    """
    Get a random number from the whole range of large integer values.
    """
    np.random.seed(seed)
    return np.random.randint(MAX_INT)
