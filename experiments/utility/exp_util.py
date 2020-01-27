"""
Utility methods to make epxeriments easier.
"""
from sklearn.metrics import roc_auc_score, accuracy_score


def performance(model, X_test, y_test, display=True, logger=None, name=''):
    """
    Returns AUROC and accuracy scores.
    """
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)

    if logger:
        logger.info('[{}] roc_auc: {:.3f}, acc: {:.3f}'.format(name, auc, acc))
    else:
        print('[{}] roc_auc: {:.3f}, acc: {:.3f}'.format(name, auc, acc))

    return auc, acc
