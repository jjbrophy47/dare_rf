"""
Utility methods to make epxeriments easier.
"""
import pickle

from sklearn.metrics import roc_auc_score, accuracy_score


def performance(model, X_test, y_test, display=True, logger=None,
                name='', do_print=False):
    """
    Returns AUROC and accuracy scores.
    """
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)

    if logger:
        logger.info('[{}] roc_auc: {:.3f}, acc: {:.3f}'.format(name, auc, acc))
    elif do_print:
        print('[{}] roc_auc: {:.3f}, acc: {:.3f}'.format(name, auc, acc))

    return auc, acc


def save_info(path, model):
    """
    Saves tree mode info such as: epsilon, lambda, n_estimators,
    max_features, and max_depth.
    """
    with open(path, 'wb') as f:
        pickle.dump(model.get_params(), f, pickle.HIGHEST_PROTOCOL)


def load_info(path):
    """
    Loads dictonary containing model information.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)