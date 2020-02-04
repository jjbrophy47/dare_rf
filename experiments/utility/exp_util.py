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


def check_args(args):
    """
    Checks specific args thta support multiple dtypes.
    """
    if hasattr(args, 'max_features'):
        if args.max_features == 'sqrt':
            return args
        elif '.' in args.max_features:
            args.max_features = float(args.max_features)
        else:
            args.max_features = int(args.max_features)

    if hasattr(args, 'max_samples'):
        if args.max_samples is None:
            return args
        elif '.' in args.max_samples:
            args.max_samples = float(args.max_samples)
        else:
            args.max_samples = int(args.max_samples)

    return args
