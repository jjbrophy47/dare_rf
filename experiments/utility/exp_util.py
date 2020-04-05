"""
Utility methods to make epxeriments easier.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV

MAX_INT = 2147483647


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


def get_random_state(seed):
    """
    Get a random number from the whole range of large insteger values.
    """
    np.random.seed(seed)
    return np.random.randint(MAX_INT)


def tune_model(model, X_train, y_train, param_grid, cv=2,
               scoring='accuracy', tol=0.01, verbose=0,
               logger=None, seed=0):
    """
    Tune the hyperparameters of a CeDAR model.
    """
    gs = GridSearchCV(model, param_grid, scoring=scoring, cv=cv,
                      verbose=verbose, refit=False)
    gs = gs.fit(X_train, y_train)

    cols = ['mean_fit_time', 'mean_test_score', 'rank_test_score']
    cols += ['param_{}'.format(param) for param in param_grid.keys()]

    # filter the parameters with the highest performances
    df = pd.DataFrame(gs.cv_results_)
    qf = df[df['mean_test_score'].max() - df['mean_test_score'] <= tol]

    if 'lmbda' in param_grid:
        qf = qf[qf['param_lmbda'] != -1]

    best_df = qf.sort_values('mean_fit_time').reset_index().loc[0]
    best_ndx = best_df['index']
    best_params = best_df['params']

    if logger and verbose > 0:
        logger.info('gridsearch results:')
        logger.info(df[cols].sort_values('rank_test_score'))
        logger.info('tolerance: {}'.format(tol))
        logger.info('best_index: {}, best_params: {}'.format(best_ndx, best_params))

    return best_params
