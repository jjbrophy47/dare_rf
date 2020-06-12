"""
CeDAR implementation selector.
"""
from . import cedar1
from . import cedar2
# from . import cedar3


def forest(epsilon=1.0, lmbda=0.1, n_estimators=100, max_features='sqrt',
           max_depth=10, criterion='gini', min_samples_split=2, min_samples_leaf=1,
           random_state=None, verbose=0, cedar_type='1'):
    """
    CeDAR Forest.

    Parameters:
    -----------
    epsilon: float (default=0.1)
        Controls the level of indistinguishability; lower for stricter gaurantees,
        higher for more deletion efficiency.
    lmbda: float (default=0.1)
        Controls the amount of noise injected into the learning algorithm.
        Set to -1 for detemrinistic trees; equivalent to setting it to infty.
    n_estimators: int (default=100)
        Number of trees in the forest.
    max_features: int float, or str (default='sqrt')
        If int, then max_features at each split.
        If float, then max_features=int(max_features * n_features) at each split.
        If None or 'sqrt', then max_features=sqrt(n_features).
    max_depth: int (default=None)
        The maximum depth of a tree.
    criterion: str (default='gini')
        Splitting criterion to use.
    min_samples_split: int (default=2)
        The minimum number of samples needed to make a split when building a tree.
    min_samples_leaf: int (default=1)
        The minimum number of samples needed to make a leaf.
    random_state: int (default=None)
        Random state for reproducibility.
    verbose: int (default=0)
        Verbosity level.
    """
    if cedar_type == '1':
        model_func = cedar1.Forest

    elif cedar_type == '2':
        model_func = cedar2.Forest

    else:
        model_func = cedar3.Forest

    model = model_func(epsilon=epsilon,
                       lmbda=lmbda,
                       n_estimators=n_estimators,
                       max_features=max_features,
                       max_depth=max_depth,
                       criterion=criterion,
                       min_samples_split=min_samples_split,
                       min_samples_leaf=min_samples_leaf,
                       random_state=random_state,
                       verbose=verbose)

    return model


def tree(epsilon=0.1, lmbda=0.1, max_depth=4, criterion='gini',
         min_samples_split=2, min_samples_leaf=1, random_state=None,
         verbose=0, cedar_type='1'):
    """
    CeDAR Tree.

    Parameters:
    -----------
    epsilon: float (default=0.1)
        Controls the level of indistinguishability; lower for stricter gaurantees,
        higher for more deletion efficiency.
    lmbda: float (default=0.1)
        Controls the amount of noise injected into the learning algorithm.
        Set to -1 for a detrminisic tree; equivalent to setting it to infty.
    max_depth: int (default=None)
        The maximum depth of a tree.
    criterion: str (default='gini')
        Splitting criterion to use.
    min_samples_split: int (default=2)
        The minimum number of samples needed to make a split when building a tree.
    min_samples_leaf: int (default=1)
        The minimum number of samples needed to make a leaf.
    random_state: int (default=None)
        Random state for reproducibility.
    verbose: int (default=0)
        Verbosity level.
    """
    if cedar_type == '1':
        model_func = cedar1.Tree

    elif cedar_type == '2':
        model_func = cedar2.Tree

    else:
        model_func = cedar3.Tree

    model = model_func(epsilon=epsilon,
                       lmbda=lmbda,
                       max_depth=max_depth,
                       criterion=criterion,
                       min_samples_split=min_samples_split,
                       min_samples_leaf=min_samples_leaf,
                       random_state=random_state,
                       verbose=verbose)

    return model
