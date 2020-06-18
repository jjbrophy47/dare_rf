"""
CeDAR implementation selector.
"""
from . import cedar_single
from . import cedar_layer
from . import cedar_pyramid
from . import exact


def forest(epsilon=1.0, lmbda=0.1, criterion='gini', n_estimators=100, max_features='sqrt',
           max_depth=10, min_samples_split=2, min_samples_leaf=1,
           cedar_type='pyramid', random_state=None, verbose=0):
    """
    CeDAR Forest.

    Parameters:
    -----------
    epsilon: float (default=0.1)
        Controls the level of indistinguishability; lower for stricter gaurantees,
        higher for more deletion efficiency.
    lmbda: float (default=0.1)
        Controls the amount of noise injected into the learning algorithm.
        Set to -1 for deterministic trees; equivalent to setting it to infinity.
    criterion: str (default='gini')
        Splitting criterion to use.
    n_estimators: int (default=100)
        Number of trees in the forest.
    max_features: int float, or str (default='sqrt')
        If int, then max_features at each split.
        If float, then max_features=int(max_features * n_features) at each split.
        If None or 'sqrt', then max_features=sqrt(n_features).
    max_depth: int (default=None)
        The maximum depth of a tree.
    min_samples_split: int (default=2)
        The minimum number of samples needed to make a split when building a tree.
    min_samples_leaf: int (default=1)
        The minimum number of samples needed to make a leaf.
    cedar_type: str {'single', 'layer', 'pyramid'} (default='pyramid')
        Different types represent different budget allocation protocols.
    random_state: int (default=None)
        Random state for reproducibility.
    verbose: int (default=0)
        Verbosity level.
    """
    if cedar_type == 'single':
        model_func = cedar_single.Forest

    elif cedar_type == 'layer':
        model_func = cedar_layer.Forest

    elif cedar_type == 'pyramid':
        model_func = cedar_pyramid.Forest

    elif cedar_type == 'exact':
        model_func = exact.Forest
        lmbda = -1
        epsilon = 0

    else:
        raise ValueError('Unknown cedar_type: {}'.format(cedar_type))

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


def tree(epsilon=0.1, lmbda=0.1, criterion='gini', max_depth=4,
         min_samples_split=2, min_samples_leaf=1,
         cedar_type='pyramid', random_state=None, verbose=0):
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
    criterion: str (default='gini')
        Splitting criterion to use.
    max_depth: int (default=None)
        The maximum depth of a tree.
    min_samples_split: int (default=2)
        The minimum number of samples needed to make a split when building a tree.
    min_samples_leaf: int (default=1)
        The minimum number of samples needed to make a leaf.
    cedar_type: str (default='3')
        Different types represent different budget allocation protocols.
    topd: int (default=-1)
        Number of top layers to share a budget (only relevant if `cedar_type`='3').
    random_state: int (default=None)
        Random state for reproducibility.
    verbose: int (default=0)
        Verbosity level.
    """
    if cedar_type == 'single':
        model_func = cedar_single.Tree

    elif cedar_type == 'layer':
        model_func = cedar_layer.Tree

    elif cedar_type == 'pyramid':
        model_func = cedar_pyramid.Tree

    elif cedar_type == 'exact':
        model_func = exact.Tree
        lmbda = -1
        epsilon = 0

    else:
        raise ValueError('Unknown cedar_type: {}'.format(cedar_type))

    model = model_func(epsilon=epsilon,
                       lmbda=lmbda,
                       max_depth=max_depth,
                       criterion=criterion,
                       min_samples_split=min_samples_split,
                       min_samples_leaf=min_samples_leaf,
                       random_state=random_state,
                       verbose=verbose)

    return model
