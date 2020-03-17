import numpy as np
cimport numpy as np

from ._manager cimport _DataManager
from ._splitter cimport Meta
from ._tree cimport _Tree
from ._tree cimport _TreeBuilder

cdef struct RemovalSplitRecord:
    # Data to track sample split
    int* left_indices          # Samples in left branch of feature.
    int  left_count            # Number of samples in left branch.
    int* right_indices         # Samples in right branch of feature.
    int  right_count           # Number of samples in right branch.

cdef class _Remover:
    """
    Recursively removes data from a _Tree built using _TreeBuilder; retrains
    when the indistinguishability bounds epsilon has been violated.
    """

    # Inner structures
    cdef _DataManager manager  # Database manager
    cdef double epsilon        # Indistinguishability parameter
    cdef double lmbda          # Noise parameter

    # Python API
    cpdef int remove(self, _Tree tree, _TreeBuilder tree_builder,
                     np.ndarray remove_indices)

    # C API
    cdef int _node_remove(self, int node_id, int** X, int* y,
                          int* samples, int n_samples,
                          int min_samples_split, int min_samples_leaf,
                          int chosen_feature, double parent_p,
                          RemovalSplitRecord *split, Meta* meta) nogil
    cdef int _collect_leaf_samples(self, int node_id, int is_left, int parent,
                                   _Tree tree, int* remove_samples,
                                   int n_remove_samples,
                                   int** rebuild_samples_ptr)
    cdef int _update_leaf(self, int node_id, _Tree tree, int* y, int* samples,
                          int n_samples) nogil
    cdef int _update_decision_node(self, int node_id, _Tree tree,
                                   int n_samples, Meta* meta) nogil
