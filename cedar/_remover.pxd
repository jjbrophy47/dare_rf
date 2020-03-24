import numpy as np
cimport numpy as np

from ._manager cimport _DataManager
from ._splitter cimport SplitRecord
from ._tree cimport Node
from ._tree cimport _Tree
from ._tree cimport _TreeBuilder

cdef class _Remover:
    """
    Recursively removes data from a _Tree built using _TreeBuilder; retrains
    when the indistinguishability bounds epsilon has been violated.
    """

    # Inner structures
    cdef _DataManager manager        # Database manager
    cdef _TreeBuilder tree_builder   # Tree Builder
    cdef double epsilon              # Indistinguishability parameter
    cdef double lmbda                # Noise parameter
    cdef int min_samples_leaf        # Minimum number of samples for a leaf
    cdef int min_samples_split       # Minimum number of samples for a split

    # Metric structures
    cdef int capacity           # Number of removal allocations for space
    cdef int  remove_count      # Number of removals
    cdef int* remove_types      # Removal type: retrain / no retrain
    cdef int* remove_depths     # Depth of leaf or node needing retraining

    # Python API
    cpdef int remove(self, _Tree tree, np.ndarray remove_indices)
    cpdef void clear_removal_metrics(self)

    # C API
    cdef void _remove(self, Node** node_ptr, int** X, int* y,
                      int* samples, int n_samples, double parent_p) nogil
    cdef int _check_node(self, Node* node, int** X, int* y,
                          int* samples, int n_samples, int pos_count,
                          double parent_p, SplitRecord *split) nogil
    cdef int _update_splits(self, Node** node_ptr, int** X, int* y,
                            int* samples, int n_samples, int pos_count) nogil

    cdef void _update_leaf(self, Node** node_ptr, int* y, int* samples,
                           int n_samples, int pos_count) nogil
    cdef void _convert_to_leaf(self, Node** node_ptr, int* samples, int n_samples,
                               SplitRecord *split) nogil
    cdef void _retrain(self, Node** node_ptr, int** X, int* y, int* samples,
                       int n_samples, double parent_p, SplitRecord *split) nogil
    cdef void _get_leaf_samples(self, Node* node, int* remove_samples,
                                int n_remove_samples, int** leaf_samples_ptr,
                                int* leaf_samples_count_ptr) nogil
    cdef void _update_decision_node(self, Node** node_ptr, SplitRecord *split) nogil

    cdef void _resize_metrics(self, int capacity=*) nogil
    cdef void _add_removal_type(self, int remove_type, int remove_depth) nogil
    cdef np.ndarray _get_int_ndarray(self, int *data, int n_elem)
