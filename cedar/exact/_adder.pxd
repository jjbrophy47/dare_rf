import numpy as np
cimport numpy as np

from ._manager cimport _DataManager
from ._splitter cimport SplitRecord
from ._tree cimport Node
from ._tree cimport _Tree
from ._tree cimport _TreeBuilder

cdef class _Adder:
    """
    Recursively adds data to a _Tree built using _TreeBuilder; retrains
    when the indistinguishability bounds epsilon has been violated.
    """

    # Inner structures
    cdef _DataManager manager        # Database manager
    cdef _TreeBuilder tree_builder   # Tree Builder
    cdef double epsilon              # Indistinguishability parameter
    cdef double lmbda                # Noise parameter
    cdef bint use_gini               # Controls splitting criterion
    cdef int min_samples_leaf        # Minimum number of samples for a leaf
    cdef int min_samples_split       # Minimum number of samples for a split

    # Metric structures
    cdef int capacity           # Number of addition allocations for space
    cdef int  add_count         # Number of additions
    cdef int* add_types         # Add type
    cdef int* add_depths        # Depth of leaf or node needing retraining

    # Python API
    cpdef int add(self, _Tree tree)
    cpdef void clear_add_metrics(self)

    # C API
    cdef void _add(self, Node** node_ptr, int** X, int* y,
                   int* samples, int n_samples, double parent_p) nogil
    cdef int _check_node(self, Node* node, int** X, int* y,
                          int* samples, int n_samples, int pos_count,
                          double parent_p, SplitRecord *split) nogil
    cdef int _update_splits(self, Node** node_ptr, int** X, int* y,
                            int* samples, int n_samples, int pos_count) nogil

    cdef void _update_leaf(self, Node** node_ptr, int* y,
                           int* samples, int n_samples, int pos_count) nogil
    cdef void _retrain(self, Node*** node_ptr, int** X, int* y, int* samples,
                       int n_samples, double parent_p, SplitRecord *split) nogil
    cdef void _get_leaf_samples(self, Node* node, int** leaf_samples_ptr,
                                int* leaf_samples_count_ptr) nogil
    cdef void _add_leaf_samples(self, int* samples, int n_samples,
                                int** leaf_samples_ptr,
                                int*  leaf_samples_count_ptr) nogil
    cdef void _update_decision_node(self, Node** node_ptr, SplitRecord *split) nogil

    cdef void _resize_metrics(self, int capacity=*) nogil
    cdef void _add_add_type(self, int add_type, int add_depth) nogil
    cdef np.ndarray _get_int_ndarray(self, int *data, int n_elem)
