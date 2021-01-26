import numpy as np
cimport numpy as np

from ._manager cimport _DataManager
from ._splitter cimport SplitRecord
from ._tree cimport Node
from ._tree cimport _Tree
from ._tree cimport _TreeBuilder
from ._utils cimport DTYPE_t
from ._utils cimport SIZE_t
from ._utils cimport INT32_t
from ._utils cimport UINT32_t

cdef class _Adder:
    """
    Recursively adds data to a _Tree built using _TreeBuilder; retrains
    when the indistinguishability bounds epsilon has been violated.
    """

    # Inner structures
    cdef _DataManager manager              # Database manager
    cdef _TreeBuilder tree_builder         # Tree Builder
    cdef bint         use_gini             # Controls splitting criterion
    cdef SIZE_t       min_samples_leaf     # Minimum number of samples for a leaf
    cdef SIZE_t       min_samples_split    # Minimum number of samples for a split

    # Metric structures
    cdef SIZE_t   capacity                 # Number of addition allocations for space
    cdef SIZE_t   add_count                # Number of additions
    cdef INT32_t* add_types                # Add types
    cdef INT32_t* add_depths               # Depth of leaf or node needing retraining
    cdef SIZE_t   retrain_sample_count     # Number of samples used for retraining

    # Python API
    cpdef INT32_t add(self, _Tree tree)
    cpdef void    clear_add_metrics(self)

    # C API
    cdef void _add(self, Node** node_ptr, double** X, int* y,
                   int* samples, int n_samples) nogil

    cdef int _check_node(self, Node* node, double** X, int* y,
                          int* samples, int n_samples, int pos_count,
                          SplitRecord *split) nogil

    cdef int _update_splits(self, Node** node_ptr, double** X, int* y,
                            int* samples, int n_samples, int pos_count) nogil

    cdef void _update_leaf(self, Node** node_ptr, int* y,
                           int* samples, int n_samples, int pos_count) nogil

    cdef void _retrain(self, Node*** node_ptr, double** X, int* y, int* samples,
                       int n_samples) nogil

    cdef void _get_leaf_samples(self, Node* node, int** leaf_samples_ptr,
                                int* leaf_samples_count_ptr) nogil

    cdef void _add_leaf_samples(self, int* samples, int n_samples,
                                int** leaf_samples_ptr,
                                int*  leaf_samples_count_ptr) nogil

    cdef void _add_samples(self, int* samples, int n_samples,
                           int** leaf_samples_ptr,
                           int*  leaf_samples_count_ptr) nogil

    cdef void _update_decision_node(self, Node** node_ptr, SplitRecord *split) nogil

    cdef void _add_add_type(self, int add_type, int add_depth) nogil
    cdef np.ndarray _get_int_ndarray(self, int *data, int n_elem)
