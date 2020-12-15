import numpy as np
cimport numpy as np

from ._tree cimport Node
from ._utils cimport UINT32_t


cdef struct SplitRecord:

    # Data to track sample split
    int  feature                 # Which feature to split on.
    int* left_indices            # Samples in left branch of feature.
    int  left_count              # Number of samples in left branch.
    int* right_indices           # Samples in right branch of feature.
    int  right_count             # Number of samples in right branch.
    int* invalid_left_features   # Invalid features to consider for left children.
    int* invalid_right_features  # Invalid features to consider for right children.
    int  invalid_features_count  # Number of invalid features after split.

    # Extra metadata
    int     count               # Number of samples in the node
    int     pos_count           # Number of positive samples in the node

cdef class _Splitter:
    """
    The splitter searches in the input space for a feature to split on.
    """
    # Internal structures
    cdef public int min_samples_leaf       # Min samples in a leaf
    cdef bint use_gini                     # Controls splitting criterion

    # Methods
    cdef int split_node(self, Node* node, int** X, int* y,
                        int* samples, int n_samples,
                        int topd, int min_support,
                        UINT32_t* random_state,
                        SplitRecord *split) nogil

    cdef int compute_splits(self, Node** node_ptr, int** X, int* y,
                            int* samples, int n_samples) nogil

    cdef void select_features(self, Node** node, int n_features, int n_max_features,
                              int* invalid_features, int n_invalid_features,
                              UINT32_t* random_state, int* features) nogil
