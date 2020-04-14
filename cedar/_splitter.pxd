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
    int* left_features           # Valid features to consider for left children.
    int* right_features          # Valid features to consider for right children.
    int  features_count          # Number of valid features after split.

    # Extra metadata
    double p                   # Total probability of chosen feature
    int    count               # Number of samples in the node
    int    pos_count           # Number of positive samples in the node
    int*   left_counts         # Number of left samples for each attribute
    int*   left_pos_counts     # Number of left positive samples for each attribute
    int*   right_counts        # Number of right samples for each attribute
    int*   right_pos_counts    # Number of right positive samples for each attribute

cdef class _Splitter:
    """
    The splitter searches in the input space for a feature to split on.
    """
    # Internal structures
    cdef public int min_samples_leaf       # Min samples in a leaf
    cdef double lmbda                      # Noise control parameter

    cdef object random_state               # Random state reference
    cdef UINT32_t rand_r_state             # sklearn_rand_r random number state

    # Methods
    cdef int split_node(self, Node* node, int** X, int* y,
                        int* samples, int n_samples, double parent_p,
                        SplitRecord *split) nogil
    cdef int compute_splits(self, Node** node_ptr, int** X, int* y,
                            int* samples, int n_samples,
                            int* features, int n_features) nogil
