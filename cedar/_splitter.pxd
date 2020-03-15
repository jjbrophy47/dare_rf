import numpy as np
cimport numpy as np

cdef struct Meta:
    # Sufficient statistics to save for each attribute
    double p                 # Total probability of chosen feature
    int  count               # Number of samples in the node
    int  pos_count           # Number of pos samples in the node
    int  feature_count       # Number of features in the node
    int* left_counts         # Number of left samples for each attribute
    int* left_pos_counts     # Number of left positive samples for each attribute
    int* right_counts        # Number of right samples for each attribute
    int* right_pos_counts    # Number of right positive samples for each attribute
    int* features            # Valid features considered in the node


cdef struct SplitRecord:
    # Data to track sample split
    int  feature                 # Which feature to split on.
    int* left_indices            # Samples in left branch of feature.
    int* left_original_indices   # Original samples in left branch of feature.
    int  left_count              # Number of samples in left branch.
    int* right_indices           # Samples in right branch of feature.
    int* right_original_indices  # Original samples in right branch of feature.
    int  right_count             # Number of samples in right branch.
    int* features                # Valid features to consider for descendants.
    int  n_features              # Number of valid features after split.

cdef class _Splitter:
    """
    The splitter searches in the input space for a feature and a threshold
    to split the samples samples[start:end].
    The impurity computations are delegated to a criterion object.
    """
    # Internal structures
    cdef public int min_samples_leaf       # Min samples in a leaf
    cdef double lmbda                      # Noise control parameter
    cdef int random_state                  # Random state reference

    # Methods
    cdef int node_split(self, int[::1, :] X, int[::1] y, double parent_p,
                        int* samples, int* original_samples,
                        int* features, int n_features,
                        SplitRecord* split, Meta* meta)
