import numpy as np
cimport numpy as np

# from tree cimport DTYPE_t          # Type of X
# from tree cimport DOUBLE_t
# from tree cimport SIZE_t
from tree cimport INT32_t
from tree cimport UINT32_t

# from criterion cimport Criterion

cdef struct SplitRecord:
    # Data to track sample split
    int  feature              # Which feature to split on.
    int* left_indices         # Samples in left branch of feature.
    int  left_count           # Number of samples in left branch.
    int* right_indices        # Samples in right branch of feature.
    int  right_count          # Number of samples in right branch.
    int* features             # Valid features to consider for descendants.
    int  n_features           # Number of valid features after split.
    # SIZE_t pos             # Split samples array at the given position,
    #                        # i.e. count of samples below threshold for feature.
    #                        # pos is >= end if the node is a leaf.
    # double improvement     # Impurity improvement given parent node.
    # double impurity_left   # Impurity of the left split.
    # double impurity_right  # Impurity of the right split.

cdef class Splitter:
    """
    The splitter searches in the input space for a feature and a threshold
    to split the samples samples[start:end].
    The impurity computations are delegated to a criterion object.
    """

    # Internal structures
    # cdef public Criterion criterion      # Impurity criterion
    # cdef public SIZE_t max_features      # Number of features to test
    cdef public int min_samples_leaf       # Min samples in a leaf
    cdef double lmbda                      # Noise control parameter
    cdef UINT32_t random_state                  # Random state reference

    # cdef object random_state             # Random state
    # cdef UINT32_t rand_r_state           # sklearn_rand_r random number state

    # cdef int* samples                 # Sample indices in X, y
    # cdef int n_samples                # X.shape[0]
    # cdef int* features                # Feature indices in X
    # cdef SIZE_t* constant_features       # Constant features indices
    # cdef int n_features               # X.shape[1]
    # cdef DTYPE_t* feature_values         # temp. array holding feature values

    # cdef SIZE_t start                    # Start position for the current node
    # cdef SIZE_t end                      # End position for the current node

    # cdef np.ndarray[INT32_t, ndim=2] X
    # cdef np.ndarray[INT32_t, ndim=1] y

    # cdef np.ndarray X
    # cdef np.ndarray y
    # cdef DOUBLE_t* sample_weight

    # The samples vector `samples` is maintained by the Splitter object such
    # that the samples contained in a node are contiguous. With this setting,
    # `node_split` reorganizes the node samples `samples[start:end]` in two
    # subsets `samples[start:pos]` and `samples[pos:end]`.

    # The 1-d  `features` array of size n_features contains the features
    # indices and allows fast sampling without replacement of features.

    # The 1-d `constant_features` array of size n_features holds in
    # `constant_features[:n_constant_features]` the feature ids with
    # constant values for all the samples that reached a specific node.
    # The value `n_constant_features` is given by the parent node to its
    # child nodes.  The content of the range `[n_constant_features:]` is left
    # undefined, but preallocated for performance reasons
    # This allows optimization with depth-based tree building.

    # Methods
    cdef int init(self)

    # cdef int node_reset(self, SIZE_t start, SIZE_t end) nogil except -1

    cdef int node_split(self, int[::1, :] X, int[::1] y, int[::1] f,
                        int* samples, int n_samples, int* features, int n_features,
                        SplitRecord* split)

    cdef double _compute_gini(self, double count, double left_count, double right_count, 
                              int left_pos_count, int right_pos_count) nogil
    cdef int _generate_distribution(self, double* distribution, double* gini_indices,
                                    int n_gini_indices) nogil
    cdef int _sample_distribution(self, double* distribution, int n_distribution) nogil
    # cpdef int node_split(self, int[:, :] X, int[:] y)
    # cpdef int node_split(self, np.ndarray[INT32_t, ndim=2] X, np.ndarray[INT32_t, ndim=1] y)

    # cdef void node_value(self, double* dest) nogil

    # cdef double node_impurity(self) nogil
