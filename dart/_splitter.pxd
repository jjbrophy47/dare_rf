# import numpy as np
# cimport numpy as np

from ._tree cimport Node
from ._utils cimport UINT32_t


"""
Object to keep track of the data parition during a split.
"""
cdef struct SplitRecord:
    int*       left_samples       # Samples in left branch of feature
    int*       right_samples      # Samples in right branch of feature
    int        n_left_samples     # Number of samples in left branch
    int        n_right_samples    # Number of samples in right branch

cdef class _Splitter:
    """
    The splitter searches in the input space for a feature to split on.
    """
    # Internal structures
    cdef public int min_samples_leaf       # Min samples in a leaf
    cdef bint use_gini                     # Controls splitting criterion
    cdef int  k                            # Number of thresholds to sample

    # Methods
    cdef int split_node(self,
                        Node**       node_ptr,
                        double**     X,
                        int*         y,
                        int*         samples,
                        int          n_samples,
                        int          topd,
                        UINT32_t*    random_state,
                        SplitRecord* split) nogil

    cdef int compute_metadata(self,
                              Node**   node_ptr,
                              double** X,
                              int*     y,
                              int*     samples,
                              int      n_samples,
                              UINT32_t* random_state) nogil

    cdef void select_features(self,
                              Node**    node_ptr,
                              int       n_total_features,
                              int       n_max_features,
                              UINT32_t* random_state) nogil
