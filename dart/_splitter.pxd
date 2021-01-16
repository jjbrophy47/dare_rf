from ._tree cimport Node
from ._utils cimport DTYPE_t
from ._utils cimport SIZE_t
from ._utils cimport INT32_t
from ._utils cimport UINT32_t


"""
Object to keep track of the data parition during a split.
"""
cdef struct SplitRecord:
    SIZE_t*    left_samples       # Samples in left branch of feature
    SIZE_t*    right_samples      # Samples in right branch of feature
    SIZE_t     n_left_samples     # Number of samples in left branch
    SIZE_t     n_right_samples    # Number of samples in right branch

cdef class _Splitter:
    """
    The splitter searches in the input space for a feature to split on.
    """
    # Internal structures
    cdef public SIZE_t min_samples_leaf    # Min samples in a leaf
    cdef bint use_gini                     # Controls splitting criterion
    cdef SIZE_t  k                         # Number of thresholds to sample

    # Methods
    cdef void split_node(self,
                         Node**       node_ptr,
                         DTYPE_t**    X,
                         INT32_t*     y,
                         SIZE_t*      samples,
                         SIZE_t       n_samples,
                         SIZE_t       topd,
                         UINT32_t*    random_state,
                         SplitRecord* split) nogil

    cdef SIZE_t compute_metadata(self,
                                 Node**    node_ptr,
                                 DTYPE_t** X,
                                 INT32_t*  y,
                                 SIZE_t*   samples,
                                 SIZE_t    n_samples,
                                 UINT32_t* random_state) nogil

    cdef void select_features(self,
                              Node**    node_ptr,
                              SIZE_t    n_total_features,
                              SIZE_t    n_max_features,
                              UINT32_t* random_state) nogil
