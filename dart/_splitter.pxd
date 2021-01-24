from ._tree cimport Node
from ._tree cimport Feature
from ._tree cimport Threshold
from ._tree cimport IntList
from ._config cimport _Config
from ._utils cimport DTYPE_t
from ._utils cimport SIZE_t
from ._utils cimport INT32_t
from ._utils cimport UINT32_t

# constants
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7


"""
Object to keep track of the data parition during a split.
"""
cdef struct SplitRecord:
    IntList* left_samples                  # Samples in left branch of feature
    IntList* right_samples                 # Samples in right branch of feature
    IntList* left_constant_features        # Samples in left branch of feature
    IntList* right_constant_features       # Samples in right branch of feature

cdef class _Splitter:
    """
    The splitter searches in the input space for a feature to split on.
    """

    # Internal properties
    cdef _Config config                    # Configuration object

    # methods
    cdef SIZE_t select_threshold(self,
                                 Node*        node,
                                 DTYPE_t**    X,
                                 INT32_t*     y,
                                 IntList*     samples,
                                 SIZE_t       n_total_features) nogil

# Helper methods
cdef SIZE_t select_greedy_threshold(Node*     node,
                                    DTYPE_t** X,
                                    INT32_t*  y,
                                    IntList* samples,
                                    SIZE_t    n_total_features,
                                    _Config   config) nogil

cdef SIZE_t select_random_threshold(Node*     node,
                                    DTYPE_t** X,
                                    IntList*  samples,
                                    SIZE_t    n_total_features,
                                    _Config   config) nogil

cdef SIZE_t get_candidate_thresholds(DTYPE_t*     values,
                                     INT32_t*     labels,
                                     SIZE_t*      indices,
                                     SIZE_t       n_samples,
                                     SIZE_t       n_pos_samples,
                                     SIZE_t       min_samples_leaf,
                                     Threshold**  thresholds) nogil
