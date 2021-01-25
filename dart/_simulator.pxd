import numpy as np
cimport numpy as np

from ._manager cimport _DataManager
from ._tree cimport Node
from ._tree cimport Feature
from ._tree cimport Threshold
from ._tree cimport IntList
from ._tree cimport _Tree
from ._config cimport _Config
from ._utils cimport DTYPE_t
from ._utils cimport SIZE_t
from ._utils cimport INT32_t
from ._utils cimport UINT32_t

cdef class _Simulator:
    """
    Class to simulate the deletion of a training sample.
    """

    # Inner structures
    cdef _DataManager manager                # Database manager
    cdef _Config      config                 # Configuration object

    # Python API
    cpdef INT32_t sim_delete(self, _Tree tree, SIZE_t remove_index)

    # C API
    cdef INT32_t _sim_delete(self,
                             Node*     node,
                             DTYPE_t** X,
                             INT32_t*  y,
                             SIZE_t    remove_index) nogil

    cdef INT32_t check_optimal_split(self,
                                     Node*     node,
                                     Feature** features,
                                     SIZE_t    n_features) nogil

    cdef SIZE_t update_metadata(self,
                                Node*      node,
                                DTYPE_t**  X,
                                INT32_t*   y,
                                SIZE_t     remove_index,
                                Feature*** features_ptr,
                                SIZE_t*    n_features_ptr) nogil

    cdef SIZE_t update_greedy_node_metadata(self,
                                            Node*      node,
                                            DTYPE_t**  X,
                                            INT32_t*   y,
                                            SIZE_t     remove_index,
                                            Feature*** features_ptr,
                                            SIZE_t*    n_features_ptr) nogil

    cdef SIZE_t update_random_node_metadata(self,
                                            Node*     node,
                                            DTYPE_t** X,
                                            INT32_t*  y,
                                            SIZE_t    remove_index) nogil

# helper methods
cdef void remove_invalid_thresholds(Feature* feature,
                                    SIZE_t   n_valid_thresholds,
                                    SIZE_t*  threshold_validities) nogil

cdef SIZE_t sample_new_thresholds(Feature*  feature,
                                  SIZE_t    n_valid_thresholds,
                                  SIZE_t*   threshold_validities,
                                  Node*     node,
                                  DTYPE_t** X,
                                  INT32_t*  y,
                                  SIZE_t    remove_index,
                                  bint*     is_constant_feature_ptr,
                                  _Config   config) nogil

cdef SIZE_t sample_new_features(Feature*** features_ptr,
                                SIZE_t*    n_features_ptr,
                                IntList*   invalid_features,
                                SIZE_t     n_total_features,
                                Node*      node,
                                DTYPE_t**  X,
                                INT32_t*   y,
                                SIZE_t     remove_index,
                                _Config    config) nogil

cdef void get_leaf_samples(Node*    node,
                           SIZE_t   remove_index,
                           IntList* leaf_samples) nogil