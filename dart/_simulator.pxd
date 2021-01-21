import numpy as np
cimport numpy as np

from ._manager cimport _DataManager
from ._tree cimport Node
from ._tree cimport Feature
from ._tree cimport _Tree
from ._config cimport _Config
from ._utils cimport DTYPE_t
from ._utils cimport SIZE_t
from ._utils cimport INT32_t
from ._utils cimport UINT32_t

cdef class _Simulator:
    """
    Object to simulate deletion operations without
    actually updating the model and database.
    """

    # Inner structures
    cdef _DataManager manager                # Database manager
    cdef _Config      config                 # Configuration object

    # Python API
    cpdef INT32_t sim_delete(self, _Tree tree, SIZE_t remove_index)

    # C API
    cdef INT32_t _sim_delete(self,
                             Node**    node_ptr,
                             DTYPE_t** X,
                             INT32_t*  y,
                             SIZE_t    remove_index) nogil

    cdef INT32_t check_optimal_split(self,
                                 Node*     node,
                                 Feature** features,
                                 SIZE_t    n_features) nogil

    cdef SIZE_t update_metadata(self,
                                Node*     node,
                                DTYPE_t** X,
                                INT32_t*  y,
                                SIZE_t    remove_index,
                                Feature** features,
                                SIZE_t    n_features) nogil

    cdef void get_leaf_samples(self,
                               Node*    node,
                               SIZE_t   remove_index,
                               SIZE_t** leaf_samples_ptr,
                               SIZE_t*  leaf_samples_count_ptr) nogil
