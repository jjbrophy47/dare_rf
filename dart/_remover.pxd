import numpy as np
cimport numpy as np

from ._manager cimport _DataManager
from ._splitter cimport SplitRecord
from ._tree cimport Node
from ._tree cimport _Tree
from ._tree cimport _TreeBuilder
from ._config cimport _Config
from ._utils cimport DTYPE_t
from ._utils cimport SIZE_t
from ._utils cimport INT32_t
from ._utils cimport UINT32_t

cdef class _Remover:
    """
    Recursively removes data from a _Tree built using _TreeBuilder;
    retrains a node / subtree when a better feature / threshold is optimal.
    """

    # Inner structures
    cdef _DataManager manager                # Database manager
    cdef _TreeBuilder tree_builder           # Tree Builder
    cdef _Config      config                 # Configuration object

    # Metric structures
    cdef SIZE_t   capacity                   # Number of removal allocations for space
    cdef SIZE_t   remove_count               # Number of removals
    cdef INT32_t* remove_types               # Type of deletion that occurs
    cdef INT32_t* remove_depths              # Depth of leaf or node needing retraining
    cdef INT32_t* remove_costs               # No. samples that need to be retrained

    # Python API
    cpdef INT32_t remove(self, _Tree tree, np.ndarray remove_indices)
    cpdef void clear_metrics(self)

    # C API
    cdef void _remove(self,
                      Node**    node_ptr,
                      DTYPE_t** X,
                      INT32_t*  y,
                      SIZE_t*   remove_samples,
                      SIZE_t    n_remove_samples) nogil

    cdef void update_node(self,
                          Node**   node_ptr,
                          INT32_t* y,
                          SIZE_t*  remove_samples,
                          SIZE_t   n_remove_samples) nogil

    cdef void update_leaf(self,
                          Node**  node_ptr,
                          SIZE_t* remove_samples,
                          SIZE_t  n_remove_samples) nogil

    cdef void convert_to_leaf(self,
                              Node**       node_ptr,
                              SIZE_t*      remove_samples,
                              SIZE_t       n_remove_samples,
                              SplitRecord* split) nogil

    cdef void retrain(self,
                      Node***   node_pp,
                      DTYPE_t** X,
                      INT32_t*  y,
                      SIZE_t*   samples,
                      SIZE_t    n_samples) nogil

    cdef void get_leaf_samples(self,
                               Node*    node,
                               SIZE_t*  remove_samples,
                               SIZE_t   n_remove_samples,
                               SIZE_t** leaf_samples_ptr,
                               SIZE_t*  leaf_samples_count_ptr) nogil

    cdef INT32_t check_optimal_split(self,
                                 Node* node) nogil

    cdef SIZE_t update_metadata(self,
                                Node**    node_ptr,
                                DTYPE_t** X,
                                INT32_t*  y,
                                SIZE_t*   samples,
                                SIZE_t    n_samples) nogil

    # metric methods
    cdef void add_metric(self,
                         INT32_t remove_type,
                         INT32_t remove_depth,
                         INT32_t remove_cost) nogil

    cdef np.ndarray get_int_ndarray(self,
                                    INT32_t *data,
                                    SIZE_t n_elem)
