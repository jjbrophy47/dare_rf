import numpy as np
cimport numpy as np

from ._manager cimport _DataManager
from ._splitter cimport SplitRecord
from ._tree cimport Node
from ._tree cimport _Tree
from ._tree cimport _TreeBuilder
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
    cdef bint         use_gini               # Controls splitting criterion
    cdef SIZE_t       k                      # Number of thresholds to sample

    # Metric structures
    cdef SIZE_t   capacity                   # Number of removal allocations for space
    cdef SIZE_t   remove_count               # Number of removals
    cdef INT32_t* remove_types               # Removal type
    cdef INT32_t* remove_depths              # Depth of leaf or node needing retraining
    cdef SIZE_t   retrain_sample_count       # Number of samples used for retraining

    # Python API
    cpdef INT32_t remove(self, _Tree tree, np.ndarray remove_indices)
    cpdef void clear_remove_metrics(self)

    # C API
    cdef void _remove(self,
                      Node**    node_ptr,
                      DTYPE_t** X,
                      INT32_t*  y,
                      SIZE_t*   remove_samples,
                      SIZE_t    n_remove_samples) nogil

    cdef SIZE_t update_node(self,
                            Node**   node_ptr,
                            INT32_t* y,
                            SIZE_t*  samples,
                            SIZE_t   n_samples) nogil

    cdef void update_leaf(self,
                          Node**  node_ptr,
                          SIZE_t* samples,
                          SIZE_t  n_samples) nogil

    cdef void convert_to_leaf(self,
                              Node**       node_ptr,
                              SIZE_t*      samples,
                              SIZE_t       n_samples,
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

    cdef void add_removal_type(self,
                               INT32_t remove_type,
                               INT32_t remove_depth) nogil

    cdef np.ndarray get_int_ndarray(self,
                                    INT32_t *data,
                                    SIZE_t n_elem)
