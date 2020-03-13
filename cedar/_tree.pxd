import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

from ._splitter cimport Meta
from ._splitter cimport SplitRecord
from ._splitter cimport _Splitter

cdef class _Tree:
    """
    The Tree object is a binary tree structure constructed by the
    TreeBuilder. The tree structure is used for predictions.
    """

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public int node_count           # Counter for node IDs
    cdef public int capacity             # Capacity of tree, in terms of nodes
    cdef double* values                  # Array of values, shape=[capacity]
    cdef double* p                       # Array of probabilities, shape=[capacity]
    cdef int* chosen_features            # Array of chosen features, shape=[capacity]
    cdef int* left_children              # Array of left children indices, shape=[capacity]
    cdef int* right_children             # Array of right children indices, shape=[capacity]
    cdef int* depth                      # Array of depths, shape=[capacity]

    # Internal metadata, stored for efficient updating
    cdef int* count                # Array of sample counts, shape=[capacity]
    cdef int* pos_count            # Array of positive sample counts, shape=[capacity]
    cdef int* feature_count        # Array of feature counts, shape=[capacity]
    cdef int** left_counts         # Array of arrays of left counts, shape=[capacity, n_features]
    cdef int** left_pos_counts     # Array of arrays of left positive counts, shape=[capacity, n_features]
    cdef int** right_counts        # Array of arrays of right counts, shape=[capacity, n_features]
    cdef int** right_pos_counts    # Array of arrays of right positive counts, shape=[capacity, n_features]
    cdef int** features            # Array of arrays of feature indices for decision nodes, shape[capacity, n_features]
    cdef int** leaf_samples        # Array of arrays of sample indices for leaf nodes, shape[capacity, count]

    # Python/C API
    cpdef np.ndarray predict(self, object X)
    cpdef np.ndarray _get_left_counts(self, node_id)
    cpdef np.ndarray _get_left_pos_counts(self, node_id)
    cpdef np.ndarray _get_right_counts(self, node_id)
    cpdef np.ndarray _get_right_pos_counts(self, node_id)
    cpdef np.ndarray _get_features(self, node_id)
    cpdef np.ndarray _get_leaf_samples(self, node_id)

    # C API
    cdef int add_node(self, int parent, bint is_left, bint is_leaf, int feature,
                      double value, int depth, int* samples, Meta* meta) nogil except -1
    cdef np.ndarray _get_double_ndarray(self, double *data)
    cdef np.ndarray _get_int_ndarray(self, int *data)
    cdef int _resize(self, int capacity=*) nogil except -1

cdef class _TreeBuilder:
    """
    The TreeBuilder recursively builds a Tree object from training samples,
    using a Splitter object for splitting internal nodes and assigning values to leaves.

    This class controls the various stopping criteria and the node splitting
    evaluation order, e.g. depth-first or breadth-first.
    """

    cdef _Splitter splitter           # Splitter object that chooses the attribute to split on
    cdef int min_samples_split       # Minimum number of samples in an internal node
    cdef int min_samples_leaf        # Minimum number of samples in a leaf
    cdef int max_depth               # Maximal tree depth
    cdef int random_state            # Random state

    # Python API
    cpdef void build(self, _Tree tree, object X, np.ndarray y, np.ndarray f)

    # C API
    cdef void build_at_node(self, _Tree tree, object X, np.ndarray y, np.ndarray f,
                            int node_id, int depth, int parent, double parent_p,
                            bint is_left, int* samples, int* features, int n_features)
    cdef double _leaf_value(self, int[::1] y, int* samples, int n_samples, Meta* meta) nogil
