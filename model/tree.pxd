import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

from splitter cimport Meta
from splitter cimport SplitRecord
from splitter cimport Splitter

cdef class Tree:
    """
    The Tree object is a binary tree structure constructed by the
    TreeBuilder. The tree structure is used for predictions.
    """

    # Input/Output layout
    cdef public SIZE_t n_features        # Number of features in X

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public int max_depth            # Max depth of the tree
    cdef public int node_count           # Counter for node IDs
    cdef public int capacity             # Capacity of tree, in terms of nodes
    cdef double* values                  # Array of values, shape=[capacity]
    cdef int* chosen_features            # Array of chosen features, shape=[capacity]
    cdef int* left_children              # Array of left children indices, shape=[capacity]
    cdef int* right_children             # Array of right children indices, shape=[capacity]

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

    # C API
    cdef int add_node(self, int parent, bint is_left, bint is_leaf, int feature,
                      double value, int* samples, Meta* meta) nogil except -1
    cdef np.ndarray _get_double_ndarray(self, double *data)
    cdef np.ndarray _get_int_ndarray(self, int *data)
    cdef int _resize(self, int capacity=*) nogil except -1

cdef class TreeBuilder:
    """
    The TreeBuilder recursively builds a Tree object from training samples,
    using a Splitter object for splitting internal nodes and assigning values to leaves.

    This class controls the various stopping criteria and the node splitting
    evaluation order, e.g. depth-first or breadth-first.
    """

    cdef Splitter splitter           # Splitter object that chooses the attribute to split on
    cdef int min_samples_split       # Minimum number of samples in an internal node
    cdef int min_samples_leaf        # Minimum number of samples in a leaf
    cdef int max_depth               # Maximal tree depth

    cpdef void build(self, Tree tree, object X, np.ndarray y, np.ndarray f)
    cdef inline _check_input(self, object X, np.ndarray y, np.ndarray f)
    cdef double _leaf_value(self, int[::1] y, int* samples, int n_samples, Meta* meta) nogil
