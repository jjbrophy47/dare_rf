import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

from splitter cimport SplitRecord
from splitter cimport Splitter

# cdef struct Meta:
#     # Storage structure of metadata at each node.
#     SIZE_t n_left                        # Number of left branch samples
#     SIZE_t n_left_pos                    # Number of left branch positive samples
#     SIZE_t n_right                       # Number of right branch samples
#     SIZE_t n_right_pos                   # Number of right branch positive samples

# cdef class Node:
#     """
#     Base storage unit of the binary tree.
#     """
#     cdef public SIZE_t left_child                    # id of the left child of the node
#     cdef public SIZE_t right_child                   # id of the right child of the node
#     cdef public SIZE_t feature                       # Feature used for splitting the node
#     cdef public SIZE_t n_node_samples                # Number of samples at the node
#     cdef Meta* meta                                  # Metadata of for each attribute split

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
    cdef int* n_samples                  # Array of sample counts, shape=[capacity]
    cdef int* features                   # Array of chosen features, shape=[capacity]
    cdef int* left_children              # Array of left children indices, shape=[capacity]
    cdef int* right_children             # Array of right children indices, shape=[capacity]

    # Methods
    cdef int add_node(self, int parent, bint is_left, bint is_leaf,
                      int feature, int n_node_samples,
                      double value) nogil except -1

    # # python/C API
    # cpdef np.ndarray predict(self, object X)
    # cpdef np.ndarray apply(self, object X)

    # # C API
    cdef np.ndarray _get_double_ndarray(self, double *data)
    cdef np.ndarray _get_int_ndarray(self, int *data)
    # cdef np.ndarray _get_node_ndarray(self)
    cdef int _resize(self, int capacity=*) nogil except -1
    # cdef int _resize_c(self, SIZE_t capacity=*) nogil except -1

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
    cdef double _leaf_value(self, int[::1] y, int* samples, int n_samples) nogil
