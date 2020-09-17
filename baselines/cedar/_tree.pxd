import numpy as np
cimport numpy as np

from ._manager cimport _DataManager
from ._splitter cimport SplitRecord
from ._splitter cimport _Splitter
from ._utils cimport UINT32_t

cdef struct Node:

    # Structure for predictions
    bint   is_leaf                   # Whether this node is a leaf
    double value                     # Value of node if leaf
    int    feature                   # Chosen split feature if decision node
    Node*  left                      # Left child node
    Node*  right                     # Right child node

    # Extra information
    int    depth                     # Depth of node
    bint   is_left                   # Whether this node is a left child

    # Metadata necessary for efficient updating
    double   budget                  # Total divergence for this node.
    double** layer_budget_ptr        # Pointer to pointer of Array of budgets, shape=[max_depth]
    int      count                   # Number of samples in the node
    int      pos_count               # Number of pos samples in the node
    int      features_count          # Number of features in the node
    int      invalid_features_count  # Number of invalid features in the node
    int*     features                # Array of valid features, shape=[feature_count]
    int*     invalid_features        # Array of invalid features, shape=[feature_count]
    int*     left_counts             # Array of left sample counts, shape=[feature_count]
    int*     left_pos_counts         # Array of left positive sample counts, shape=[feature_count]
    int*     right_counts            # Array of right sample counts, shape=[feature_count]
    int*     right_pos_counts        # Array of right positive sample counts, shape=[feature_count]
    double*  sspd                    # Array of feature probabilities, shape=[feature_count]
    int*     leaf_samples            # Array of sample indices if leaf, shape=[feature_count]

cdef class _Tree:
    """
    The Tree object is a binary tree structure constructed by the
    TreeBuilder. The tree structure is used for predictions.
    """

    # Inner structures
    cdef Node*   root                    # Root node

    # Python/C API
    cpdef np.ndarray predict(self, int[:, :] X)
    cpdef int get_node_count(self)
    cpdef int get_exact_node_count(self, double lmbda, int topd, int min_support)
    cpdef int get_semi_random_node_count(self, double lmbda, int topd, int min_support)
    cpdef void print_node_count(self)
    cpdef void print_node_type_count(self, double lmbda, int topd, int min_support)
    cpdef void print_n_samples(self)
    cpdef void print_depth(self)
    cpdef void print_feature(self)
    cpdef void print_value(self)

    # C API
    cdef int _get_node_count(self, Node* node) nogil
    cdef int _get_exact_node_count(self, Node* node, double lmbda,
                                   int topd, int min_support) nogil
    cdef int _get_semi_random_node_count(self, Node* node, double lmbda,
                                         int topd, int min_support) nogil
    cdef void _print_n_samples(self, Node* node) nogil
    cdef void _print_depth(self, Node* node) nogil
    cdef void _print_feature(self, Node* node) nogil
    cdef void _print_value(self, Node* node) nogil

cdef class _TreeBuilder:
    """
    The TreeBuilder recursively builds a Tree object from training samples,
    using a Splitter object for splitting internal nodes and assigning values to leaves.

    This class controls the various stopping criteria and the node splitting
    evaluation order, e.g. depth-first.
    """

    cdef _DataManager manager              # Database manager
    cdef _Splitter    splitter             # Splitter object that chooses the attribute to split on
    cdef int          min_samples_split    # Minimum number of samples in an internal node
    cdef int          min_samples_leaf     # Minimum number of samples in a leaf
    cdef int          max_depth            # Maximal tree depth
    cdef double       tree_budget          # Indistinguishability budget for this tree
    cdef int          topd                 # Number of top semi-random layers
    cdef int          min_support          # Minimum number of samples to be a semi-random node
    cdef int          max_features         # Maximum number of features to consider at each split
    cdef UINT32_t     rand_r_state         # sklearn_rand_r random number state

    # Python API
    cpdef void build(self, _Tree tree)

    # C API
    cdef Node* _build(self, int** X, int* y, int* samples, int n_samples,
                      int* invalid_features, int n_invalid_features,
                      int depth, bint is_left) nogil

    cdef void _set_leaf_node(self, Node** node_ptr, int* y, int* samples, int n_samples,
                        bint is_bottom_leaf) nogil

    cdef void _set_decision_node(self, Node** node_ptr, SplitRecord* split) nogil

    cdef Node* _initialize_node(self, int depth, int is_left) nogil
