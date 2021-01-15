import numpy as np
cimport numpy as np

from ._manager cimport _DataManager
from ._splitter cimport SplitRecord
from ._splitter cimport _Splitter
from ._utils cimport UINT32_t

"""
Struct that holds high-level information about each node in the tree.
"""
cdef struct Node:

    # Leaf information
    bint     is_leaf                   # Whether this node is a leaf
    double   value                     # Value, if leaf node
    int*     leaf_samples              # Array of sample indices if leaf

    # Decision node information
    Feature**  features                # Array of valid feature pointers
    int        n_features              # Number of features in the array
    Feature*   chosen_feature          # Chosen feature, if decision node
    Threshold* chosen_threshold        # Chosen threshold, if decision node

    # Structure variables
    Node*    left                      # Left child node
    Node*    right                     # Right child node

    # Extra information
    int      n_samples                 # Number of samples in the node
    int      n_pos_samples             # Number of pos samples in the node
    int      depth                     # Depth of node
    bint     is_left                   # Whether this node is a left child

    # int      feature                 # Chosen split feature if decision node
    # int      features_count          # Number of features in the node
    # int      invalid_features_count  # Number of invalid features in the node
    # int*     features                # Array of valid features, shape=[feature_count]
    # int*     invalid_features        # Array of invalid features, shape=[feature_count]
    # int*     left_counts             # Array of left sample counts, shape=[feature_count]
    # int*     left_pos_counts         # Array of left positive sample counts, shape=[feature_count]
    # int*     right_counts            # Array of right sample counts, shape=[feature_count]
    # int*     right_pos_counts        # Array of right positive sample counts, shape=[feature_count]
    # int*     leaf_samples            # Array of sample indices if leaf, shape=[feature_count]

"""
Struct to hold feature information: index ID, candidate split array, etc.
"""
cdef struct Feature:
    int         index                 # Feature index pertaining to the original database
    Threshold** thresholds            # Array of candidate threshold pointers
    # int*        indices               # Array of usable candidate threshold pointers
    int         n_thresholds          # Number of candidate thresholds for this feature
    # int         n_indices             # Number of candidate thresholds to consider

"""
Struct to hold metadata pertaining to a particular feature threshold.
"""
cdef struct Threshold:
    double     value                 # Midway point between two adjacent feature values
    int        n_v1_samples          # Number of samples for feature 1
    int        n_v1_pos_samples      # Number of positive samples for feature 1
    int        n_v2_samples          # Number of samples for feature 2
    int        n_v2_pos_samples      # Number of positive samples for feature 2
    int        n_left_samples        # Number of samples for the left branch
    int        n_left_pos_samples    # Number of positive samples for the left branch
    int        n_right_samples       # Number of samples for the right branch
    int        n_right_pos_samples   # Number of positive samples for the right branch

cdef class _Tree:
    """
    The Tree object is a binary tree structure constructed by the
    TreeBuilder. The tree structure is used for predictions.
    """

    # Inner structures
    cdef Node*   root                    # Root node

    # Python/C API
    cpdef np.ndarray predict(self, double[:, :] X)
    cpdef int get_node_count(self)
    cpdef int get_exact_node_count(self, int topd)
    cpdef int get_random_node_count(self, int topd)
    cpdef void print_node_count(self)
    cpdef void print_node_type_count(self, int topd)
    cpdef void print_n_samples(self)
    cpdef void print_depth(self)
    cpdef void print_feature(self)
    cpdef void print_value(self)

    # C API
    cdef int _get_node_count(self, Node* node) nogil
    cdef int _get_exact_node_count(self, Node* node, int topd) nogil
    cdef int _get_random_node_count(self, Node* node, int topd) nogil
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
    cdef int          topd                 # Number of top semi-random layers
    cdef int          k                    # Number of candidate thresholds to consider for each feature
    cdef int          max_features         # Maximum number of features to consider at each split
    cdef UINT32_t     rand_r_state         # sklearn_rand_r random number state
    cdef bint         sim_mode             # Activates simulation mode
    cdef int          sim_depth            # Depth of previous operation completion
    cdef int*         features             # Features to use whn retraining a node

    # Python API
    cpdef void set_sim_mode(self, bint sim_mode)
    cpdef void build(self, _Tree tree)

    # C API
    cdef Node* _build(self, double** X, int* y, int* samples, int n_samples, int depth, bint is_left) nogil
    cdef void  _set_leaf_node(self, Node** node_ptr, int* samples) nogil
    cdef Node* _initialize_node(self, int depth, int is_left, int* y, int* samples, int n_samples) nogil
