import numpy as np
cimport numpy as np

from ._manager cimport _DataManager
from ._splitter cimport SplitRecord
from ._splitter cimport _Splitter
from ._config cimport _Config
from ._utils cimport DTYPE_t
from ._utils cimport SIZE_t
from ._utils cimport INT32_t
from ._utils cimport UINT32_t

"""
Struct that holds high-level information about each node in the tree.
"""
cdef struct Node:

    # Leaf information
    bint     is_leaf                   # Whether this node is a leaf
    DTYPE_t  value                     # Value, if leaf node
    SIZE_t*  leaf_samples              # Array of sample indices if leaf

    # Decision node information
    Feature**  features                # Array of valid feature pointers
    SIZE_t     n_features              # Number of features in the array
    Feature*   chosen_feature          # Chosen feature, if decision node
    Threshold* chosen_threshold        # Chosen threshold, if decision node

    # Structure variables
    Node*    left                      # Left child node
    Node*    right                     # Right child node

    # Extra information
    SIZE_t   n_samples                 # Number of samples in the node
    SIZE_t   n_pos_samples             # Number of pos samples in the node
    SIZE_t   depth                     # Depth of node
    bint     is_left                   # Whether this node is a left child

"""
Struct to hold feature information: index ID, candidate split array, etc.
"""
cdef struct Feature:
    SIZE_t      index                 # Feature index pertaining to the original database
    Threshold** thresholds            # Array of candidate threshold pointers
    SIZE_t      n_thresholds          # Number of candidate thresholds for this feature

"""
Struct to hold metadata pertaining to a particular feature threshold.
"""
cdef struct Threshold:
    DTYPE_t    value                 # Midway point between two adjacent feature values
    DTYPE_t    v1                    # Lower value of adjacent values
    DTYPE_t    v2                    # Upper value of adjacent values
    SIZE_t     n_v1_samples          # Number of samples for value 1
    SIZE_t     n_v1_pos_samples      # Number of positive samples for value 1
    SIZE_t     n_v2_samples          # Number of samples for value 2
    SIZE_t     n_v2_pos_samples      # Number of positive samples for value 2
    SIZE_t     n_left_samples        # Number of samples for the left branch
    SIZE_t     n_left_pos_samples    # Number of positive samples for the left branch
    SIZE_t     n_right_samples       # Number of samples for the right branch
    SIZE_t     n_right_pos_samples   # Number of positive samples for the right branch

cdef class _Tree:
    """
    The Tree object is a binary tree structure constructed by the
    TreeBuilder. The tree structure is used for predictions.
    """

    # Inner structures
    cdef Node*   root                    # Root node

    # Python/C API
    cpdef np.ndarray predict(self, float[:, :] X)
    cpdef SIZE_t get_node_count(self)
    cpdef SIZE_t get_exact_node_count(self, SIZE_t topd)
    cpdef SIZE_t get_random_node_count(self, SIZE_t topd)
    cpdef void print_node_count(self)
    cpdef void print_node_type_count(self, SIZE_t topd)
    cpdef void print_n_samples(self)
    cpdef void print_depth(self)
    cpdef void print_feature(self)
    cpdef void print_value(self)

    # C API
    cdef SIZE_t _get_node_count(self, Node* node) nogil
    cdef SIZE_t _get_exact_node_count(self, Node* node, SIZE_t topd) nogil
    cdef SIZE_t _get_random_node_count(self, Node* node, SIZE_t topd) nogil
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
    cdef _Config      config               # Configuration object holding training parameters

    # Python API
    cpdef void build(self, _Tree tree)

    # C API
    cdef Node* _build(self,
                      DTYPE_t** X,
                      INT32_t*  y,
                      SIZE_t*   samples,
                      SIZE_t    n_samples,
                      SIZE_t    depth,
                      bint      is_left) nogil

    cdef void  _set_leaf_node(self,
                              Node**  node_ptr,
                              SIZE_t* samples) nogil

    cdef Node* _initialize_node(self,
                                SIZE_t   depth,
                                bint     is_left,
                                INT32_t* y,
                                SIZE_t*  samples,
                                SIZE_t   n_samples) nogil
