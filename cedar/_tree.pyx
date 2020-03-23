"""
CeDAR binary tree implementation; only supports binary attributes and a binary label.
Represenation is a number of parllel arrays.
Adapted from: https://github.com/scikit-learn/scikit-learn/blob/b194674c42d54b26137a456c510c5fdba1ba23e0/sklearn/tree/_tree.pyx
"""
cimport cython

from cpython cimport Py_INCREF
from cpython.ref cimport PyObject

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport convert_int_ndarray
from ._utils cimport set_srand
from ._utils cimport dealloc

# constants
from numpy import int32 as INT

cdef int UNDEF = -1

# =====================================
# TreeBuilder
# =====================================

cdef class _TreeBuilder:
    """
    Build a decision tree in depth-first fashion.
    """

    def __cinit__(self, _DataManager manager, _Splitter splitter, int min_samples_split,
                  int min_samples_leaf, int max_depth, int random_state):
        self.manager = manager
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = 1000 if max_depth == -1 else max_depth
        self.random_state = random_state
        set_srand(self.random_state)

    cpdef void build(self, _Tree tree):
        """
        Build a decision tree from the training set (X, y).
        """

        # Data containers
        cdef int** X = NULL
        cdef int* y = NULL
        self.manager.get_data(&X, &y)

        cdef int* samples
        cdef int n_samples = self.manager.n_samples
        cdef int* features = tree.feature_indices
        cdef int n_features = tree.n_feature_indices

        # fill in samples
        samples = <int *>malloc(n_samples * sizeof(int))
        for i in range(n_samples):
            samples[i] = i

        tree.root = self._build(X, y, samples, n_samples, features, n_features, 0, 0, 1.0)

    cdef Node* _build(self, int** X, int* y, int* samples, int n_samples,
                      int* features, int n_features,
                      int depth, bint is_left, double parent_p) nogil:
        """
        Builds a subtree given samples to train from.
        """

        cdef SplitRecord split
        cdef int result

        cdef Node *node = <Node *>malloc(sizeof(Node))
        node.depth = depth
        node.is_left = is_left

        # printf('\n(%d, %d, %.7f, %d, %d)\n', depth, is_left, parent_p, n_samples, n_features)

        cdef bint is_bottom_leaf = (depth >= self.max_depth or n_features < 1)
        cdef bint is_middle_leaf = (n_samples < self.min_samples_split or
                                    n_samples < 2 * self.min_samples_leaf)

        if is_bottom_leaf:
            # printf('bottom leaf!\n')
            self._set_leaf_node(&node, y, samples, n_samples, is_bottom_leaf)

        else:
            # printf('compute splits\n')
            self.splitter.compute_splits(&node, X, y, samples, n_samples, features,
                                         n_features)

            if not is_middle_leaf:
                # printf('split node\n')
                is_middle_leaf = self.splitter.split_node(node, X, y, samples, n_samples,
                                                       parent_p, &split)
                # printf('result: %d\n', is_middle_leaf)

            if is_middle_leaf:
                # printf('middle leaf\n')
                self._set_leaf_node(&node, y, samples, n_samples, 0)

            else:
                free(samples)
                self._set_decision_node(&node, &split)

                node.left = self._build(X, y, split.left_indices, split.left_count,
                                        split.features, split.features_count,
                                        depth + 1, 1, node.p)

                node.right = self._build(X, y, split.right_indices, split.right_count,
                                         split.features, split.features_count,
                                         depth + 1, 0, node.p)

        return node

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _set_leaf_node(self, Node** node_ptr, int* y, int* samples, int n_samples,
                        bint is_bottom_leaf) nogil:
        """
        Compute leaf value and set all other attributes.
        """
        cdef int pos_count = 0
        cdef int i

        cdef Node *node = node_ptr[0]

        if is_bottom_leaf:
            node.p = UNDEF
            node.feature = UNDEF
            node.features_count = UNDEF
            node.features = NULL
            node.left_counts = NULL
            node.left_pos_counts = NULL
            node.right_counts = NULL
            node.right_pos_counts = NULL

            for i in range(n_samples):
                pos_count += y[samples[i]]

            node.count = n_samples
            node.pos_count = pos_count

        node.is_leaf = 1
        node.value = pos_count / <double> n_samples
        node.leaf_samples = samples

        node.left = NULL
        node.right = NULL

    cdef void _set_decision_node(self, Node** node_ptr, SplitRecord* split) nogil:
        """
        Set all attributes for decision node.
        """
        cdef Node* node = node_ptr[0]

        node.is_leaf = 0
        node.value = UNDEF
        node.leaf_samples = NULL

        node.p = split.p
        node.feature = split.feature


# =====================================
# Tree
# =====================================

cdef class _Tree:

    def __cinit__(self, np.ndarray features):
        """
        Constructor.
        """

        # features this tree is built on
        self.n_feature_indices = features.shape[0]
        self.feature_indices = convert_int_ndarray(features)

        # internal data structures
        self.node_count = 0
        self.root = NULL

    def __dealloc__(self):
        """
        Destructor.
        """
        if self.root:
            dealloc(self.root)
            free(self.root)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray predict(self, int[:,:] X):
        """
        Predict probability of positive label for X.
        """

        # In / out
        cdef int n_samples = X.shape[0]
        cdef np.ndarray[double] out = np.zeros((n_samples,), dtype=np.double)

        # Incrementers
        cdef int i = 0
        cdef Node* node

        with nogil:

            for i in range(n_samples):
                node = self.root

                while not node.is_leaf:
                    if X[i, node.feature] == 1:
                        node = node.left
                    else:
                        node = node.right

                out[i] = node.value

        return out

    cpdef void print_depth(self):
        printf('depth: [ ')
        self._print_depth(self.root)
        printf(']\n')

    cpdef void print_node_count(self):
        cdef int node_count = self._get_node_count(self.root)
        printf('node_count: %d\n', node_count)

    # private
    cdef void _print_depth(self, Node* node) nogil:
        """
        Print depth of each node.
        """
        if node:
            printf('%d ', node.depth)
            self._print_depth(node.left)
            self._print_depth(node.right)

    cdef int _get_node_count(self, Node* node) nogil:
        """
        Get number of nodes total.
        """
        if not node:
            return 0
        else:
            return 1 + self._get_node_count(node.left) + self._get_node_count(node.right)
