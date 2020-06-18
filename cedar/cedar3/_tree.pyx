"""
CeDAR
"""
cimport cython

from cpython cimport Py_INCREF
from cpython.ref cimport PyObject

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf
from libc.math cimport pow

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport convert_int_ndarray
from ._utils cimport dealloc

# constants
cdef int UNDEF = -1
cdef double UNDEF_LEAF_VAL = 0.5

# =====================================
# TreeBuilder
# =====================================

cdef class _TreeBuilder:
    """
    Build a decision tree in depth-first fashion.
    """

    def __cinit__(self, _DataManager manager, _Splitter splitter, int min_samples_split,
                  int min_samples_leaf, int max_depth, double tree_budget):
        self.manager = manager
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.tree_budget = tree_budget
        # self.topd = topd

    cpdef void build(self, _Tree tree):
        """
        Build a decision tree from the training set (X, y).
        """

        # Data containers
        cdef int** X = NULL
        cdef int* y = NULL
        self.manager.get_data(&X, &y)

        cdef int* samples
        cdef int  n_samples = self.manager.n_samples
        cdef int* features = tree.feature_indices
        cdef int  n_features = tree.n_feature_indices

        # fill in samples
        samples = <int *>malloc(n_samples * sizeof(int))
        for i in range(n_samples):
            samples[i] = i

        tree.layer_budget = <double *>malloc(sizeof(double) * self.max_depth)
        tree.root = self._build(X, y, samples, n_samples, features, n_features, 0, 0,
                                &tree.layer_budget)

    @cython.cdivision(True)
    cdef Node* _build(self, int** X, int* y, int* samples, int n_samples,
                      int* features, int n_features,
                      int depth, bint is_left, double** layer_budget_ptr) nogil:
        """
        Builds a subtree given samples to train from.
        """

        cdef SplitRecord split
        cdef int result

        # allocate budget for this layer
        # if depth <= self.topd - 1:
        #     layer_budget_ptr[0][0] = (self.tree_budget / self.max_depth) * self.topd
        # else:
        layer_budget_ptr[0][depth] = self.tree_budget / self.max_depth

        cdef Node *node = <Node *>malloc(sizeof(Node))
        node.layer_budget_ptr = layer_budget_ptr
        node.divergence = 0
        node.depth = depth
        node.is_left = is_left

        cdef bint is_bottom_leaf = (depth >= self.max_depth or n_features < 1)
        cdef bint is_middle_leaf = (n_samples < self.min_samples_split or
                                    n_samples < 2 * self.min_samples_leaf)

        # printf('\n(depth, is_left, is_leaf, n_samples): (%d, %d, %d, %d)\n', node.depth, node.is_left, node.is_leaf, n_samples)

        if is_bottom_leaf:
            # printf('bottom leaf\n')
            self._set_leaf_node(&node, y, samples, n_samples, is_bottom_leaf)

        else:
            # printf('compute splits...\n')
            self.splitter.compute_splits(&node, X, y, samples, n_samples, features,
                                         n_features)

            if not is_middle_leaf:
                # printf('split node\n')
                is_middle_leaf = self.splitter.split_node(node, X, y, samples, n_samples,
                                                          &split)

            if is_middle_leaf:
                # printf('leaf node\n')
                self._set_leaf_node(&node, y, samples, n_samples, 0)

            else:
                # printf('free samples\n')
                free(samples)
                # printf('done freeing samples\n')
                self._set_decision_node(&node, &split)

                node.left = self._build(X, y, split.left_indices, split.left_count,
                                        split.left_features, split.features_count,
                                        depth + 1, 1, layer_budget_ptr)

                node.right = self._build(X, y, split.right_indices, split.right_count,
                                         split.right_features, split.features_count,
                                         depth + 1, 0, layer_budget_ptr)

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
        node.leaf_samples = samples
        if node.count > 0:
            node.value = node.pos_count / <double> node.count
        else:
            node.value = UNDEF_LEAF_VAL

        node.sspd = NULL
        node.feature = UNDEF

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

        node.sspd = split.sspd
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
        self.layer_budget = NULL
        self.root = NULL

    def __dealloc__(self):
        """
        Destructor.
        """
        if self.layer_budget:
            free(self.layer_budget)
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

    # tree information
    cpdef void print_node_count(self):
        """
        Get number of nodes total.
        """
        cdef int node_count = self._get_node_count(self.root)
        printf('node_count: %d\n', node_count)

    # node information
    cpdef void print_depth(self):
        """
        Print depth of each node.
        """
        printf('depth: [ ')
        self._print_depth(self.root)
        printf(']\n')

    cpdef void print_feature(self):
        """
        Print depth of each node.
        """
        printf('feature: [ ')
        self._print_feature(self.root)
        printf(']\n')

    cpdef void print_value(self):
        """
        Print value of each node.
        """
        printf('value: [ ')
        self._print_value(self.root)
        printf(']\n')

    # private
    cdef int _get_node_count(self, Node* node) nogil:
        if not node:
            return 0
        else:
            return 1 + self._get_node_count(node.left) + self._get_node_count(node.right)

    cdef void _print_depth(self, Node* node) nogil:
        if node:
            printf('%d ', node.depth)
            self._print_depth(node.left)
            self._print_depth(node.right)

    cdef void _print_feature(self, Node* node) nogil:
        if node:
            printf('%d ', node.feature)
            self._print_feature(node.left)
            self._print_feature(node.right)

    cdef void _print_value(self, Node* node) nogil:
        if node:
            printf('%.3f ', node.value)
            self._print_value(node.left)
            self._print_value(node.right)
