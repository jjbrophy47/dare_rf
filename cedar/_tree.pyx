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

        # Parameters
        cdef _DataManager manager = self.manager

        # get data
        cdef int** X = NULL
        cdef int* y = NULL

        cdef int* samples
        cdef int n_samples = manager.n_samples
        cdef int* features = tree.feature_indices
        cdef int n_features = tree.n_feature_indices

        manager.get_data(&X, &y)

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

        # printf('(%d, %d, %.7f, %d, %d)\n', depth, is_left, parent_p, n_samples, n_features)

        cdef bint is_leaf = (depth >= self.max_depth or
                             n_samples < self.min_samples_split or
                             n_samples < 2 * self.min_samples_leaf or
                             n_features <= 1)

        if not is_leaf:
            result = self.splitter.node_split(X, y, samples, n_samples, features,
                                              n_features, parent_p, &split)
            if result == -1:
                is_leaf = 1

        if is_leaf:
            self._set_leaf_node(&node, y, samples, n_samples)

        else:
            free(samples)
            self._set_decision_node(&node, &split)

            node.left = self._build(X, y, split.left_indices, split.left_count,
                                    split.valid_features, split.feature_count,
                                    depth + 1, 1, node.p)

            node.right = self._build(X, y, split.right_indices, split.right_count,
                                     split.valid_features, split.feature_count,
                                     depth + 1, 0, node.p)

        return node

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _set_leaf_node(self, Node** node_ptr, int* y, int* samples, int n_samples) nogil:
        """
        Compute leaf value and set all other attributes.
        """
        cdef int pos_count = 0
        cdef int i

        cdef Node *node = node_ptr[0]

        for i in range(n_samples):
            pos_count += y[samples[i]]

        node.count = n_samples
        node.pos_count = pos_count

        node.is_leaf = 1
        node.value = pos_count / <double> n_samples
        node.leaf_samples = samples

        node.p = UNDEF
        node.feature = UNDEF
        node.feature_count = UNDEF
        node.valid_features = NULL
        node.left_counts = NULL
        node.left_pos_counts = NULL
        node.right_counts = NULL
        node.right_pos_counts = NULL

        node.left = NULL
        node.right = NULL

    cdef void _set_decision_node(self, Node** node_ptr, SplitRecord* split) nogil:
        """
        Set all attributes for decision node.
        """
        cdef Node* node = node_ptr[0]

        node.count = split.count
        node.pos_count = split.pos_count

        node.is_leaf = 0
        node.value = UNDEF
        node.leaf_samples = NULL

        node.p = split.p
        node.feature = split.feature
        node.feature_count = split.feature_count
        node.valid_features = split.valid_features
        node.left_counts = split.left_counts
        node.left_pos_counts = split.left_pos_counts
        node.right_counts = split.right_counts
        node.right_pos_counts = split.right_pos_counts


# =====================================
# Tree
# =====================================

# TODO: recursively travese the tree and return properties in a depth-first manner
cdef class _Tree:

    # property n_nodes:
    #     def __get__(self):
    #         return self.node_count - self.vacant_ids.top

    # property feature_indices:
    #     def __get__(self):
    #         return self._get_int_ndarray(self.feature_indices, self.n_feature_indices)

    # property values:
    #     def __get__(self):
    #         return self._get_double_ndarray(self.values, self.node_count)

    # property p:
    #     def __get__(self):
    #         return self._get_double_ndarray(self.p, self.node_count)

    # property chosen_features:
    #     def __get__(self):
    #         return self._get_int_ndarray(self.chosen_features, self.node_count)

    # property left_children:
    #     def __get__(self):
    #         return self._get_int_ndarray(self.left_children, self.node_count)

    # property right_children:
    #     def __get__(self):
    #         return self._get_int_ndarray(self.right_children, self.node_count)

    # property depth:
    #     def __get__(self):
    #         cdef int  node_count = self._get_node_count(self.root)
    #         cdef int *depth = <int *>malloc(node_count * sizeof(int))
    #         cdef int  depth_count = 0
    #         self._get_depth(self.root, &depth, &depth_count)
    #         return self._get_int_ndarray(depth, depth_count)

    # metadata
    # property counts:
    #     def __get__(self):
    #         return self._get_int_ndarray(self.count, self.node_count)

    # property pos_counts:
    #     def __get__(self):
    #         return self._get_int_ndarray(self.pos_count, self.node_count)

    # property feature_counts:
    #     def __get__(self):
    #         return self._get_int_ndarray(self.feature_count, self.node_count)

    # cpdef np.ndarray _get_left_counts(self, node_id):
    #     return self._get_int_ndarray(self.left_counts[node_id], self.feature_count[node_id])

    # cpdef np.ndarray _get_left_pos_counts(self, node_id):
    #     return self._get_int_ndarray(self.left_pos_counts[node_id], self.feature_count[node_id])

    # cpdef np.ndarray _get_right_counts(self, node_id):
    #     return self._get_int_ndarray(self.right_counts[node_id], self.feature_count[node_id])

    # cpdef np.ndarray _get_right_pos_counts(self, node_id):
    #     return self._get_int_ndarray(self.right_pos_counts[node_id], self.feature_count[node_id])

    # cpdef np.ndarray _get_features(self, node_id):
    #     return self._get_int_ndarray(self.features[node_id], self.feature_count[node_id])

    # cpdef np.ndarray _get_leaf_samples(self, node_id):
    #     return self._get_int_ndarray(self.leaf_samples[node_id], self.count[node_id])

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
        free(self.feature_indices)
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

    # cdef np.ndarray _get_double_ndarray(self, double *data, int n_elem):
    #     """
    #     Wraps value as a 1-d NumPy array.
    #     The array keeps a reference to this Tree, which manages the underlying memory.
    #     """
    #     cdef np.npy_intp shape[1]
    #     shape[0] = n_elem
    #     cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, data)
    #     Py_INCREF(self)
    #     arr.base = <PyObject*> self
    #     return arr

    # cdef np.ndarray _get_int_ndarray(self, int *data, int n_elem):
    #     """
    #     Wraps value as a 1-d NumPy array.
    #     The array keeps a reference to this Tree, which manages the underlying memory.
    #     """
    #     cdef np.npy_intp shape[1]
    #     shape[0] = n_elem
    #     cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, data)
    #     Py_INCREF(self)
    #     arr.base = <PyObject*> self
    #     return arr
