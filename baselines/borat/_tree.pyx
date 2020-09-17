"""
Tree and tree builder objects.
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
from ._utils cimport RAND_R_MAX

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

    def __cinit__(self,
                  _DataManager manager,
                  _Splitter splitter,
                  int min_samples_split,
                  int min_samples_leaf,
                  int max_depth,
                  int topd,
                  int min_support,
                  int max_features,
                  object random_state):

        self.manager = manager
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.topd = topd
        self.min_support = min_support
        self.max_features = max_features
        self.rand_r_state = random_state.randint(0, RAND_R_MAX)

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
        cdef int* invalid_features = NULL
        cdef int  n_invalid_features = 0

        # fill in samples
        samples = <int *>malloc(n_samples * sizeof(int))
        for i in range(n_samples):
            samples[i] = i

        tree.root = self._build(X, y, samples, n_samples,
                                invalid_features, n_invalid_features, 0, 0)

    @cython.cdivision(True)
    cdef Node* _build(self, int** X, int* y, int* samples, int n_samples,
                      int* invalid_features, int n_invalid_features,
                      int depth, bint is_left) nogil:
        """
        Builds a subtree given samples to train from.
        """
        cdef SplitRecord split
        cdef int result

        cdef int topd = self.topd
        cdef int min_support = self.min_support
        cdef UINT32_t* random_state = &self.rand_r_state
        cdef int n_features = self.manager.n_features
        cdef int n_max_features = self.max_features

        cdef Node *node = self._initialize_node(depth, is_left)

        cdef bint is_bottom_leaf = (depth >= self.max_depth or
                                    n_features - n_invalid_features < 1)
        cdef bint is_middle_leaf = (n_samples < self.min_samples_split or
                                    n_samples < 2 * self.min_samples_leaf)

        if is_bottom_leaf:
            self._set_leaf_node(&node, y, samples, n_samples, is_bottom_leaf)
            free(invalid_features)

        else:
            self.splitter.select_features(&node, n_features, n_max_features,
                                          invalid_features, n_invalid_features,
                                          random_state)
            self.splitter.compute_splits(&node, X, y, samples, n_samples)

            if not is_middle_leaf:
                is_middle_leaf = self.splitter.split_node(node, X, y, samples, n_samples,
                                                          topd, min_support, random_state,
                                                          &split)

            if is_middle_leaf:
                self._set_leaf_node(&node, y, samples, n_samples, 0)

            else:
                free(samples)
                self._set_decision_node(&node, &split)

                node.left = self._build(X, y, split.left_indices, split.left_count,
                                        split.invalid_left_features, split.invalid_features_count,
                                        depth + 1, 1)

                node.right = self._build(X, y, split.right_indices, split.right_count,
                                         split.invalid_right_features, split.invalid_features_count,
                                         depth + 1, 0)

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
            node.invalid_features_count = UNDEF
            node.features = NULL
            node.invalid_features = NULL
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
        node.feature = split.feature

    cdef Node* _initialize_node(self, int depth, int is_left) nogil:
        """
        Create and initialize a new node.
        """
        cdef Node *node = <Node *>malloc(sizeof(Node))
        node.depth = depth
        node.is_left = is_left
        return node


# =====================================
# Tree
# =====================================

cdef class _Tree:

    def __cinit__(self):
        """
        Constructor.
        """
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

    # tree information
    cpdef void print_node_count(self):
        """
        Get number of nodes total.
        """
        cdef int node_count = self._get_node_count(self.root)
        printf('node_count: %d\n', node_count)

    cpdef int get_node_count(self):
        """
        Get number of nodes total.
        """
        return self._get_node_count(self.root)

    cpdef int get_exact_node_count(self, int topd, int min_support):
        """
        Get number of exact nodes in the top d layers.
        """
        return self._get_exact_node_count(self.root, topd, min_support)

    cpdef int get_random_node_count(self, int topd, int min_support):
        """
        Get number of semi-random nodes in the top d layers.
        """
        return self._get_random_node_count(self.root, topd, min_support)

    cpdef void print_node_type_count(self, int topd, int min_support):
        """
        Print the number of exact and semi-random nodes in the
        top d layers.
        """
        cdef int exact_node_count = self._get_exact_node_count(self.root, topd, min_support)
        cdef int random_node_count = self._get_random_node_count(self.root, topd, min_support)
        printf('no. exact: %d, no. semi-random: %d\n', exact_node_count, random_node_count)

    # node information
    cpdef void print_n_samples(self):
        """
        Get number of samples for each node.
        """
        printf('n_samples: [ ')
        self._print_n_samples(self.root)
        printf(']\n')

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

    cdef int _get_exact_node_count(self, Node* node, int topd, int min_support) nogil:
        cdef int result = 0

        if not node or node.is_leaf:
            result = 0

        else:
            if node.depth < topd:
                result = 1

            result += self._get_exact_node_count(node.left, topd, min_support)
            result += self._get_exact_node_count(node.right, topd, min_support)

        return result

    cdef int _get_random_node_count(self, Node* node, int topd, int min_support) nogil:
        cdef int result = 0

        if not node or node.is_leaf:
            result = 0

        else:
            if node.depth >= topd and node.count >= min_support:
                result = 1

            result += self._get_random_node_count(node.left, topd, min_support)
            result += self._get_random_node_count(node.right, topd, min_support)

        return result

    cdef void _print_n_samples(self, Node* node) nogil:
        if node:
            printf('%d ', node.count)
            self._print_n_samples(node.left)
            self._print_n_samples(node.right)

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
