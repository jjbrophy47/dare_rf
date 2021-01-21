# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
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

from ._utils cimport dealloc

# constants
cdef INT32_t UNDEF = -1
cdef DTYPE_t UNDEF_LEAF_VAL = 0.5

# =====================================
# TreeBuilder
# =====================================

cdef class _TreeBuilder:
    """
    Build a decision tree in depth-first fashion.
    """

    def __cinit__(self,
                  _DataManager manager,
                  _Splitter    splitter,
                  _Config      config):

        self.manager = manager
        self.splitter = splitter
        self.config = config

    cpdef void build(self, _Tree tree):
        """
        Build a decision tree from the training set (X, y).
        """

        # Data containers
        cdef DTYPE_t** X = NULL
        cdef INT32_t*  y = NULL
        self.manager.get_data(&X, &y)

        cdef SIZE_t* samples
        cdef SIZE_t  n_samples = self.manager.n_samples

        # fill in samples
        samples = <SIZE_t *>malloc(n_samples * sizeof(SIZE_t))
        for i in range(n_samples):
            samples[i] = i

        tree.root = self._build(X, y, samples, n_samples, 0, 0)

    cdef Node* _build(self,
                      DTYPE_t** X,
                      INT32_t*  y,
                      SIZE_t*   samples,
                      SIZE_t    n_samples,
                      SIZE_t    depth,
                      bint      is_left) nogil:
        """
        Build a subtree given a partition of samples.
        """
        cdef Node *node = self._initialize_node(depth, is_left, y, samples, n_samples)

        # class parameters
        cdef SIZE_t    topd = self.config.topd
        cdef SIZE_t    n_total_features = self.manager.n_features
        cdef SIZE_t    n_max_features = self.config.max_features
        cdef UINT32_t* random_state = &self.config.rand_r_state

        # boolean variables
        cdef bint is_bottom_leaf = (depth >= self.config.max_depth)
        cdef bint is_middle_leaf = (n_samples < self.config.min_samples_split or
                                    n_samples < 2 * self.config.min_samples_leaf or
                                    node.n_pos_samples == 0 or
                                    node.n_pos_samples == node.n_samples)

        # result containers
        cdef SplitRecord split
        cdef SIZE_t      n_usable_thresholds = 0

        # printf('\n[B] n_samples: %ld, depth: %ld, is_left: %d\n', n_samples, depth, is_left)

        # leaf node
        if is_bottom_leaf or is_middle_leaf:
            self._set_leaf_node(&node, samples)
            # printf('[B] leaf.value: %.2f\n', node.value)

        # leaf or decision node
        else:

            # randomly select a subset of features to use at this node
            self.splitter.select_features(&node, n_total_features, n_max_features, random_state)
            # printf('[B] no. features: %ld\n', node.n_features)

            # identify and compute metadata for all thresholds for each feature
            n_usable_thresholds = self.splitter.compute_metadata(&node, X, y, samples, n_samples, random_state)
            # printf('[B] computed metadata\n')

            # no usable thresholds, create leaf
            # printf('[B] n_usable_thresholds: %ld\n', n_usable_thresholds)
            if n_usable_thresholds == 0:
                dealloc(node)  # free allocated memory
                self._set_leaf_node(&node, samples)
                # printf('[B] leaf.value: %.2f\n', node.value)

            # decision node
            else:
                # printf('[B] decision node\n')

                # choose a feature / threshold and partition the data
                self.splitter.split_node(&node, X, y, samples, n_samples, topd, random_state, &split)
                # printf('[B] depth: %ld, chosen_feature.index: %ld, chosen_threshold.value: %.2f\n',
                #       node.depth, node.chosen_feature.index, node.chosen_threshold.value)

                # traverse to left and right branches
                node.left = self._build(X, y, split.left_samples, split.n_left_samples, depth + 1, 1)
                node.right = self._build(X, y, split.right_samples, split.n_right_samples, depth + 1, 0)

        return node

    cdef void _set_leaf_node(self,
                             Node**  node_ptr,
                             SIZE_t* samples) nogil:
        """
        Compute leaf value and set all other attributes.
        """
        cdef Node *node = node_ptr[0]

        # cdef SIZE_t i = 0
        # for i in range(node.n_samples):
        #     printf('[B - SLN] samples[%ld]: %ld\n', i, samples[i])

        # set leaf node variables
        node.is_leaf = 1
        node.leaf_samples = samples
        if node.n_samples > 0:
            node.value = node.n_pos_samples / <double> node.n_samples
        else:
            node.value = UNDEF_LEAF_VAL

        # clear decision node variables
        node.features = NULL
        node.n_features = UNDEF
        node.chosen_feature = NULL
        node.chosen_threshold = NULL

        # no children
        node.left = NULL
        node.right = NULL

    cdef Node* _initialize_node(self,
                                SIZE_t   depth,
                                bint     is_left,
                                INT32_t* y,
                                SIZE_t*  samples,
                                SIZE_t   n_samples) nogil:
        """
        Create and initialize a new node.
        """
        # compute number of positive samples
        cdef SIZE_t n_pos_samples = 0
        for i in range(n_samples):
            n_pos_samples += y[samples[i]]

        cdef Node *node = <Node *>malloc(sizeof(Node))
        node.depth = depth
        node.is_left = is_left
        node.n_samples = n_samples
        node.n_pos_samples = n_pos_samples
        node.left = NULL
        node.right = NULL

        node.is_leaf = False
        node.value = UNDEF_LEAF_VAL
        node.leaf_samples = NULL

        node.features = NULL
        node.n_features = 0
        node.chosen_feature = NULL
        node.chosen_threshold = NULL

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

    cpdef np.ndarray predict(self, float[:,:] X):
        """
        Predict probability of positive label for X.
        """

        # In / out
        cdef SIZE_t n_samples = X.shape[0]
        cdef np.ndarray[float] out = np.zeros((n_samples,), dtype=np.float32)

        # Incrementers
        cdef SIZE_t i = 0
        cdef Node* node

        with nogil:

            for i in range(n_samples):
                node = self.root

                while not node.is_leaf:
                    if X[i, node.chosen_feature.index] <= node.chosen_threshold.value:
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
        cdef SIZE_t node_count = self._get_node_count(self.root)
        printf('node_count: %ld\n', node_count)

    cpdef SIZE_t get_node_count(self):
        """
        Get number of nodes total.
        """
        return self._get_node_count(self.root)

    cpdef SIZE_t get_exact_node_count(self, SIZE_t topd):
        """
        Get number of exact nodes in the top d layers.
        """
        return self._get_exact_node_count(self.root, topd)

    cpdef SIZE_t get_random_node_count(self, SIZE_t topd):
        """
        Get number of semi-random nodes in the top d layers.
        """
        return self._get_random_node_count(self.root, topd)

    cpdef void print_node_type_count(self, SIZE_t topd):
        """
        Print the number of exact and semi-random nodes in the
        top d layers.
        """
        cdef SIZE_t exact_node_count = self._get_exact_node_count(self.root, topd)
        cdef SIZE_t random_node_count = self._get_random_node_count(self.root, topd)
        printf('no. exact: %ld, no. semi-random: %ld\n', exact_node_count, random_node_count)

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
    cdef SIZE_t _get_node_count(self, Node* node) nogil:
        if not node:
            return 0
        else:
            return 1 + self._get_node_count(node.left) + self._get_node_count(node.right)

    cdef SIZE_t _get_exact_node_count(self, Node* node, SIZE_t topd) nogil:
        cdef SIZE_t result = 0

        if not node or node.is_leaf:
            result = 0

        else:
            if node.depth >= topd:
                result = 1

            result += self._get_exact_node_count(node.left, topd)
            result += self._get_exact_node_count(node.right, topd)

        return result

    cdef SIZE_t _get_random_node_count(self, Node* node, SIZE_t topd) nogil:
        cdef SIZE_t result = 0

        if not node or node.is_leaf:
            result = 0

        else:
            if node.depth < topd:
                result = 1

            result += self._get_random_node_count(node.left, topd)
            result += self._get_random_node_count(node.right, topd)

        return result

    cdef void _print_n_samples(self, Node* node) nogil:
        if node:
            printf('%ld ', node.n_samples)
            self._print_n_samples(node.left)
            self._print_n_samples(node.right)

    cdef void _print_depth(self, Node* node) nogil:
        if node:
            printf('%ld ', node.depth)
            self._print_depth(node.left)
            self._print_depth(node.right)

    cdef void _print_feature(self, Node* node) nogil:
        if node:
            printf('%ld ', node.chosen_feature.index)
            self._print_feature(node.left)
            self._print_feature(node.right)

    cdef void _print_value(self, Node* node) nogil:
        if node:
            printf('%.3f ', node.value)
            self._print_value(node.left)
            self._print_value(node.right)
