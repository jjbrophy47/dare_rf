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

from ._utils cimport split_samples
from ._utils cimport copy_indices
from ._utils cimport create_intlist
from ._utils cimport free_intlist
from ._utils cimport dealloc

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

        # create list of sample indices
        cdef IntList* samples = create_intlist(self.manager.n_samples, 1)
        for i in range(samples.n):
            samples.arr[i] = i

        # initialize container for constant features
        cdef IntList* constant_features = create_intlist(self.manager.n_features, 0)

        tree.root = self._build(X, y, samples, constant_features, 0, 0)

    cdef Node* _build(self,
                      DTYPE_t** X,
                      INT32_t*  y,
                      IntList*  samples,
                      IntList*  constant_features,
                      SIZE_t    depth,
                      bint      is_left) nogil:
        """
        Build a subtree given a partition of samples.
        """

        # create node
        cdef Node *node = self.initialize_node(depth, is_left, y, samples, constant_features)

        # data variables
        cdef SIZE_t n_total_features = self.manager.n_features

        # boolean variables
        cdef bint is_bottom_leaf = (depth >= self.config.max_depth)
        cdef bint is_middle_leaf = (samples.n < self.config.min_samples_split or
                                    samples.n < 2 * self.config.min_samples_leaf or
                                    node.n_pos_samples == 0 or
                                    node.n_pos_samples == node.n_samples)

        # result containers
        cdef SplitRecord split
        cdef SIZE_t      n_usable_thresholds = 0

        # printf('\n[B] samples.n: %ld, depth: %ld, is_left: %d\n', samples.n, depth, is_left)

        # leaf node
        if is_bottom_leaf or is_middle_leaf:
            # printf('[B] bottom / middle leaf\n')
            self.set_leaf_node(node, samples)
            # printf('[B] leaf.value: %.2f\n', node.value)

        # leaf or decision node
        else:

            # select a threshold to to split the samples
            # printf('[B] select threshold\n')
            n_usable_thresholds = self.splitter.select_threshold(node, X, y, samples, n_total_features)
            # printf('[B] no_usable_thresholds: %ld\n', n_usable_thresholds)

            # no usable thresholds, create leaf
            if n_usable_thresholds == 0:
                # printf('no usable thresholds\n')
                dealloc(node)  # free allocated memory
                self.set_leaf_node(node, samples)
                # printf('[B] leaf.value: %.2f\n', node.value)

            # decision node
            else:
                # printf('[B] split samples\n')
                split_samples(node, X, y, samples, &split, 1)
                # printf('[B] depth: %ld, chosen_feature.index: %ld, chosen_threshold.value: %.2f\n',
                #       node.depth, node.chosen_feature.index, node.chosen_threshold.value)

                # printf('[B] split.left_samples.n: %ld, split.right_samples.n: %ld\n',
                #        split.left_samples.n, split.right_samples.n)

                # traverse to left and right branches
                node.left = self._build(X, y, split.left_samples, split.left_constant_features, depth + 1, 1)
                node.right = self._build(X, y, split.right_samples, split.right_constant_features, depth + 1, 0)

        return node

    cdef void set_leaf_node(self,
                            Node*    node,
                            IntList* samples) nogil:
        """
        Compute leaf value and set all other attributes.
        """

        # set leaf node properties
        node.is_leaf = True
        node.leaf_samples = copy_indices(samples.arr, samples.n)
        node.value = node.n_pos_samples / <double> node.n_samples

        # set greedy node properties
        node.features = NULL
        node.n_features = 0

        # set greedy / random node properties
        if node.constant_features != NULL:
            free_intlist(node.constant_features)
        node.constant_features = NULL
        node.chosen_feature = NULL
        node.chosen_threshold = NULL

        # no children
        node.left = NULL
        node.right = NULL

        # free samples
        free_intlist(samples)

    cdef Node* initialize_node(self,
                               SIZE_t   depth,
                               bint     is_left,
                               INT32_t* y,
                               IntList* samples,
                               IntList* constant_features) nogil:
        """
        Create and initialize a new node.
        """

        # compute number of positive samples
        cdef SIZE_t n_pos_samples = 0
        for i in range(samples.n):
            n_pos_samples += y[samples.arr[i]]

        # create node
        cdef Node *node = <Node *>malloc(sizeof(Node))

        # initialize mandatory properties
        node.n_samples = samples.n
        node.n_pos_samples = n_pos_samples
        node.depth = depth
        node.is_left = is_left
        node.left = NULL
        node.right = NULL

        # initialize greedy node properties
        node.features = NULL
        node.n_features = 0

        # initialize greedy / random node properties
        node.constant_features = constant_features
        node.chosen_feature = NULL
        node.chosen_threshold = NULL

        # initialize leaf-specific properties
        node.is_leaf = False
        node.value = UNDEF_LEAF_VAL
        node.leaf_samples = NULL

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
        # printf('deallocing tree\n')
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
    cpdef SIZE_t get_node_count(self):
        """
        Get number of nodes total.
        """
        return self._get_node_count(self.root)

    cpdef SIZE_t get_random_node_count(self, SIZE_t topd):
        """
        Count number of exact nodes in the top d layers.
        """
        return self._get_random_node_count(self.root, topd)

    cpdef SIZE_t get_greedy_node_count(self, SIZE_t topd):
        """
        Count number of greedy nodes.
        """
        return self._get_greedy_node_count(self.root, topd)

    # private
    cdef SIZE_t _get_node_count(self, Node* node) nogil:
        """
        Count total no. of nodes in the tree.
        """
        if not node:
            return 0
        else:
            return 1 + self._get_node_count(node.left) + self._get_node_count(node.right)

    cdef SIZE_t _get_random_node_count(self, Node* node, SIZE_t topd) nogil:
        """
        Count no. random nodes in the tree.
        """
        cdef SIZE_t result = 0

        if not node or node.is_leaf:
            result = 0

        else:
            if node.depth < topd:
                result = 1

            result += self._get_random_node_count(node.left, topd)
            result += self._get_random_node_count(node.right, topd)

        return result

    cdef SIZE_t _get_greedy_node_count(self, Node* node, SIZE_t topd) nogil:
        """
        Count no. greedy nodes in the tree.
        """
        cdef SIZE_t result = 0

        if not node or node.is_leaf:
            result = 0

        else:
            if node.depth >= topd:
                result = 1

            result += self._get_greedy_node_count(node.left, topd)
            result += self._get_greedy_node_count(node.right, topd)

        return result
