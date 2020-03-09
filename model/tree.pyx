"""
CeDAR binary tree implementation; only support binary attributes and a binary label.
Represenation is a number of parllel arrays.
Adapted from: https://github.com/scikit-learn/scikit-learn/blob/b194674c42d54b26137a456c510c5fdba1ba23e0/sklearn/tree/_tree.pyx
"""

# imports
cimport cython

from cpython cimport Py_INCREF
from cpython.ref cimport PyObject

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX
from libc.stdio cimport printf

import numpy as np
cimport numpy as np
np.import_array()

from utils cimport Stack
from utils cimport StackRecord
# from utils cimport safe_realloc
# from utils cimport sizet_ptr_to_ndarray

# constants
from numpy import int32 as INT
from numpy import float32 as FLOAT
from numpy import float64 as DOUBLE

# cdef double INFINITY = np.inf
# cdef double EPSILON = np.finfo('double').eps

cdef SIZE_t _TREE_LEAF = -1
cdef SIZE_t _TREE_UNDEFINED = -2
cdef SIZE_t INITIAL_STACK_SIZE = 10

# # Repeat struct definition for numpy
# NODE_DTYPE = np.dtype({
#     'names': ['left_child', 'right_child', 'feature', 'impurity', 'n_node_samples'],
#     'formats': [np.intp, np.intp, np.intp, np.float64, np.intp],
#     'offsets': [
#         <Py_ssize_t> &(<Node*> NULL).left_child,
#         <Py_ssize_t> &(<Node*> NULL).right_child,
#         <Py_ssize_t> &(<Node*> NULL).feature,
#         <Py_ssize_t> &(<Node*> NULL).impurity,
#         <Py_ssize_t> &(<Node*> NULL).n_node_samples,
#     ]
# })

# =====================================
# TreeBuilder
# =====================================

cdef class TreeBuilder:
    """
    Interface for different tree building strategies.
    """

    cpdef void build(self, Tree tree, object X, np.ndarray y, np.ndarray f):
        """
        Build a decision tree from the training set (X, y).
        """
        pass

    cdef inline _check_input(self, object X, np.ndarray y, np.ndarray f):
        """
        Check input dtype, layout and format.
        """
        if X.dtype != INT:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=np.int32)

        if y.dtype != INT or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.int32)

        if f.dtype != INT or not f.flags.contiguous:
            f = np.ascontiguousarray(f, dtype=np.int32)

        return X, y, f

    cdef double _leaf_value(self, int[::1] y, int* samples, int n_samples) nogil:
        """
        Compute the leaf value according to the lables of the samples in the node.
        """
        pass

cdef class DepthFirstTreeBuilder(TreeBuilder):
    """
    Build a decision tree in depth-first fashion.
    """

    def __cinit__(self, Splitter splitter, int min_samples_split,
                  int min_samples_leaf, int max_depth):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

    cpdef void build(self, Tree tree, object X, np.ndarray y, np.ndarray f):
        """
        Build a decision tree from the training set (X, y).
        """

        # check input
        X, y, f = self._check_input(X, y, f)

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef int max_depth = self.max_depth
        cdef int min_samples_leaf = self.min_samples_leaf
        cdef int min_samples_split = self.min_samples_split

        # StackRecord parameters
        cdef StackRecord stack_record
        cdef int depth
        cdef int parent
        cdef bint is_left
        cdef int* samples
        cdef int n_samples = X.shape[0]
        cdef int* features
        cdef int n_features = X.shape[1]
        cdef Stack stack = Stack(INITIAL_STACK_SIZE)

        # compute variables
        cdef SplitRecord split
        cdef bint is_leaf
        cdef int feature
        cdef double value
        cdef int node_id

        cdef int i
        cdef max_depth_seen = 0

        # fill in samples and features arrays
        samples = <int *>malloc(n_samples * sizeof(int))
        features = <int *>malloc(n_features * sizeof(int))

        for i in range(n_samples):
            samples[i] = i

        for i in range(n_features):
            features[i] = i



        # with nogil:

        # push root node onto stack
        # TODO: add checks for out-of-memory
        rc = stack.push(0, _TREE_UNDEFINED, 0, samples, n_samples, features, n_features)

        while not stack.is_empty():
            stack.pop(&stack_record)
            depth = stack_record.depth
            parent = stack_record.parent
            is_left = stack_record.is_left
            samples = stack_record.samples
            n_samples = stack_record.n_samples
            features = stack_record.features
            n_features = stack_record.n_features

            # printf("\npopping (%d, %d, %d, %d, %d)\n", depth, parent, is_left, n_samples, n_features)

            is_leaf = (depth >= max_depth or
                       n_samples < min_samples_split or
                       n_samples < 2 * min_samples_leaf or
                       n_features <= 1)

            if not is_leaf:
                # printf("splitting node\n")
                rc = splitter.node_split(X, y, f, samples, n_samples, features, n_features, &split)
                if rc == -2:
                    # printf("failed to split node, creating leaf\n")
                    is_leaf = 1
                else:
                    feature = split.feature
                    value = _TREE_UNDEFINED
            
            if is_leaf:
                # printf("creating leaf\n")
                # printf("n_samples: %d\n", n_samples)
                # printf("n_features: %d\n", n_features)
                value = self._leaf_value(y, samples, n_samples)

            # printf("value: %.7f\n", value)
            node_id = tree.add_node(parent, is_left, is_leaf, feature, n_samples, value)
            # printf("done adding node\n")

            if not is_leaf:

                # Push right child on stack
                # printf("pushing right (%d, %d, %d, %d, %d)\n", depth + 1, node_id, 0,
                #        split.right_count, split.n_features)
                rc = stack.push(depth + 1, node_id, 0, split.right_indices, split.right_count,
                                split.features, split.n_features)
                if rc == -1:
                    break

                # Push left child on stack
                # printf("pushing left (%d, %d, %d, %d, %d)\n", depth + 1, node_id, 1,
                #        split.left_count, split.n_features)
                rc = stack.push(depth + 1, node_id, 1, split.left_indices, split.left_count,
                                split.features, split.n_features)
                if rc == -1:
                    break

            if depth > max_depth_seen:
                tree.max_depth = depth
                max_depth_seen = depth

            # clean up
            # printf('freeing samples\n')
            free(samples)
            if not is_left:
                # printf('freeing features\n')
                free(features)

        if rc == -1:
            # printf("memory error\n")
            raise MemoryError()

    @cython.cdivision(True)
    cdef double _leaf_value(self, int[::1] y, int* samples, int n_samples) nogil:
        cdef int pos_count = 0
        cdef int i

        for i in range(n_samples):
            pos_count += y[samples[i]]

        return pos_count / <double> n_samples


# =====================================
# Tree
# =====================================

cdef class Tree:

    property values:
        def __get__(self):
            return self._get_double_ndarray(self.values)[:self.node_count]

    property n_samples:
        def __get__(self):
            return self._get_int_ndarray(self.n_samples)[:self.node_count]

    property features:
        def __get__(self):
            return self._get_int_ndarray(self.features)[:self.node_count]

    property left_children:
        def __get__(self):
            return self._get_int_ndarray(self.left_children)[:self.node_count]

    property right_children:
        def __get__(self):
            return self._get_int_ndarray(self.right_children)[:self.node_count]

    def __cinit__(self, int n_features):
        """
        Constructor.
        """
        self.n_features = n_features

        # internal data structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 3
        self.values = <double *>malloc(self.capacity * sizeof(double))
        self.n_samples = <int *>malloc(self.capacity * sizeof(int))
        self.features = <int *>malloc(self.capacity * sizeof(int))
        self.left_children = <int *>malloc(self.capacity * sizeof(int))
        self.right_children = <int *>malloc(self.capacity * sizeof(int))


    def __dealloc__(self):
        """
        Destructor.
        """
        free(self.values)
        free(self.features)
        free(self.left_children)
        free(self.right_children)

    cdef int add_node(self, int parent, bint is_left, bint is_leaf, int feature,
                      int n_samples, double value) nogil except -1:

        cdef int node_id = self.node_count

        if node_id >= self.capacity:
            # printf("resizing\n")
            self._resize()
            # printf("done resizing\n")

        self.n_samples[node_id] = n_samples

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.left_children[parent] = node_id
            else:
                self.right_children[parent] = node_id

        if is_leaf:
            self.values[node_id] = value
            self.features[node_id] = _TREE_UNDEFINED
            self.left_children[node_id] = _TREE_LEAF
            self.right_children[node_id] = _TREE_LEAF
        else:
            # children will be set later
            self.values[node_id] = _TREE_UNDEFINED
            self.features[node_id] = feature

        self.node_count += 1

        return node_id

    # private
    cdef int _resize(self, int capacity=0) nogil except -1:

        if capacity > self.capacity:
            self.capacity = int(capacity)

        if capacity <= 0 and self.node_count == self.capacity:
            self.capacity *= 2

        # printf("capacity: %d\n", self.capacity)

        self.values = <double *>realloc(self.values, self.capacity * sizeof(double))
        self.n_samples = <int *>realloc(self.n_samples, self.capacity * sizeof(int))
        self.features = <int *>realloc(self.features, self.capacity * sizeof(int))
        self.left_children = <int *>realloc(self.left_children, self.capacity * sizeof(int))
        self.right_children = <int *>realloc(self.right_children, self.capacity * sizeof(int))

        return 0

    cdef np.ndarray _get_double_ndarray(self, double *data):
        """
        Wraps value as a 1-d NumPy array.
        The array keeps a reference to this Tree, which manages the underlying memory.
        """
        cdef np.npy_intp shape[1]
        shape[0] = self.node_count
        cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, data)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cdef np.ndarray _get_int_ndarray(self, int *data):
        """
        Wraps value as a 1-d NumPy array.
        The array keeps a reference to this Tree, which manages the underlying memory.
        """
        cdef np.npy_intp shape[1]
        shape[0] = self.node_count
        cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, data)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr
