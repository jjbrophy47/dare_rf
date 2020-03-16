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
from libc.stdlib cimport rand
from libc.stdlib cimport srand
from libc.stdio cimport printf

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport Stack
from ._utils cimport StackRecord

# constants
from numpy import int32 as INT

cdef int _TREE_LEAF = -1
cdef int _TREE_UNDEFINED = -2
cdef int INITIAL_STACK_SIZE = 10

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
        srand(random_state)  # do once? yes, once per tree
        # get rid of garbage first value:
        # https://stackoverflow.com/questions/30430137/first-random-number-is-always-smaller-than-rest
        rand()

    cpdef void build(self, _Tree tree):
        """
        Build a decision tree from the training set (X, y).
        """
        # printf('build\n')

        # Parameters
        cdef _DataManager manager = self.manager
        cdef _Splitter splitter = self.splitter
        cdef int max_depth = self.max_depth
        cdef int min_samples_leaf = self.min_samples_leaf
        cdef int min_samples_split = self.min_samples_split

        # get data
        cdef int** X = NULL
        cdef int* y = NULL
        cdef int* features = NULL
        cdef int n_samples = manager.n_samples
        cdef int n_features = manager.n_features
        manager.get_data(&X, &y)
        manager.get_features(&features)

        # Initial capacity
        cdef int init_capacity

        if max_depth <= 10:
            init_capacity = (2 ** (max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # StackRecord parameters
        cdef StackRecord stack_record
        cdef int depth
        cdef int parent
        cdef bint is_left
        cdef int* samples
        # cdef int* original_samples
        cdef Stack stack = Stack(INITIAL_STACK_SIZE)

        # compute variables
        cdef SplitRecord split
        cdef bint is_leaf
        cdef int feature
        cdef double value
        cdef int node_id

        cdef int i
        cdef Meta meta

        cdef int rc

        # fill in samples and features arrays
        samples = <int *>malloc(n_samples * sizeof(int))
        # original_samples = <int *>malloc(n_samples * sizeof(int))

        for i in range(n_samples):
            samples[i] = i

        # for i in range(n_samples):
        #     original_samples[i] = i

        # push root node onto stack
        rc = stack.push(0, _TREE_UNDEFINED, 1, 0, samples, n_samples, features, n_features)

        while not stack.is_empty():

            # populate stack record
            stack.pop(&stack_record)
            depth = stack_record.depth
            parent = stack_record.parent
            parent_p = stack_record.parent_p
            is_left = stack_record.is_left
            samples = stack_record.samples
            # original_samples = stack_record.original_samples
            n_samples = stack_record.n_samples
            features = stack_record.features
            n_features = stack_record.n_features

            meta.count = n_samples

            # printf("\npopping (%d, %d, %d, %d, %d)\n", depth, parent, is_left, n_samples, n_features)

            is_leaf = (depth >= max_depth or
                       n_samples < min_samples_split or
                       n_samples < 2 * min_samples_leaf or
                       n_features <= 1)

            if not is_leaf:
                rc = splitter.node_split(X, y, samples, n_samples, features,
                                         n_features, parent_p, &split, &meta)
                if rc == -2:
                    is_leaf = 1
                else:
                    feature = split.feature
                    value = _TREE_UNDEFINED
            
            if is_leaf:
                value = self._leaf_value(y, samples, n_samples, &meta)
                meta.feature_count = _TREE_UNDEFINED

            node_id = tree.add_node(parent, is_left, is_leaf, feature, value,
                                    depth, samples, &meta)

            if not is_leaf:

                # Push right child on stack
                rc = stack.push(depth + 1, node_id, meta.p, 0, split.right_indices,
                                split.right_count, split.features, split.n_features)

                # Push left child on stack
                rc = stack.push(depth + 1, node_id, meta.p, 1, split.left_indices,
                                split.left_count, split.features, split.n_features)

            # clean up
            if not is_leaf:
                free(samples)

    cdef void build_at_node(self, int node_id, _Tree tree,
                            int* samples, int n_samples,
                            int* features, int n_features,
                            int depth, int parent, double parent_p,
                            bint is_left):
        """
        Build subtree from the training set (X, y) starting at `node_id`.
        """
        printf('rebuilding at node %d\n', node_id)

        # Parameters
        cdef _DataManager manager = self.manager
        cdef _Splitter splitter = self.splitter
        cdef int max_depth = self.max_depth
        cdef int min_samples_leaf = self.min_samples_leaf
        cdef int min_samples_split = self.min_samples_split

        # get data
        cdef int** X = NULL
        cdef int* y = NULL
        manager.get_data(&X, &y)

        # StackRecord parameters
        cdef StackRecord stack_record
        cdef Stack stack = Stack(INITIAL_STACK_SIZE)

        # compute variables
        cdef SplitRecord split
        cdef bint is_leaf
        cdef int feature
        cdef double value

        cdef int i
        cdef Meta meta

        manager.get_data(&X, &y)

        # cdef int *samples = <int *>malloc(n_samples * sizeof(int))
        # for i in range(n_samples):
        #     samples[i] = i

        # push root node onto stack
        # TODO: add checks for out-of-memory
        rc = stack.push(depth, parent, parent_p, is_left, samples,
                        n_samples, features, n_features)

        while not stack.is_empty():

            # populate record
            stack.pop(&stack_record)
            depth = stack_record.depth
            parent = stack_record.parent
            parent_p = stack_record.parent_p
            is_left = stack_record.is_left
            samples = stack_record.samples
            # original_samples = stack_record.original_samples
            n_samples = stack_record.n_samples
            features = stack_record.features
            n_features = stack_record.n_features

            meta.count = n_samples

            # printf("\npopping (%d, %d, %.7f, %d, %d, %d)\n", depth, parent, parent_p, is_left, n_samples, n_features)

            is_leaf = (depth >= max_depth or
                       n_samples < min_samples_split or
                       n_samples < 2 * min_samples_leaf or
                       n_features <= 1)

            if not is_leaf:
                # printf('node_split\n')
                rc = splitter.node_split(X, y, samples, n_samples, features, n_features,
                                         parent_p, &split, &meta)
                # printf('rc: %d\n', rc)
                if rc == -2:
                    is_leaf = 1
                else:
                    feature = split.feature
                    value = _TREE_UNDEFINED
            
            if is_leaf:
                value = self._leaf_value(y, samples, n_samples, &meta)
                meta.feature_count = _TREE_UNDEFINED

            # printf('adding node\n')
            node_id = tree.add_node(parent, is_left, is_leaf, feature, value,
                                    depth, samples, &meta)
            # printf('done adding node\n')

            if not is_leaf:

                # Push right child on stack
                printf("pushing right (%d, %d, %d, %d, %d)\n", depth + 1, node_id, 0,
                       split.right_count, split.n_features)
                rc = stack.push(depth + 1, node_id, meta.p, 0, split.right_indices, 
                                split.right_count, split.features, split.n_features)

                # Push left child on stack
                printf("pushing left (%d, %d, %d, %d, %d)\n", depth + 1, node_id, 1,
                       split.left_count, split.n_features)
                rc = stack.push(depth + 1, node_id, meta.p, 1, split.left_indices,
                                split.left_count, split.features, split.n_features)

            # clean up
            # free(samples)
            if not is_leaf:
                free(samples)
        # free(X)
        # free(y)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef double _leaf_value(self, int* y, int* samples, int n_samples, Meta* meta) nogil:
        cdef int pos_count = 0
        cdef int i

        for i in range(n_samples):
            pos_count += y[samples[i]]

        meta.pos_count = pos_count
        return pos_count / <double> n_samples


# =====================================
# Tree
# =====================================

cdef class _Tree:

    property n_nodes:
        def __get__(self):
            return self.node_count

    property values:
        def __get__(self):
            return self._get_double_ndarray(self.values)[:self.node_count]

    property p:
        def __get__(self):
            return self._get_double_ndarray(self.p)[:self.node_count]

    property chosen_features:
        def __get__(self):
            return self._get_int_ndarray(self.chosen_features)[:self.node_count]

    property left_children:
        def __get__(self):
            return self._get_int_ndarray(self.left_children)[:self.node_count]

    property right_children:
        def __get__(self):
            return self._get_int_ndarray(self.right_children)[:self.node_count]

    property depth:
        def __get__(self):
            return self._get_int_ndarray(self.depth)[:self.node_count]

    # metadata
    property counts:
        def __get__(self):
            return self._get_int_ndarray(self.count)[:self.node_count]

    property pos_counts:
        def __get__(self):
            return self._get_int_ndarray(self.pos_count)[:self.node_count]

    property feature_counts:
        def __get__(self):
            return self._get_int_ndarray(self.feature_count)[:self.node_count]

    cpdef np.ndarray _get_left_counts(self, node_id):
        return self._get_int_ndarray(self.left_counts[node_id])[:self.feature_count[node_id]]

    cpdef np.ndarray _get_left_pos_counts(self, node_id):
        return self._get_int_ndarray(self.left_pos_counts[node_id])[:self.feature_count[node_id]]

    cpdef np.ndarray _get_right_counts(self, node_id):
        return self._get_int_ndarray(self.right_counts[node_id])[:self.feature_count[node_id]]

    cpdef np.ndarray _get_right_pos_counts(self, node_id):
        return self._get_int_ndarray(self.right_pos_counts[node_id])[:self.feature_count[node_id]]

    cpdef np.ndarray _get_features(self, node_id):
        return self._get_int_ndarray(self.features[node_id])[:self.feature_count[node_id]]

    cpdef np.ndarray _get_leaf_samples(self, node_id):
        return self._get_int_ndarray(self.leaf_samples[node_id])[:self.count[node_id]]

    def __cinit__(self):
        """
        Constructor.
        """

        # internal data structures
        self.node_count = 0
        self.capacity = 3
        self.values = <double *>malloc(self.capacity * sizeof(double))
        self.p = <double *>malloc(self.capacity * sizeof(double))
        self.chosen_features = <int *>malloc(self.capacity * sizeof(int))
        self.left_children = <int *>malloc(self.capacity * sizeof(int))
        self.right_children = <int *>malloc(self.capacity * sizeof(int))
        self.depth = <int *>malloc(self.capacity * sizeof(int))

        # internal metadata
        self.count = <int *>malloc(self.capacity * sizeof(int))
        self.pos_count = <int *>malloc(self.capacity * sizeof(int))
        self.feature_count = <int *>malloc(self.capacity * sizeof(int))
        self.left_counts = <int **>malloc(self.capacity * sizeof(int *))
        self.left_pos_counts = <int **>malloc(self.capacity * sizeof(int *))
        self.right_counts = <int **>malloc(self.capacity * sizeof(int *))
        self.right_pos_counts = <int **>malloc(self.capacity * sizeof(int *))
        self.features = <int **>malloc(self.capacity * sizeof(int *))
        self.leaf_samples = <int **>malloc(self.capacity * sizeof(int *))

    def __dealloc__(self):
        """
        Destructor.
        """
        free(self.values)
        free(self.p)
        free(self.chosen_features)
        free(self.left_children)
        free(self.right_children)
        free(self.depth)

        free(self.count)
        free(self.pos_count)
        free(self.feature_count)
        free(self.left_counts)
        free(self.left_pos_counts)
        free(self.right_counts)
        free(self.right_pos_counts)
        free(self.features)

    cdef int add_node(self, int parent, bint is_left, bint is_leaf, int feature,
                      double value, int depth, int* samples, Meta* meta) nogil except -1:

        cdef int node_id = self.node_count

        if node_id >= self.capacity:
            # printf("resizing\n")
            self._resize()
            # printf("done resizing\n")

        self.count[node_id] = meta.count
        self.pos_count[node_id] = meta.pos_count
        self.depth[node_id] = depth

        # printf('parent: %d, node_id: %d\n', parent, node_id)

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.left_children[parent] = node_id
            else:
                self.right_children[parent] = node_id

        if is_leaf:
            self.values[node_id] = value
            self.p[node_id] = _TREE_UNDEFINED
            self.chosen_features[node_id] = _TREE_UNDEFINED
            self.left_children[node_id] = _TREE_LEAF
            self.right_children[node_id] = _TREE_LEAF

            self.leaf_samples[node_id] = samples
            self.feature_count[node_id] = _TREE_UNDEFINED
        else:
            # children will be set later
            self.values[node_id] = _TREE_UNDEFINED
            self.p[node_id] = meta.p
            self.chosen_features[node_id] = feature

            self.left_counts[node_id] = meta.left_counts
            self.left_pos_counts[node_id] = meta.left_pos_counts
            self.right_counts[node_id] = meta.right_counts
            self.right_pos_counts[node_id] = meta.right_pos_counts
            self.feature_count[node_id] = meta.feature_count
            self.features[node_id] = meta.features

        self.node_count += 1

        return node_id

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray predict(self, object X):
        """
        Predict probability of positive label for X.
        """

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s" % type(X))

        if X.dtype != INT:
            raise ValueError("X.dtype should be np.int32, got %s" % X.dtype)

        # Extract input
        cdef int[:, :] X_ndarray = X
        cdef int n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[double] out = np.zeros((n_samples,), dtype=np.double)

        # incrementers
        cdef int i = 0
        cdef int j

        with nogil:

            for i in range(n_samples):
                j = 0

                while self.values[j] == _TREE_UNDEFINED:
                    if X_ndarray[i, self.chosen_features[j]] == 1:
                        j = self.left_children[j]
                    else:
                        j = self.right_children[j]

                out[i] = self.values[j]

        return out

    # private
    cdef int _resize(self, int capacity=0) nogil except -1:

        if capacity > self.capacity:
            self.capacity = int(capacity)

        if capacity <= 0 and self.node_count == self.capacity:
            self.capacity *= 2

        # tree info
        self.values = <double *>realloc(self.values, self.capacity * sizeof(double))
        self.p = <double *>realloc(self.p, self.capacity * sizeof(double))
        self.chosen_features = <int *>realloc(self.chosen_features, self.capacity * sizeof(int))
        self.left_children = <int *>realloc(self.left_children, self.capacity * sizeof(int))
        self.right_children = <int *>realloc(self.right_children, self.capacity * sizeof(int))
        self.depth = <int *>realloc(self.depth, self.capacity * sizeof(int))

        # metadata
        self.count = <int *>realloc(self.count, self.capacity * sizeof(int))
        self.pos_count = <int *>realloc(self.pos_count, self.capacity * sizeof(int))
        self.feature_count = <int *>realloc(self.feature_count, self.capacity * sizeof(int))
        self.left_counts = <int **>realloc(self.left_counts, self.capacity * sizeof(int *))
        self.left_pos_counts = <int **>realloc(self.left_pos_counts, self.capacity * sizeof(int *))
        self.right_counts = <int **>realloc(self.right_counts, self.capacity * sizeof(int *))
        self.right_pos_counts = <int **>realloc(self.right_pos_counts, self.capacity * sizeof(int *))
        self.features = <int **>realloc(self.features, self.capacity * sizeof(int *))
        self.leaf_samples = <int **>realloc(self.leaf_samples, self.capacity * sizeof(int *))

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
