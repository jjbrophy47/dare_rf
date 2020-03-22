"""
CeDAR binary tree implementation; only supports binary attributes and a binary label.
Represenation is a number of parllel arrays.
Adapted from: https://github.com/scikit-learn/scikit-learn/blob/b194674c42d54b26137a456c510c5fdba1ba23e0/sklearn/tree/_tree.pyx
"""
from cpython cimport Py_INCREF
from cpython.ref cimport PyObject

cimport cython

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf
from libc.math cimport exp
from libc.time cimport time

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport IntStack
from ._utils cimport RemovalStack
from ._utils cimport RemovalStackRecord
from ._utils cimport compute_gini
from ._utils cimport generate_distribution
from ._utils cimport convert_int_ndarray

cdef int _TREE_LEAF = -1
cdef int _TREE_UNDEFINED = -2
cdef int INITIAL_STACK_SIZE = 10

# =====================================
# Remover
# =====================================

cdef class _Remover:
    """
    Removes data from a learned tree.
    """

    # removal metrics
    property remove_types:
        def __get__(self):
            return self._get_int_ndarray(self.remove_types, self.remove_count)

    property remove_depths:
        def __get__(self):
            return self._get_int_ndarray(self.remove_depths, self.remove_count)

    def __cinit__(self, _DataManager manager, double epsilon, double lmbda):
        """
        Constructor.
        """
        self.manager = manager
        self.epsilon = epsilon
        self.lmbda = lmbda

        self.capacity = 10
        self.remove_count = 0
        self.remove_types = <int *>malloc(self.capacity * sizeof(int))
        self.remove_depths = <int *>malloc(self.capacity * sizeof(int))

    def __dealloc__(self):
        """
        Destructor.
        """
        if self.remove_types:
            free(self.remove_types)
        if self.remove_depths:
            free(self.remove_depths)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int remove(self, _Tree tree, _TreeBuilder tree_builder,
                     np.ndarray remove_indices):
        """
        Remove the data (X, y) from the learned _Tree.
        """

        # Parameters
        cdef int min_samples_leaf = tree_builder.min_samples_leaf
        cdef int min_samples_split = tree_builder.min_samples_split
        cdef _DataManager manager = self.manager

        # get data
        cdef int** X = NULL
        cdef int* y = NULL

        # StackRecord parameters
        cdef RemovalStackRecord stack_record
        cdef int depth
        cdef int node_id
        cdef double parent_p
        cdef int* samples = convert_int_ndarray(remove_indices)
        cdef int n_samples = remove_indices.shape[0]
        cdef RemovalStack stack = RemovalStack(INITIAL_STACK_SIZE)

        # compute variables
        cdef RemovalSplitRecord split
        cdef Meta meta
        cdef int chosen_feature

        cdef int rc
        cdef int i

        cdef int remove_count = 0
        cdef int* remove_types = <int *>malloc(n_samples * sizeof(int))
        cdef int* remove_depths = <int *>malloc(n_samples * sizeof(int))

        cdef int* rebuild_samples = NULL
        cdef int n_rebuild_samples

        cdef int root = tree.root

        # make room for new deletions
        self._resize(n_samples)

        # check if any sample has already been deleted
        rc = manager.check_sample_validity(samples, n_samples)
        if rc == -1:
            return -1
        manager.get_data(&X, &y)

        # push root node onto stack
        rc = stack.push(0, root, 0, _TREE_UNDEFINED, 1, samples, n_samples)

        while not stack.is_empty():

            # populate split data
            stack.pop(&stack_record)
            depth = stack_record.depth
            node_id = stack_record.node_id
            is_left = stack_record.is_left
            parent = stack_record.parent
            parent_p = stack_record.parent_p
            samples = stack_record.samples
            n_samples = stack_record.n_samples

            # populate node metadata
            meta.p = tree.p[node_id]
            meta.count = tree.count[node_id]
            meta.pos_count = tree.pos_count[node_id]
            meta.feature_count = tree.feature_count[node_id]
            meta.left_counts = tree.left_counts[node_id]
            meta.left_pos_counts = tree.left_pos_counts[node_id]
            meta.right_counts = tree.right_counts[node_id]
            meta.right_pos_counts = tree.right_pos_counts[node_id]
            meta.features = tree.features[node_id]

            printf("\npopping_r (%d, %d, %d, %d, %.7f, %d)\n", depth, node_id, is_left, parent, parent_p, n_samples)
            printf('node %d count: %d, pos_count: %d\n', node_id, meta.count, meta.pos_count)

            # if node_id == 3154:
            #     printf('left_counts[3154][0]: %d\n', tree.left_counts[3154][0])

            # for i in range(n_samples):
            #     printf('samples[%d]: %d\n', i, samples[i])

            # leaf
            if tree.values[node_id] >= 0:
                printf('leaf!\n')
                self._update_leaf(node_id, tree, y, samples, n_samples)
                remove_types[remove_count] = 0
                remove_depths[remove_count] = depth
                remove_count += 1
                free(samples)

            # decision node
            else:
                chosen_feature = tree.chosen_features[node_id]
                # printf('chosen_feature: %d\n', chosen_feature)
                rc = self._node_remove(node_id, X, y, samples, n_samples,
                                       min_samples_split, min_samples_leaf,
                                       chosen_feature, parent_p, &split, &meta)
                printf('rc: %d\n', rc)

                # retrain
                if rc > 0:
                    printf('_collect_leaf_samples\n')
                    n_rebuild_samples = self._collect_leaf_samples(node_id, is_left, parent,
                                                                   tree, samples, n_samples,
                                                                   &rebuild_samples)
                    printf('n_rebuild_samples: %d\n', n_rebuild_samples)
                    tree_builder.build_at_node(node_id, tree, rebuild_samples, n_rebuild_samples,
                                               meta.features, meta.feature_count,
                                               depth, parent, parent_p, is_left)
                    # printf('done rebuilding\n')
                    remove_types[remove_count] = rc
                    remove_depths[remove_count] = depth
                    remove_count += 1

                else:

                    self._update_decision_node(node_id, tree, n_samples, &meta)

                    # push right branch
                    if split.right_count > 0:
                        stack.push(depth + 1, tree.right_children[node_id], 0, node_id,
                                   meta.p, split.right_indices, split.right_count)

                    # push left branch
                    if split.left_count > 0:
                        stack.push(depth + 1, tree.left_children[node_id], 1, node_id,
                                   meta.p, split.left_indices, split.left_count)

                free(samples)

        # update removal metrics
        self._update_removal_metrics(remove_types, remove_depths, remove_count)

    # private
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int _update_leaf(self, int node_id, _Tree tree, int* y, int* samples,
                          int n_samples) nogil:
        """
        Update leaf node: count, pos_count, value, leaf_samples.
        """
        cdef int pos_count = 0
        cdef int updated_count
        cdef int updated_pos_count

        cdef int* leaf_samples = tree.leaf_samples[node_id]
        cdef int* updated_leaf_samples = <int *>malloc(updated_count * sizeof(int))
        cdef int leaf_sample_count = 0
        cdef bint add_sample

        printf('tree.values[%d]: %.20f\n', node_id, tree.values[node_id])

        # count number of pos labels in removal data
        for i in range(n_samples):
            if y[samples[i]] == 1:
                pos_count += 1

        updated_count = tree.count[node_id] - n_samples
        updated_pos_count = tree.pos_count[node_id] - pos_count

        printf('current_count: %d, current_pos_count: %d\n', tree.count[node_id], tree.pos_count[node_id])
        printf('updated_count: %d, updated_pos_count: %d\n', updated_count, updated_pos_count)

        if updated_count == 0:
            printf('leaf count is zero!\n')
            # exit(0)

        # remove deleted samples from the leaf
        for i in range(tree.count[node_id]):
            add_sample = 1

            for j in range(n_samples):
                if leaf_samples[i] == samples[j]:
                    add_sample = 0
                    break

            if add_sample:
                updated_leaf_samples[leaf_sample_count] = leaf_samples[i]
                leaf_sample_count += 1

        # update tree
        free(leaf_samples)
        tree.count[node_id] = updated_count
        tree.pos_count[node_id] = updated_pos_count
        tree.values[node_id] = tree.pos_count[node_id] / <double> tree.count[node_id]
        tree.leaf_samples[node_id] = updated_leaf_samples

        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _update_decision_node(self, int node_id, _Tree tree, int n_samples, Meta* meta) nogil:
        """
        Update tree with node metadata.
        """
        tree.count[node_id] = meta.count - n_samples
        tree.pos_count[node_id] = meta.pos_count
        tree.feature_count[node_id] = meta.feature_count
        tree.left_counts[node_id] = meta.left_counts
        tree.left_pos_counts[node_id] = meta.left_pos_counts
        tree.right_counts[node_id] = meta.right_counts
        tree.right_pos_counts[node_id] = meta.right_pos_counts
        tree.features[node_id] = meta.features
        return 0


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int _node_remove(self, int node_id, int** X, int* y,
                          int* samples, int n_samples,
                          int min_samples_split, int min_samples_leaf,
                          int chosen_feature, double parent_p,
                          RemovalSplitRecord *split, Meta* meta) nogil:
        """
        Update node statistics based on the removal data (X, y).
        Return 0 for a successful update, -1 to signal a retrain,
          -2 to signal a leaf creation.
        """

        # parameters
        cdef double epsilon = self.epsilon
        cdef double lmbda = self.lmbda

        cdef int count = n_samples
        cdef int pos_count = 0
        cdef int left_count
        cdef int left_pos_count
        cdef int right_count
        cdef int right_pos_count

        cdef int updated_count = meta.count - count
        cdef int updated_pos_count
        cdef int updated_left_count
        cdef int updated_left_pos_count
        cdef int updated_right_count
        cdef int updated_right_pos_count

        cdef int i
        cdef int j
        cdef int k

        cdef int feature_count = 0
        cdef int result = 0

        cdef bint chosen_feature_validated = 0
        cdef int chosen_ndx

        cdef double p
        cdef double ratio

        cdef double* gini_indices
        cdef double* distribution
        cdef int* valid_features

        cdef int* left_counts
        cdef int* left_pos_counts
        cdef int* right_counts
        cdef int* right_pos_counts

        cdef int chosen_left_count
        cdef int chosen_right_count

        # count number of pos labels in removal data
        for i in range(n_samples):
            if y[samples[i]] == 1:
                pos_count += 1

        updated_pos_count = meta.pos_count - pos_count

        # no samples left in this node => retrain
        if updated_count <= 0:  # this branch will not be reached
            printf('meta count <= count\n')
            result = 2

        # only samples from one class are left in this node => create leaf
        elif updated_pos_count == 0 or updated_pos_count == updated_count:
            printf('only samples from one class\n')
            result = 1

        else:

            # if node_id == 2269:
            #     printf('node count: %d\n', meta.count)
            #     printf('n_samples: %d\n', n_samples)

            gini_indices = <double *>malloc(meta.feature_count * sizeof(double))
            distribution = <double *>malloc(meta.feature_count * sizeof(double))
            valid_features = <int *>malloc(meta.feature_count * sizeof(int))

            updated_left_counts = <int *>malloc(meta.feature_count * sizeof(int))
            updated_left_pos_counts = <int *>malloc(meta.feature_count * sizeof(int))
            updated_right_counts = <int *>malloc(meta.feature_count * sizeof(int))
            updated_right_pos_counts = <int *>malloc(meta.feature_count * sizeof(int))

            # compute statistics of the removal data for each attribute
            for j in range(meta.feature_count):

                left_count = 0
                left_pos_count = 0

                for i in range(n_samples):

                    if X[samples[i]][meta.features[j]] == 1:
                        left_count += 1
                        left_pos_count += y[samples[i]]

                right_count = count - left_count
                right_pos_count = pos_count - left_pos_count

                updated_left_count = meta.left_counts[feature_count] - left_count
                updated_left_pos_count = meta.left_pos_counts[feature_count] - left_pos_count
                updated_right_count = meta.right_counts[feature_count] - right_count
                updated_right_pos_count = meta.right_pos_counts[feature_count] - right_pos_count

                # printf('%d: left_count: %d, right_count: %d\n', meta.features[j], updated_left_count, updated_right_count)

                # validate split
                if updated_left_count >= min_samples_leaf and updated_right_count >= min_samples_leaf:
                    valid_features[feature_count] = meta.features[j]
                    gini_indices[feature_count] = compute_gini(updated_count, updated_left_count,
                        updated_right_count, updated_left_pos_count, updated_right_pos_count)

                    # update metadata
                    updated_left_counts[feature_count] = updated_left_count
                    updated_left_pos_counts[feature_count] = updated_left_pos_count
                    updated_right_counts[feature_count] = updated_right_count
                    updated_right_pos_counts[feature_count] = updated_right_pos_count

                    if meta.features[j] == chosen_feature:
                        chosen_feature_validated = 1
                        chosen_ndx = feature_count
                        chosen_left_count = left_count
                        chosen_right_count = right_count

                    feature_count += 1

            # no valid features after data removal => create leaf
            if feature_count == 0:
                printf('feature_count is zero\n')
                result = 1
                free(gini_indices)
                free(distribution)
                free(valid_features)
                free(updated_left_counts)
                free(updated_left_pos_counts)
                free(updated_right_counts)
                free(updated_right_pos_counts)

                free(meta.features)
                meta.feature_count = feature_count

            # current feature no longer valid => retrain
            elif not chosen_feature_validated:
                printf('chosen feature not validated\n')
                result = 2
                free(gini_indices)
                free(distribution)

                free(updated_left_counts)
                free(updated_left_pos_counts)
                free(updated_right_counts)
                free(updated_right_pos_counts)

                free(meta.features)
                meta.features = valid_features
                meta.feature_count = feature_count

            else:

                # remove invalid features
                gini_indices = <double *>realloc(gini_indices, feature_count * sizeof(double))
                distribution = <double *>realloc(distribution, feature_count * sizeof(double))
                valid_features = <int *>realloc(valid_features, feature_count * sizeof(int))

                updated_left_counts = <int *>realloc(updated_left_counts, feature_count * sizeof(int))
                updated_left_pos_counts = <int *>realloc(updated_left_pos_counts, feature_count * sizeof(int))
                updated_right_counts = <int *>realloc(updated_right_counts, feature_count * sizeof(int))
                updated_right_pos_counts = <int *>realloc(updated_right_pos_counts, feature_count * sizeof(int))

                # compute new probability for the chosen feature
                generate_distribution(lmbda, distribution, gini_indices, feature_count)
                p = parent_p * distribution[chosen_ndx]
                ratio = p / meta.p

                printf('ratio: %.3f, epsilon: %.3f, lmbda: %.3f\n', ratio, epsilon, lmbda)

                # compare with previous probability => retrain if necessary
                if ratio < exp(-epsilon) or ratio > exp(epsilon):
                    result = 2
                    free(gini_indices)
                    free(distribution)

                    free(updated_left_counts)
                    free(updated_left_pos_counts)
                    free(updated_right_counts)
                    free(updated_right_pos_counts)

                    free(meta.features)
                    meta.features = valid_features
                    meta.feature_count = feature_count

                else:

                    # split removal data based on the chosen feature
                    split.left_indices = <int *>malloc(chosen_left_count * sizeof(int))
                    split.right_indices = <int *>malloc(chosen_right_count * sizeof(int))
                    j = 0
                    k = 0
                    for i in range(n_samples):
                        if X[samples[i]][valid_features[chosen_ndx]] == 1:
                            split.left_indices[j] = samples[i]
                            j += 1
                        else:
                            split.right_indices[k] = samples[i]
                            k += 1
                    split.left_count = j
                    split.right_count = k

                    # cleanup
                    free(gini_indices)
                    free(distribution)

                    free(meta.left_counts)
                    free(meta.left_pos_counts)
                    free(meta.right_counts)
                    free(meta.right_pos_counts)
                    free(meta.features)

                    meta.pos_count = updated_pos_count
                    meta.feature_count = feature_count
                    meta.left_counts = updated_left_counts
                    meta.left_pos_counts = updated_left_pos_counts
                    meta.right_counts = updated_right_counts
                    meta.right_pos_counts = updated_right_pos_counts
                    meta.features = valid_features

        return result

    cdef int _collect_leaf_samples(self, int node_id, int is_left,
                                   int parent, _Tree tree, int*
                                   remove_samples, int n_remove_samples,
                                   int** rebuild_samples_ptr):
        """
        Gathers all samples at the leaves and clears any saved metadata
        as it traverses through the tree.
        """

        cdef int i
        cdef int j

        cdef int rebuild_sample_count = 0
        cdef int *rebuild_samples = NULL

        # get leaf ids and free nodes to be retrained
        cdef int* leaf_ids = <int *>malloc(tree.count[node_id] * sizeof(int))
        cdef int leaf_count = 0

        # get all indices for nodes to be removed
        cdef int *node_remove_ids = <int *>malloc(tree.node_count * sizeof(int))
        cdef int node_remove_count = 0
        cdef int temp_id
        cdef IntStack stack = IntStack(INITIAL_STACK_SIZE)
        stack.push(node_id)

        while not stack.is_empty():
            temp_id = stack.pop()
            printf('temp_id: %d\n', temp_id)
            node_remove_ids[node_remove_count] = temp_id
            node_remove_count += 1

            # leaf
            printf('tree.values[%d]: %.20f\n', temp_id, tree.values[temp_id])
            if tree.values[temp_id] >= 0:
                leaf_ids[leaf_count] = temp_id
                leaf_count += 1

            # decision node
            else:
                stack.push(tree.right_children[temp_id])
                stack.push(tree.left_children[temp_id])

                # printf('A\n')
                # printf('left_counts[%d][0]: %d\n', temp_id, tree.left_counts[temp_id][0])

                free(tree.left_counts[temp_id])
                free(tree.left_pos_counts[temp_id])
                free(tree.right_counts[temp_id])
                free(tree.right_pos_counts[temp_id])

                if temp_id != node_id:
                    free(tree.features[temp_id])

        # printf('reallocing\n')

        leaf_ids = <int *>realloc(leaf_ids, leaf_count * sizeof(int))
        node_remove_ids = <int *>realloc(node_remove_ids, node_remove_count * sizeof(int))

        # compile all samples from the leaves
        rebuild_samples = <int *>malloc(tree.count[node_id] * sizeof(int))
        rebuild_sample_count = 0

        # TODO: could check to see if all removal indices have been accounted for
        for i in range(leaf_count):
            leaf_id = leaf_ids[i]
            leaf_samples = tree.leaf_samples[leaf_id]
            n_leaf_samples = tree.count[leaf_id]

            for j in range(n_leaf_samples):
                add_sample = 1

                for k in range(n_remove_samples):
                    if leaf_samples[j] == remove_samples[k]:
                        add_sample = 0
                        break

                if add_sample:
                    rebuild_samples[rebuild_sample_count] = leaf_samples[j]
                    rebuild_sample_count += 1

            free(leaf_samples)
        free(leaf_ids)

        rebuild_samples = <int *>realloc(rebuild_samples, rebuild_sample_count * sizeof(int))

        # notfiy tree of vacant node ids
        tree.remove_nodes(node_remove_ids, node_remove_count)

        rebuild_samples_ptr[0] = rebuild_samples
        return rebuild_sample_count

    cdef void _resize(self, int capacity=0) nogil:
        """
        Increase size of removal allocations.
        """
        if capacity > self.capacity - self.remove_count:
            if self.capacity * 2 - self.remove_count > capacity:
                self.capacity *= 2
            else:
                self.capacity = int(capacity)

        # removal info
        if self.remove_types and self.remove_depths:
            self.remove_types = <int *>realloc(self.remove_types, self.capacity * sizeof(int))
            self.remove_depths = <int *>realloc(self.remove_depths, self.capacity * sizeof(int))
        else:
            self.remove_types = <int *>malloc(self.capacity * sizeof(int))
            self.remove_depths = <int *>malloc(self.capacity * sizeof(int))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _update_removal_metrics(self, int* remove_types, int* remove_depths,
                                      int remove_count) nogil:
        """
        Adds to the removal metrics.
        """
        cdef int current_count = self.remove_count
        cdef int updated_count = current_count + remove_count
        cdef int j = 0

        for i in range(current_count, updated_count):
            self.remove_types[i] = remove_types[j]
            self.remove_depths[i] = remove_depths[j]
            j += 1
        self.remove_count = updated_count

        free(remove_types)
        free(remove_depths)

    cpdef void clear_removal_metrics(self):
        """
        Resets deletion statistics.
        """
        self.remove_count = 0
        free(self.remove_types)
        free(self.remove_depths)
        self.remove_types = NULL
        self.remove_depths = NULL

    cdef np.ndarray _get_int_ndarray(self, int *data, int n_elem):
        """
        Wraps value as a 1-d NumPy array.
        The array keeps a reference to this Tree, which manages the underlying memory.
        """
        cdef np.npy_intp shape[1]
        shape[0] = n_elem
        cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, data)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr
