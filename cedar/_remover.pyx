"""
CeDAR binary tree implementation; only supports binary attributes and a binary label.
Represenation is a number of parllel arrays.
Adapted from: https://github.com/scikit-learn/scikit-learn/blob/b194674c42d54b26137a456c510c5fdba1ba23e0/sklearn/tree/_tree.pyx
"""
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

    def __cinit__(self, _DataManager manager, double epsilon, double lmbda):
        self.manager = manager
        self.epsilon = epsilon
        self.lmbda = lmbda

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
        cdef int n_samples = remove_indices.shape[0]
        cdef int n_data_indices = remove_indices.shape[0]
        cdef int* data_indices = convert_int_ndarray(remove_indices)

        # StackRecord parameters
        cdef RemovalStackRecord stack_record
        cdef int depth
        cdef int node_id
        cdef double parent_p
        cdef int* samples = <int *>malloc(n_samples * sizeof(int))
        cdef RemovalStack stack = RemovalStack(INITIAL_STACK_SIZE)

        # compute variables
        cdef RemovalSplitRecord split
        cdef Meta meta
        cdef int chosen_feature

        cdef int rc
        cdef int i

        cdef int remove_type_count = 0
        cdef int* remove_types = <int *>malloc(n_samples * sizeof(int))
        cdef int* remove_samples = <int *>malloc(n_samples * sizeof(int))

        cdef int* rebuild_samples = NULL
        cdef int n_rebuild_samples

        # check if any sample has already been deleted
        rc = manager.check_sample_validity(data_indices, n_samples)
        if rc == -1:
            return -1
        manager.get_data(data_indices, n_samples, &X, &y)

        # populate indices for removal data
        for i in range(n_samples):
            samples[i] = i
            remove_samples[i] = remove_indices[i]

        # push root node onto stack
        rc = stack.push(0, 0, 0, _TREE_UNDEFINED, 1, samples, remove_samples, n_samples)

        while not stack.is_empty():

            # populate split data
            stack.pop(&stack_record)
            depth = stack_record.depth
            node_id = stack_record.node_id
            is_left = stack_record.is_left
            parent = stack_record.parent
            parent_p = stack_record.parent_p
            samples = stack_record.samples
            remove_samples = stack_record.remove_samples
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

            # printf("\npopping_r (%d, %d, %d, %d, %.7f, %d)\n", depth, node_id, is_left, parent, parent_p, n_samples)

            # leaf
            if tree.values[node_id] >= 0:
                self._update_leaf(node_id, tree, y, samples, remove_samples, n_samples)
                remove_types[remove_type_count] = 0
                remove_type_count += 1

                free(remove_samples)
                free(samples)

            # decision node
            else:
                chosen_feature = tree.chosen_features[node_id]
                rc = self._node_remove(node_id, X, y, remove_samples, samples, n_samples,
                                       min_samples_split, min_samples_leaf,
                                       chosen_feature, parent_p, &split, &meta)
                free(samples)

                # printf('rc: %d\n', rc)

                # retrain
                if rc < 0:

                    n_rebuild_samples = self._collect_leaf_samples(node_id, tree, remove_samples,
                                                                   n_samples, &rebuild_samples)
                    free(X)
                    free(y)
                    tree_builder.build_at_node(node_id, tree, rebuild_samples, n_rebuild_samples,
                                               meta.features, meta.feature_count,
                                               depth, parent, parent_p, is_left)

                    free(remove_samples)
                    remove_types[remove_type_count] = rc
                    remove_type_count += 1

                else:

                    self._update_decision_node(node_id, tree, n_samples, &meta)

                    # traverse left branch
                    if split.left_count > 0:
                        stack.push(depth + 1, tree.left_children[node_id], 1, node_id,
                                   meta.p, split.left_indices, split.left_remove_indices,
                                   split.left_count)

                    # traverse right branch
                    if split.right_count > 0:
                        stack.push(depth + 1, tree.right_children[node_id], 0, node_id,
                                   meta.p, split.right_indices, split.right_remove_indices,
                                   split.right_count)

        # remove samples from the database
        manager.remove_data(data_indices, n_data_indices)

        # cleanup
        free(data_indices)
        free(remove_types)  # TODO: do something with this data

        return 0

    # private
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int _update_leaf(self, int node_id, _Tree tree, int* y, int* samples,
                          int* remove_samples, int n_samples) nogil:
        """
        Update leaf node: count, pos_count, value, leaf_samples.
        """
        cdef int current_sample_count = tree.count[node_id]
        cdef int updated_count = tree.count[node_id] - n_samples
        cdef int pos_count = 0

        cdef int* leaf_samples = tree.leaf_samples[node_id]
        cdef int* updated_leaf_samples = <int *>malloc(updated_count * sizeof(int))
        cdef int leaf_sample_count = 0
        cdef bint add_sample

        # count number of pos labels in removal data
        for i in range(n_samples):
            if y[samples[i]] == 1:
                pos_count += 1

        # remove deleted samples from the leaf
        for i in range(current_sample_count):
            add_sample = 1

            for j in range(n_samples):
                if leaf_samples[i] == remove_samples[j]:
                    add_sample = 0
                    break

            if add_sample:
                updated_leaf_samples[leaf_sample_count] = leaf_samples[i]
                leaf_sample_count += 1

        # update tree
        free(leaf_samples)
        tree.count[node_id] = updated_count
        tree.pos_count[node_id] -= pos_count
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
                          int* remove_samples, int* samples, int n_samples,
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
            result = -1

        # only samples from one class are left in this node => create leaf
        elif updated_pos_count == 0 or updated_pos_count == updated_count:
            printf('only samples from one class\n')
            result = -2

        else:

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

                # validate split
                printf('updated_left_count: %d, updated_right_count: %d\n', updated_left_count, updated_right_count)
                printf('min_samples_leaf: %d\n', min_samples_leaf)
                if updated_left_count >= min_samples_leaf and updated_right_count >= min_samples_leaf:
                    valid_features[feature_count] = meta.features[j]
                    gini_indices[feature_count] = compute_gini(updated_count, updated_left_count,
                        updated_right_count, updated_left_pos_count, updated_right_pos_count)

                    # update metadata
                    updated_left_counts[feature_count] = updated_left_count
                    updated_left_pos_counts[feature_count] = updated_left_pos_count
                    updated_right_counts[feature_count] = updated_right_count
                    updated_right_pos_counts[feature_count] = updated_right_pos_count

                    feature_count += 1

                    if meta.features[j] == chosen_feature:
                        chosen_feature_validated = 1
                        chosen_ndx = j
                        chosen_left_count = left_count
                        chosen_right_count = right_count

            # no valid features after data removal => create leaf
            if feature_count == 0:
                printf('feature_count is zero\n')
                result = -2
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
                result = -1
                free(gini_indices)
                free(distribution)
                free(valid_features)
                free(updated_left_counts)
                free(updated_left_pos_counts)
                free(updated_right_counts)
                free(updated_right_pos_counts)

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
                    result = -1
                    free(gini_indices)
                    free(distribution)

                    free(updated_left_counts)
                    free(updated_left_pos_counts)
                    free(updated_right_counts)
                    free(updated_right_pos_counts)
                    meta.features = valid_features

                else:

                    # split removal data based on the chosen feature
                    split.left_indices = <int *>malloc(chosen_left_count * sizeof(int))
                    split.left_remove_indices = <int *>malloc(chosen_left_count * sizeof(int))
                    split.right_indices = <int *>malloc(chosen_right_count * sizeof(int))
                    split.right_remove_indices = <int *>malloc(chosen_right_count * sizeof(int))
                    j = 0
                    k = 0
                    for i in range(n_samples):
                        if X[samples[i]][valid_features[chosen_ndx]] == 1:
                            split.left_indices[j] = samples[i]
                            split.left_remove_indices[j] = remove_samples[i]
                            j += 1
                        else:
                            split.right_indices[k] = samples[i]
                            split.right_remove_indices[j] = remove_samples[i]
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

    cdef int _collect_leaf_samples(self, int node_id, _Tree tree,
                                   int* remove_samples, int n_remove_samples,
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
        cdef int node_remove_count = 0
        cdef int temp_id
        cdef IntStack stack = IntStack(INITIAL_STACK_SIZE)
        stack.push(node_id)

        while not stack.is_empty():
            temp_id = stack.pop()
            node_remove_count += 1

            # leaf
            if tree.values[temp_id] >= 0:
                leaf_ids[leaf_count] = temp_id
                leaf_count += 1

            # decision node
            else:
                stack.push(tree.right_children[temp_id])
                stack.push(tree.left_children[temp_id])

                free(tree.left_counts[temp_id])
                free(tree.left_pos_counts[temp_id])
                free(tree.right_counts[temp_id])
                free(tree.right_pos_counts[temp_id])

                if temp_id != node_id:
                    free(tree.features[temp_id])

        leaf_ids = <int *>realloc(leaf_ids, leaf_count * sizeof(int))

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

        # restructure tree
        for i in range(tree.node_count):
            if tree.left_children[i] > node_remove_count:
                tree.left_children[i] -= node_remove_count

            if tree.right_children[i] > node_remove_count:
                tree.right_children[i] -= node_remove_count

        for i in range(node_id + node_remove_count, tree.node_count):
            j = i - node_remove_count
            tree.values[j] = tree.values[i]
            tree.chosen_features[j] = tree.chosen_features[i]
            tree.depth[j] = tree.depth[i]
            tree.left_children[j] = tree.left_children[i]
            tree.right_children[j] = tree.right_children[i]

            tree.count[j] = tree.count[i]
            tree.pos_count[j] = tree.pos_count[i]
            tree.feature_count[j] = tree.feature_count[i]
            tree.left_counts[j] = tree.left_counts[i]
            tree.left_pos_counts[j] = tree.left_pos_counts[i]
            tree.right_counts[j] = tree.right_counts[i]
            tree.right_pos_counts[j] = tree.right_pos_counts[i]
            tree.features[j] = tree.features[i]
            tree.leaf_samples[j] = tree.leaf_samples[i]
        tree.node_count -= node_remove_count

        rebuild_samples_ptr[0] = rebuild_samples
        return rebuild_sample_count

