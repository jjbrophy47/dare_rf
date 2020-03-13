
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdlib cimport free
from libc.stdlib cimport rand
from libc.stdlib cimport srand
from libc.stdlib cimport RAND_MAX
from libc.time cimport time

import numpy as np
cimport numpy as np

cdef inline double get_random() nogil:
    """
    Generate a random number between 0 and 1 sampled uniformly.
    """
    return rand() / RAND_MAX

@cython.cdivision(True)
cdef double _compute_gini(double count, double left_count, double right_count,
                          int left_pos_count, int right_pos_count) nogil:
    """
    Compute the Gini index of this attribute.
    """
    cdef double weight
    cdef double pos_prob
    cdef double neg_prob

    cdef double index
    cdef double left_weighted_index
    cdef double right_weighted_index

    weight = left_count / count
    pos_prob = left_pos_count / left_count
    neg_prob = 1 - pos_prob
    index = 1 - (pos_prob * pos_prob) - (neg_prob * neg_prob)
    left_weighted_index = weight * index

    weight = right_count / count
    pos_prob = right_pos_count / right_count
    neg_prob = 1 - pos_prob
    index = 1 - (pos_prob * pos_prob) - (neg_prob * neg_prob)
    right_weighted_index = weight * index

    return left_weighted_index + right_weighted_index

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int _generate_distribution(double lmbda, double* distribution,
                                double* gini_indices, int n_gini_indices) nogil:
    """
    Generate a probability distribution based on the Gini index values.
    """
    cdef int i
    cdef double normalizing_constant = 0

    cdef double min_gini = 1
    cdef int n_min = 0
    cdef int first_min = -1

    cdef bint deterministic = 0

    # find min and max Gini values
    for i in range(n_gini_indices):
        if gini_indices[i] < min_gini:
            n_min = 1
            first_min = i
            min_gini = gini_indices[i]
        elif gini_indices[i] == min_gini:
            n_min += 1

    # determine if tree is in deterministic mode
    if lmbda < 0 or exp(- lmbda * min_gini / 5) == 0:
        for i in range(n_gini_indices):
            distribution[i] = 0
        distribution[first_min] = 1

    # generate probability distribution over the features
    else:
        for i in range(n_gini_indices):
            distribution[i] = exp(- lmbda * gini_indices[i] / 5)
            normalizing_constant += distribution[i]

        for i in range(n_gini_indices):
            distribution[i] /= normalizing_constant
            # printf('distribution[%d]: %.7f\n', i, distribution[i])

    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _sample_distribution(self, double* distribution, int n_distribution) nogil:
    """
    Randomly sample a feature from the probability distribution.
    """
    cdef int i
    cdef double weight = 0

    weight = get_random()
    # printf('initial weight: %.7f\n', weight)

    for i in range(n_distribution):
        if weight < distribution[i]:
            break
        weight -= distribution[i]

    return i

# =============================================================================
# Stack data structure
# =============================================================================

cdef class Stack:
    """
    A LIFO data structure.

    Attributes
    ----------
    capacity : int
        The elements the stack can hold; if more added then ``self.stack_`` needs to be resized.
    top : int
        The number of elements currently on the stack.
    stack : StackRecord pointer
        The stack of records (upward in the stack corresponds to the right).
    """

    def __cinit__(self, int capacity):
        self.capacity = capacity
        self.top = 0
        self.stack_ = <StackRecord*> malloc(capacity * sizeof(StackRecord))

    def __dealloc__(self):
        free(self.stack_)

    cdef bint is_empty(self) nogil:
        return self.top <= 0

    cdef int push(self, int depth, int parent, bint is_left, int* samples,
                  int n_samples, int* features, int n_features) nogil:
        """
        Push a new element onto the stack.
        """
        cdef int top = self.top
        cdef StackRecord* stack = NULL

        # Resize if capacity not sufficient
        if top >= self.capacity:
            self.capacity *= 2
            self.stack_ = <StackRecord *>realloc(self.stack_, self.capacity * sizeof(StackRecord))

        stack = self.stack_
        stack[top].depth = depth
        stack[top].parent = parent
        stack[top].is_left = is_left
        stack[top].samples = samples
        stack[top].n_samples = n_samples
        stack[top].features = features
        stack[top].n_features = n_features

        # Increment stack pointer
        self.top = top + 1
        return 0

    cdef int pop(self, StackRecord* res) nogil:
        """
        Remove the top element from the stack and copy to ``res``.
        """
        cdef int top = self.top
        cdef StackRecord* stack = self.stack_

        if top <= 0:
            return -1

        res[0] = stack[top - 1]
        self.top = top - 1

        return 0


# =============================================================================
# Removal Stack structure
# =============================================================================

cdef class RemovalStack:
    """
    A LIFO data structure.

    Attributes
    ----------
    capacity : int
        The elements the stack can hold; if more added then ``self.stack_`` needs to be resized.
    top : int
        The number of elements currently on the stack.
    stack : StackRecord pointer
        The stack of records (upward in the stack corresponds to the right).
    """

    def __cinit__(self, int capacity):
        self.capacity = capacity
        self.top = 0
        self.stack_ = <RemovalStackRecord*> malloc(capacity * sizeof(RemovalStackRecord))

    def __dealloc__(self):
        free(self.stack_)

    cdef bint is_empty(self) nogil:
        return self.top <= 0

    cdef int push(self, int depth, int node_id, double parent_p, int* samples,
                  int n_samples) nogil:
        """
        Push a new element onto the stack.
        """
        cdef int top = self.top
        cdef StackRecord* stack = NULL
        cdef int num_bytes

        # Resize if capacity not sufficient
        if top >= self.capacity:
            self.capacity *= 2
            num_bytes = self.capacity * sizeof(RemovalStackRecord)
            self.stack_ = <RemovalStackRecord *>realloc(self.stack_, num_bytes)

        stack = self.stack_
        stack[top].depth = depth
        stack[top].node_id = node_id
        stack[top].parent_p = parent_p
        stack[top].samples = samples
        stack[top].remove_samples = remove_samples
        stack[top].n_samples = n_samples

        # Increment stack pointer
        self.top = top + 1
        return 0

    cdef int pop(self, StackRecord* res) nogil:
        """
        Remove the top element from the stack and copy to ``res``.
        """
        cdef int top = self.top
        cdef StackRecord* stack = self.stack_

        if top <= 0:
            return -1

        res[0] = stack[top - 1]
        self.top = top - 1

        return 0



# =============================================================================
# Basic Int Stack
# =============================================================================

cdef class IntStack:
    """
    A LIFO data structure.

    Attributes
    ----------
    capacity : int
        The elements the stack can hold; if more added then ``self.stack_`` needs to be resized.
    top : int
        The number of elements currently on the stack.
    """

    def __cinit__(self, int capacity):
        self.capacity = capacity
        self.top = 0

    def __dealloc__(self):
        free(self.stack_)

    cdef bint is_empty(self) nogil:
        return self.top <= 0

    cdef int push(self, int node_id) nogil:
        """
        Push a new element onto the stack.
        """
        cdef int top = self.top

        # Resize if capacity not sufficient
        if top >= self.capacity:
            self.capacity *= 2
            self.stack_ = <int *>realloc(self.stack_, self.capacity * sizeof(int))

        stack = self.stack_
        stack[top].node_id = node_id

        # Increment stack pointer
        self.top = top + 1
        return 0

    cdef int pop(self, StackRecord* res) nogil:
        """
        Remove the top element from the stack and copy to ``res``.
        """
        cdef int top = self.top
        cdef StackRecord* stack = self.stack_

        if top <= 0:
            return -1

        res[0] = stack[top - 1]
        self.top = top - 1

        return 0
