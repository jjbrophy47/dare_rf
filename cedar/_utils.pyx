
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
                  int n_samples, int* features, int n_features) nogil except -1:
        """
        Push a new element onto the stack.
        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef int top = self.top
        cdef StackRecord* stack = NULL

        # Resize if capacity not sufficient
        if top >= self.capacity:
            self.capacity *= 2
            self.stack_ = <StackRecord *>realloc(self.stack_, self.capacity * sizeof(StackRecord))

            # # Since safe_realloc can raise MemoryError, use `except -1`
            # safe_realloc(&self.stack_, self.capacity)

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
        Returns 0 if pop was successful (and ``res`` is set); -1 otherwise.
        """
        cdef int top = self.top
        cdef StackRecord* stack = self.stack_

        if top <= 0:
            return -1

        res[0] = stack[top - 1]
        self.top = top - 1

        return 0
