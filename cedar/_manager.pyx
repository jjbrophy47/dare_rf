"""
Module that handles all manipulations to the database.
"""
cimport cython

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf
from libc.time cimport time

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport copy_int_array

cdef int UNDEF = -1

# =====================================
# Manager
# =====================================

cdef class _DataManager:
    """
    Database manager.
    """

    property n_samples:
        def __get__(self):
            return self.n_samples

    property n_features:
        def __get__(self):
            return self.n_features

    property n_add_indices:
        def __get__(self):
            return self.n_add_indices

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, int[:, :] X_in, int[:] y_in):
        """
        Constructor.
        """
        cdef int n_samples = X_in.shape[0]
        cdef int n_features = X_in.shape[1]

        cdef int** X = <int **>malloc(n_samples * sizeof(int *))
        cdef int* y = <int *>malloc(n_samples * sizeof(int))

        cdef int *vacant = NULL

        cdef int i
        cdef int j
        cdef int result

        # copy data into C pointer arrays
        for i in range(n_samples):
            X[i] = <int *>malloc(n_features * sizeof(int))
            for j in range(n_features):
                X[i][j] = X_in[i][j]
            y[i] = y_in[i]

        self.X = X
        self.y = y
        self.n_samples = n_samples
        self.n_features = n_features
        self.vacant = vacant
        self.n_vacant = 0

    def __dealloc__(self):
        """
        Destructor.
        """
        for i in range(self.n_samples + self.n_vacant):
            if self.X[i]:
                free(self.X[i])
        free(self.X)
        free(self.y)
        if self.vacant:
            free(self.vacant)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int check_sample_validity(self, int *samples, int n_samples) nogil:
        """
        Checks to make sure `samples` are in the database.
        Returns -1 if a sample is not available; 0 otherwise.
        """
        cdef int *vacant = self.vacant
        cdef int n_vacant = self.n_vacant

        cdef int result = 0
        cdef int i
        cdef int j

        if n_vacant > 0:

            for i in range(n_samples):
                for j in range(n_vacant):

                    if samples[i] == vacant[j]:
                        result = -1
                        break

                if result == -1:
                    break

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void get_data(self, int*** X_ptr, int** y_ptr) nogil:
        """
        Receive pointers to the data.
        """
        X_ptr[0] = self.X
        y_ptr[0] = self.y

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void remove_data(self, int[:] samples):
        """
        Remove selected samples from the database.
        """

        # parameters
        cdef int** X = self.X
        cdef int* y = self.y
        cdef int *vacant = self.vacant
        cdef int n_vacant = self.n_vacant

        cdef int n_samples = samples.shape[0]
        cdef int updated_n_vacant = n_vacant + n_samples

        cdef int i

        # realloc vacant array
        if n_vacant == 0:
            vacant = <int *>malloc(updated_n_vacant * sizeof(int))

        elif updated_n_vacant > n_vacant:
            vacant = <int *>realloc(vacant, updated_n_vacant * sizeof(int))

        # remove data and save the deleted indices
        for i in range(n_samples):
            free(X[samples[i]])
            y[samples[i]] = UNDEF
            vacant[n_vacant + i] = samples[i]

        self.n_samples -= n_samples
        self.n_vacant = updated_n_vacant
        self.vacant = vacant

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void add_data(self, int[:, :] X_in, int[:] y_in):
        """
        Adds data to the database.
        """

        # parameters
        cdef int** X = self.X
        cdef int*  y = self.y
        cdef int*  vacant = self.vacant
        cdef int   n_vacant = self.n_vacant
        cdef int   n_samples = self.n_samples
        cdef int   n_features = self.n_features

        cdef int  n_new_samples = X_in.shape[0]
        cdef int  updated_n_samples = n_samples + n_new_samples
        cdef int* add_indices = <int *>malloc(n_new_samples * sizeof(int))

        cdef int i
        cdef int j
        cdef int k

        # grow database
        if updated_n_samples > n_samples + n_vacant:
            X = <int **>realloc(X, updated_n_samples * sizeof(int *))
            y = <int *>realloc(y, updated_n_samples * sizeof(int))

        # copy samples to the database
        for i in range(n_new_samples):

            # recycle available ID
            if n_vacant > 0:
                n_vacant -= 1
                j = vacant[n_vacant]

            # use a new sample ID
            else:
                j = n_samples

            # copy sample
            X[j] = <int *>malloc(n_features * sizeof(int))
            for k in range(n_features):
                X[j][k] = X_in[i][k]
            y[j] = y_in[i]

            add_indices[i] = j
            n_samples += 1

        # adjust vacant array
        if n_vacant == 0:
            free(vacant)
            vacant = NULL
        elif n_vacant > 0:
            vacant = <int *>realloc(vacant, n_vacant * sizeof(int))

        self.X = X
        self.y = y
        self.vacant = vacant
        self.add_indices = add_indices
        self.n_vacant = n_vacant
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_add_indices = n_new_samples

    cpdef void clear_add_indices(self):
        """
        Clear the indices that have been added to the database.
        """
        free(self.add_indices)
        self.n_add_indices = 0
        self.add_indices = NULL

    cdef int* get_add_indices(self) nogil:
        """
        Return a copy of the add indices.
        """
        return copy_int_array(self.add_indices, self.n_add_indices)
