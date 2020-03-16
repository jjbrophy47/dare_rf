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

cdef int _UNDEFINED = -2

# =====================================
# Manager
# =====================================

cdef class _DataManager:
    """
    Database manager.
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, object X_in, np.ndarray y_in, np.ndarray f_in):
        """
        Constructor.
        """
        cdef int n_samples = X_in.shape[0]
        cdef int n_features = X_in.shape[1]

        cdef int** X = <int **>malloc(n_samples * sizeof(int *))
        cdef int* y = <int *>malloc(n_samples * sizeof(int))
        cdef int* f = <int *>malloc(n_features * sizeof(int))

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

        for i in range(n_features):
            f[i] = f_in[i]

        self.X = X
        self.y = y
        self.f = f
        self.vacant = vacant
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_vacant = 0

    def __dealloc__(self):
        """
        Destructor.
        """
        free(self.X)
        free(self.y)
        free(self.f)
        if self.vacant:
            free(self.vacant)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int get_all_data(self, int*** X_ptr, int** y_ptr, int**f_ptr,
                          int* n_samples, int* n_features) nogil:
        """
        Receive pointers to the data.
        """
        cdef int result = 0

        X_ptr[0] = self.X
        y_ptr[0] = self.y
        f_ptr[0] = self.f
        n_samples[0] = self.n_samples
        n_features[0] = self.n_features

        printf('got data!\n')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int get_data(self, int* samples, int n_samples,
                      int ***X_sub_ptr, int **y_sub_ptr) nogil:
        """
        Return a copied sliced view of the data.
        """

        # parameters
        cdef int** X = self.X
        cdef int* y = self.y

        cdef int** X_sub = <int **>malloc(n_samples * sizeof(int *))
        cdef int* y_sub = <int *>malloc(n_samples * sizeof(int))

        cdef int i
        cdef int result = 0

        for i in range(n_samples):
            X_sub[i] = X[i]
            y_sub[i] = y[i]

        # populate structures
        X_sub_ptr[0] = X_sub
        y_sub_ptr[0] = y_sub

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int remove_data(self, int* samples, int n_samples) nogil:

        # parameters
        cdef int** X = self.X
        cdef int* y = self.y
        cdef int* vacant = self.vacant
        cdef int n_vacant = self.n_vacant
        cdef int updated_n_vacant = n_vacant + n_samples

        cdef int i
        cdef int result = 0

        for i in range(n_samples):
            free(X[i])
            y[i] = _UNDEFINED

        # TODO: could hold off on this until data needs to be added
        # keep track of vacant indices to use later
        if not vacant:
            vacant = <int *>malloc(updated_n_vacant * sizeof(int))
        elif updated_n_vacant > n_vacant:
            vacant = <int *>realloc(vacant, updated_n_vacant * sizeof(int))

            i = n_vacant
            while i < updated_n_vacant:
                vacant[i] = samples[i]
                i += 1

        self.n_samples -= n_samples
        self.n_vacant = updated_n_vacant

        return result
