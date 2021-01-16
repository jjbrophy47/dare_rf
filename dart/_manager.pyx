"""
Module that handles all manipulations to the database.
"""
from cpython cimport Py_INCREF
from cpython.ref cimport PyObject

cimport cython

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf
from libc.time cimport time

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport copy_indices

cdef INT32_t UNDEF = -1

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

    property add_indices:
        def __get__(self):
            return self._get_int_ndarray(self.add_indices, self.n_add_indices)

    property n_add_indices:
        def __get__(self):
            return self.n_add_indices

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, float[:, :] X_in, int[:] y_in):
        """
        Constructor.
        """
        cdef SIZE_t n_samples = X_in.shape[0]
        cdef SIZE_t n_features = X_in.shape[1]

        cdef DTYPE_t** X = <DTYPE_t **>malloc(n_samples * sizeof(DTYPE_t *))
        cdef INT32_t*  y = <INT32_t *>malloc(n_samples * sizeof(INT32_t))

        cdef SIZE_t *vacant = NULL

        cdef SIZE_t i
        cdef SIZE_t j

        # copy data into C pointer arrays
        for i in range(n_samples):
            X[i] = <DTYPE_t *>malloc(n_features * sizeof(DTYPE_t))
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
    cdef INT32_t check_sample_validity(self, SIZE_t *samples, SIZE_t n_samples) nogil:
        """
        Checks to make sure `samples` are in the database.
        Returns -1 if a sample is not available; 0 otherwise.
        """
        cdef SIZE_t *vacant = self.vacant
        cdef SIZE_t n_vacant = self.n_vacant

        cdef INT32_t result = 0
        cdef SIZE_t i
        cdef SIZE_t j

        for i in range(n_samples):

            # check if index number is valid
            if samples[i] < 0:
                result = -1

            for j in range(n_vacant):

                # check if sample has already been deleted
                if samples[i] == vacant[j]:
                    result = -1
                    break

            if result == -1:
                break

        return result

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void get_data(self, DTYPE_t*** X_ptr, INT32_t** y_ptr) nogil:
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
        cdef DTYPE_t** X = self.X
        cdef INT32_t*  y = self.y
        cdef SIZE_t*   vacant = self.vacant
        cdef SIZE_t       n_vacant = self.n_vacant

        cdef SIZE_t n_samples = samples.shape[0]
        cdef SIZE_t updated_n_vacant = n_vacant + n_samples

        cdef SIZE_t i

        # realloc vacant array
        if n_vacant == 0:
            vacant = <SIZE_t *>malloc(updated_n_vacant * sizeof(SIZE_t))

        elif updated_n_vacant > n_vacant:
            vacant = <SIZE_t *>realloc(vacant, updated_n_vacant * sizeof(SIZE_t))

        # remove data and save the deleted indices
        for i in range(n_samples):
            free(X[samples[i]])
            X[samples[i]] = NULL
            y[samples[i]] = UNDEF
            vacant[n_vacant + i] = samples[i]

        self.n_samples -= n_samples
        self.n_vacant = updated_n_vacant
        self.vacant = vacant

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void add_data(self, float[:, :] X_in, int[:] y_in):
        """
        Adds data to the database.
        """

        # parameters
        cdef DTYPE_t** X = self.X
        cdef INT32_t*  y = self.y
        cdef SIZE_t*   vacant = self.vacant
        cdef SIZE_t    n_vacant = self.n_vacant
        cdef SIZE_t    n_samples = self.n_samples
        cdef SIZE_t    n_features = self.n_features

        cdef SIZE_t  n_new_samples = X_in.shape[0]
        cdef SIZE_t  updated_n_samples = n_samples + n_new_samples
        cdef SIZE_t* add_indices = <SIZE_t *>malloc(n_new_samples * sizeof(SIZE_t))

        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t k

        # grow database
        if updated_n_samples > n_samples + n_vacant:
            X = <DTYPE_t **>realloc(X, updated_n_samples * sizeof(DTYPE_t *))
            y = <INT32_t *>realloc(y, updated_n_samples * sizeof(INT32_t))

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
            X[j] = <DTYPE_t *>malloc(n_features * sizeof(DTYPE_t))
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
            vacant = <SIZE_t *>realloc(vacant, n_vacant * sizeof(SIZE_t))

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

    cdef SIZE_t* get_add_indices(self) nogil:
        """
        Return a copy of the add indices.
        """
        return copy_indices(self.add_indices, self.n_add_indices)

    cdef np.ndarray _get_int_ndarray(self, SIZE_t *data, SIZE_t n_elem):
        """
        Wraps value as a 1-d NumPy array.
        The array keeps a reference to this Tree, which manages the underlying memory.
        """
        cdef SIZE_t shape[1]
        shape[0] = n_elem
        cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, data)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr
