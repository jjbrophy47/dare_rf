import numpy as np
cimport numpy as np

from ._utils cimport DTYPE_t
from ._utils cimport SIZE_t
from ._utils cimport INT32_t
from ._utils cimport UINT32_t

cdef class _DataManager:
    """
    Manages the database.
    """

    # Internal structure
    cdef SIZE_t    n_samples       # Number of samples
    cdef SIZE_t    n_features      # Number of features
    cdef DTYPE_t** X               # Sample data
    cdef INT32_t*  y               # Label data
    cdef SIZE_t    n_vacant        # Number of empty indices in the database
    cdef SIZE_t*   vacant          # Empty indices in the database
    cdef SIZE_t*   add_indices     # Added indices in the database
    cdef SIZE_t    n_add_indices   # Number of indices added to the database

    # Python API
    cpdef void remove_data(self, int[:] samples)
    cpdef void add_data(self, float[:, :] X_in, int[:] y_in)
    cpdef void clear_add_indices(self)

    # C API
    cdef INT32_t check_sample_validity(self, SIZE_t *samples, SIZE_t n_samples) nogil
    cdef void get_data(self, DTYPE_t*** X_ptr, INT32_t** y_ptr) nogil
    cdef SIZE_t* get_add_indices(self) nogil
    cdef np.ndarray _get_int_ndarray(self, SIZE_t *data, SIZE_t n_elem)
