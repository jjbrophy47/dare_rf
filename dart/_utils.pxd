import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_intp    SIZE_t           # Type for indices and counters
ctypedef np.npy_int32   INT32_t          # Signed 32 bit integer
ctypedef np.npy_uint32  UINT32_t         # Unsigned 32 bit integer

from ._tree cimport Node
from ._tree cimport Threshold
from ._tree cimport Feature
from ._tree cimport IntList
from ._splitter cimport SplitRecord

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


# sampling methods
cdef UINT32_t our_rand_r(UINT32_t* seed) nogil

cdef SIZE_t rand_int(SIZE_t low, SIZE_t high,
                     UINT32_t* random_state) nogil

cdef double rand_uniform(double    low,
                         double    high,
                         UINT32_t* random_state) nogil

# split score methods
cdef DTYPE_t compute_split_score(bint    use_gini,
                                 DTYPE_t count,
                                 DTYPE_t left_count,
                                 DTYPE_t right_count,
                                 SIZE_t  left_pos_count,
                                 SIZE_t  right_pos_count) nogil

cdef DTYPE_t compute_gini(DTYPE_t count,
                          DTYPE_t left_count,
                          DTYPE_t right_count,
                          SIZE_t  left_pos_count,
                          SIZE_t  right_pos_count) nogil

cdef DTYPE_t compute_entropy(DTYPE_t count,
                             DTYPE_t left_count,
                             DTYPE_t right_count,
                             SIZE_t  left_pos_count,
                             SIZE_t  right_pos_count) nogil

# Feature / threshold methodss
cdef Feature* create_feature(SIZE_t feature_index) nogil
cdef Threshold* create_threshold(DTYPE_t value,
                                 SIZE_t n_left_samples,
                                 SIZE_t n_right_samples) nogil
cdef Feature* copy_feature(Feature* feature) nogil
cdef Threshold* copy_threshold(Threshold* threshold) nogil
cdef void free_feature(Feature* feature) nogil
cdef void free_features(Feature** features,
                        SIZE_t n_features) nogil

# IntList methods
cdef IntList* create_intlist(SIZE_t n_elem, bint initialize) nogil
cdef IntList* copy_intlist(IntList* obj, SIZE_t n_elem) nogil
cdef void free_intlist(IntList* obj) nogil

# Array methods
cdef SIZE_t* convert_int_ndarray(np.ndarray arr)
cdef INT32_t* copy_int_array(INT32_t* arr,
                             SIZE_t n_elem) nogil
cdef SIZE_t* copy_indices(SIZE_t* arr,
                          SIZE_t n_elem) nogil

# node methods
cdef void split_samples(Node*        node,
                        DTYPE_t**    X,
                        INT32_t*     y,
                        IntList*     samples,
                        SplitRecord* split) nogil

cdef void dealloc(Node *node) nogil
