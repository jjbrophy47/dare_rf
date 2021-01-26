import numpy as np
cimport numpy as np

from ._utils cimport DTYPE_t
from ._utils cimport SIZE_t
from ._utils cimport INT32_t
from ._utils cimport UINT32_t

cdef void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil
