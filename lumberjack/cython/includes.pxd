cimport numpy as np

cdef extern from "./../rust/liblumberjack.h":

    # Allow rust to remove a vector that it created and passed to Python
    void free_vector(double* ptr, int length);
    cdef struct LumberJackSeriesPtr:
        double* data_ptr
        int len

    np.int32_t **get_boxed_int()

    LumberJackSeriesPtr arange(int start, int stop);
    LumberJackSeriesPtr from_numpy_ptr(double * ptr, int length)



