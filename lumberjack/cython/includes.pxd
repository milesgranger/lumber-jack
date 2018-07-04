import ctypes

cdef extern from "./../rust/liblumberjack.h":

    # Allow rust to remove a vector that it created and passed to Python
    void free_vector(double* ptr, int length);

    cdef struct LumberJackSeriesPtr:
        double* ptr
        int len

    LumberJackSeriesPtr create_lumberjack_series();
    LumberJackSeriesPtr from_numpy_ptr(double * ptr, int length)



