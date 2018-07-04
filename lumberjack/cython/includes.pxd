import ctypes

cdef extern from "./../rust/liblumberjack.h":
    float add_two_in_rust(float a, float b);
    void double_array(double* ptr);

    # Allow rust to remove a vector that it created and passed to Python
    void free_vector(double* ptr, int length);

    cdef struct LumberJackSeriesPtr:
        double* ptr
        int len

    LumberJackSeriesPtr create_lumberjack_series();
    LumberJackSeriesPtr from_numpy_ptr(double * ptr, int length)



