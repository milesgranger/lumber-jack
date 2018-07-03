import ctypes

cdef extern from "./../rust/liblumberjack.h":
    float add_two_in_rust(float a, float b);
    void double_array(double*);
    void free_vector(double*, int);

    cdef struct LumberJackSeriesPtr:
        double* ptr
        int len

    LumberJackSeriesPtr create_lumberjack_series();



