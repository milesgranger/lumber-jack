import ctypes

cdef extern from "./../rust/liblumberjack.h":
    float add_two_in_rust(float a, float b);
    void double_array(double *);

    cdef struct LumberJackVectorPtr:
        double * data
        int len

    LumberJackVectorPtr create_array();