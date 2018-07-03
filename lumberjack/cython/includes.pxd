import ctypes

cdef extern from "./../rust/liblumberjack.h":
    float add_two_in_rust(float a, float b);
    void double_array(double *);

    cdef struct CVector:
        double * data
        int len

    CVector create_array();