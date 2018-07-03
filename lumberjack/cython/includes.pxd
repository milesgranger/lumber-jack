import ctypes

cdef extern from "./../rust/liblumberjack.h":
    float add_two_in_rust(float a, float b);
    void double_array(double*);
    void free_vector(double*, int);

    cdef struct LumberJackVectorPtr:
        double* ptr
        int len

    LumberJackVectorPtr create_vector();