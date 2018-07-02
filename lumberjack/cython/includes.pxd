cdef extern from "./../rust/rust_bindings.h":
    float add_two_in_rust(float a, float b);
