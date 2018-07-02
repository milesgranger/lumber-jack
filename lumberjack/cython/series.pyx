# -*- coding: utf-8 -*-
# distutils: language = c++

from .includes cimport add_two_in_rust


cpdef float sum_two(float a, float b):
    cdef float result
    result = add_two_in_rust(a, b)
    return result
