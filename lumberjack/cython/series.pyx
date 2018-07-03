# -*- coding: utf-8 -*-
# distutils: language = c++


from .includes cimport add_two_in_rust, double_array, create_lumberjack_series, free_vector, LumberJackSeriesPtr
import numpy as np
cimport numpy as np


np.import_array()

cdef _free_vector(double * ptr, int len):
    free_vector(ptr, len)

cdef np.ndarray create_array_from_rust_vector(LumberJackSeriesPtr vector):
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> vector.len
    array = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT64, vector.ptr)
    return array

cpdef np.ndarray _create_array():
    vector = get_rust_vector()
    array = create_array_from_rust_vector(vector)
    return array

cpdef LumberJackSeries get_lumberjack_vector():
    vector = create_lumberjack_series()
    v = LumberJackSeries.create(vector)
    return v

cdef LumberJackSeriesPtr get_rust_vector():
    vector = create_lumberjack_series()
    return vector


cpdef float sum_two(float a, float b):
    cdef float result
    result = add_two_in_rust(a, b)
    return result


cpdef double_array_in_rust(arr):

    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr) # Makes a contiguous copy of the numpy array.

    cdef double[::1] arr_view = arr
    double_array(&arr_view[0])
    return arr


cdef class LumberJackSeries:

    cdef double * ptr
    cpdef int len

    @staticmethod
    cdef create(LumberJackSeriesPtr vector):
        series = LumberJackSeries()
        series.ptr = vector.ptr
        series.len = vector.len
        return series


    def __dealloc__(self):
        print('Deallocating rust series!')
        if self.ptr != NULL:
            _free_vector(self.ptr, self.len)


