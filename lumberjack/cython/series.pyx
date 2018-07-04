# -*- coding: utf-8 -*-
# distutils: language = c++


from .includes cimport add_two_in_rust, double_array, create_lumberjack_series, free_vector, LumberJackSeriesPtr, from_numpy_ptr
import numpy as np
cimport numpy as np


np.import_array()

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
    v = LumberJackSeries.from_lumberjack_ptr(vector)
    return v

cdef LumberJackSeriesPtr get_rust_vector():
    vector = create_lumberjack_series()
    return vector


cpdef float sum_two(float a, float b):
    cdef float result
    result = add_two_in_rust(a, b)
    return result


cdef create_lumberjack_series_from_ptr(LumberJackSeriesPtr series_ptr):
    """ 
    Create a LumberJackSeries object from a LumberjackSeriesPtr Cython definition
    """
    series = LumberJackSeries()
    series.ptr = series_ptr.ptr
    series.len = series_ptr.len
    series.lj_series_ptr = series_ptr
    return series


cpdef double_array_in_rust(arr):

    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr) # Makes a contiguous copy of the numpy array.

    cdef double[::1] arr_view = arr
    double_array(&arr_view[0])
    return arr


cdef class LumberJackSeries:

    cdef LumberJackSeriesPtr lj_series_ptr
    cdef double * ptr
    cdef readonly int len

    @staticmethod
    cdef LumberJackSeries from_lumberjack_ptr(LumberJackSeriesPtr series_ptr):
        """
        Create series from Cython/C struct of LumberJackSeriesPtr which holds meta
        data about the underlying Rust vector
        """
        series = create_lumberjack_series_from_ptr(series_ptr)
        return series

    @staticmethod
    def from_numpy(np.ndarray array):
        """
        Create series from 1d numpy array
        """
        if array.ndim > 1:
            raise ValueError('Cannot make Series from an array with >1 shape, needs to be 1 dimensional. '
                             'The passed array had shape: {}'.format(array.ndim))

        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array) # Makes a contiguous copy of the numpy array.

        cdef double[::1] arr_view = array

        series_ptr = from_numpy_ptr(&arr_view[0], array.shape[0])
        series = create_lumberjack_series_from_ptr(series_ptr)
        return series

    def __repr__(self):
        return 'LumberJackSeries(length: {})'.format(self.len)


    def __dealloc__(self):
        print('Deallocating rust series!')
        if self.ptr != NULL:
            free_vector(self.ptr, self.len)


