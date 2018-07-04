# -*- coding: utf-8 -*-
# distutils: language = c++


from .includes cimport create_lumberjack_series, free_vector, LumberJackSeriesPtr, from_numpy_ptr
import numpy as np
cimport numpy as np
from cython cimport view


np.import_array()


cdef create_lumberjack_series_from_ptr(LumberJackSeriesPtr series_ptr):
    """ 
    Create a LumberJackSeries object from a LumberjackSeriesPtr Cython definition
    """
    series = LumberJackSeries()
    series.ptr = series_ptr.ptr
    series.len = series_ptr.len
    series.lj_series_ptr = series_ptr
    return series


cdef class LumberJackSeries:

    cdef LumberJackSeriesPtr lj_series_ptr
    cdef double * ptr
    cdef readonly int len


    @staticmethod
    def linspace(int start, int stop):
        series_ptr = create_lumberjack_series()
        return create_lumberjack_series_from_ptr(series_ptr)


    def to_cython_array_view(self):
        """
        Provide a cython array view to the data
        """
        cdef view.array array_view = <double[:self.len]> self.ptr
        return array_view

    def to_numpy(self):
        """
        Convert this to numpy array
        """
        cdef np.ndarray array = np.asarray(<double[:self.len]> self.ptr)
        return array

    @staticmethod
    cdef LumberJackSeries from_lumberjack_ptr(LumberJackSeriesPtr series_ptr):
        """
        Create series from Cython/C struct of LumberJackSeriesPtr which holds meta
        data about the underlying Rust vector
        """
        return create_lumberjack_series_from_ptr(series_ptr)

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


