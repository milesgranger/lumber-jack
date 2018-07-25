# -*- coding: utf-8 -*-
# distutils: language = c++

import logging
import cloudpickle
import numpy as np
import pandas as pd

cimport numpy as np

from libc.string cimport memcpy
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from libcpp cimport bool
from cython cimport view
from lumberjack.cython.includes cimport free_data, DataPtr, DType, Tag
cimport lumberjack.cython.operators as ops

logger = logging.getLogger(__name__)

np.import_array()

cdef class _DataPtr:
    """
    Holds generic access to various data types from a DataPtr
    """
    # Flag to avoid double freeing, when inplace ops are done in rust, the original
    # data is consumed and freed simultaneously
    cdef bool is_owner

    # Possible array pointers for different dtypes
    cdef double* vec_ptr_float64
    cdef np.int32_t* vec_ptr_int32

    # Static attrs across all dtypes of a DataPtr object.
    cdef readonly view.array array_view
    cdef readonly int len
    cdef DataPtr data_ptr

    @staticmethod
    cdef _DataPtr from_ptr(DataPtr ptr):
        cdef _DataPtr _data_ptr
        _data_ptr = _DataPtr()

        if ptr.tag == Tag.Tag_Float64:
            _data_ptr.vec_ptr_float64 = ptr.float64.data_ptr
            _data_ptr.array_view = <double[:ptr.float64.len]> ptr.float64.data_ptr
            _data_ptr.len = ptr.float64.len

        elif ptr.tag == Tag.Tag_Int32:
            _data_ptr.vec_ptr_int32 = ptr.int32.data_ptr
            _data_ptr.array_view = <np.int32_t[:ptr.int32.len]> ptr.int32.data_ptr
            _data_ptr.len = ptr.int32.len

        else:
            raise ValueError('Got unknown Dtype: {}'.format(ptr.tag))

        _data_ptr.data_ptr = ptr
        _data_ptr.is_owner = True
        return _data_ptr

    @staticmethod
    cdef _DataPtr from_ptr_ref(DataPtr *ptr):

        cdef _DataPtr _data_ptr
        _data_ptr = _DataPtr()

        if ptr[0].tag == Tag.Tag_Float64:
            _data_ptr.vec_ptr_float64 = ptr[0].float64.data_ptr
            _data_ptr.array_view = <double[:ptr[0].float64.len]> ptr[0].float64.data_ptr
            _data_ptr.len = ptr[0].float64.len

        elif ptr[0].tag == Tag.Tag_Int32:
            _data_ptr.vec_ptr_int32 = ptr[0].int32.data_ptr
            _data_ptr.array_view = <np.int32_t[:ptr[0].int32.len]> ptr[0].int32.data_ptr
            _data_ptr.len = ptr[0].int32.len

        else:
            raise ValueError('Got unknown Dtype: {}'.format(ptr[0].tag))

        _data_ptr.data_ptr = ptr[0]
        _data_ptr.is_owner = False

        return _data_ptr


    def __dealloc__(self):
        if self.is_owner:
            if self.vec_ptr_float64 != NULL or \
                    self.vec_ptr_int32 != NULL:
                free_data(self.data_ptr)



cdef class LumberJackSeries(object):
    """
    LumberJackSeries

    Some implementations of Numpy / Pandas functionality with bindings to Rust.
    """
    cdef _DataPtr _data_ptr
    cdef DataPtr *data_ptr

    def __getstate__(self):
        return (self._get_state(),)

    def __setstate__(self, state):
        self._set_state(*state)

    cpdef bytes _get_state(self):
        return <bytes>(<char *>self.data_ptr)[:sizeof(DataPtr)]

    cpdef void _set_state(self, bytes data):
        PyMem_Free(self.data_ptr)
        self.data_ptr = <DataPtr*>PyMem_Malloc(sizeof(DataPtr))
        if not self.data_ptr:
            raise MemoryError()
        memcpy(self.data_ptr, <char *>data, sizeof(DataPtr))
        self._data_ptr = _DataPtr.from_ptr_ref(self.data_ptr)


    cpdef map(self, func):
        cdef bytes  func_pickled = cloudpickle.dumps(func)
        cdef np.ndarray array = np.fromstring(func_pickled, dtype=np.uint8)

        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array) # Makes a contiguous copy of the numpy array.

        cdef np.uint8_t[::1] arr_view = array
        return ops.series_map(self._data_ptr.data_ptr, &arr_view[0], array.shape[0])

    cpdef _scalar_arithmetic_factory(self, double scalar, str op, bool inplace):
        """
        Helper function to facilitate dunder methods requiring access to _DataPtr object
        which will not work inside of those.
        """
        cdef DataPtr ptr
        if op == 'mul':
            ptr = ops.multiply_by_scalar(self._data_ptr.data_ptr, scalar, inplace)
        elif op == 'add':
            ptr = ops.add_by_scalar(self._data_ptr.data_ptr, scalar, inplace)
        else:
            raise ValueError('Unknown operation: {}'.format(op))

        # If this was in inplace op, rust has already consumed the data, avoid double free
        if inplace:
            self._data_ptr.is_owner = False
        series = LumberJackSeries()
        series._data_ptr = _DataPtr.from_ptr(ptr)
        return series

    def __mul__(self, other):
        return self._scalar_arithmetic_factory(float(other), 'mul', False)

    def __imul__(self, other):
        self = self._scalar_arithmetic_factory(float(other), 'mul', True)
        return self

    def __add__(self, other):
        return self._scalar_arithmetic_factory(float(other), 'add', False)

    def __iadd__(self, other):
        self =  self._scalar_arithmetic_factory(float(other), 'add', True)
        return self

    @staticmethod
    def arange(int start, int stop):
        """
        This is ~2x faster than numpy's arange (tested 100000 times with range 0-100000)
        """
        cdef DataPtr ptr = ops.arange(start, stop, DType.Int32)
        cdef LumberJackSeries series = LumberJackSeries()
        series._data_ptr = _DataPtr.from_ptr(ptr)
        series.data_ptr = &series._data_ptr.data_ptr
        return series

    def mean(self):
        cdef double avg
        avg = ops.mean(self._data_ptr.data_ptr)
        return avg

    def sum(self):
        cdef DataPtr ptr = ops.sum(self._data_ptr.data_ptr)
        series = LumberJackSeries()
        series._data_ptr = _DataPtr.from_ptr(ptr)
        return series._data_ptr.array_view[0]

    def cumsum(self):
        cdef DataPtr ptr = ops.cumsum(self._data_ptr.data_ptr)
        series = LumberJackSeries()
        series._data_ptr = _DataPtr.from_ptr(ptr)
        return series

    def to_cython_array_view(self):
        """
        Provide a cython array view to the data
        """
        return self._data_ptr.array_view

    def to_numpy(self):
        """
        Convert this to numpy array
        """
        cdef np.ndarray array = np.asarray(self._data_ptr.array_view)
        return array

    def __len__(self):
        return self._data_ptr.len

    def __iter__(self):
        return (self._data_ptr.array_view[i] for i in range(self._data_ptr.len))

    def __getattr__(self, item):
        def method(*args, **kwargs):
            series = pd.Series(self.to_numpy())  # TODO: pass more features to constructor as we add them.
            return getattr(series, item)(*args, **kwargs)
        return method

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

        #series_ptr = from_numpy_ptr(&arr_view[0], array.shape[0])
        #series = create_lumberjack_series_from_ptr(series_ptr)
        #return series

    def __repr__(self):
        return 'LumberJackSeries(length: {})'.format(self._data_ptr.len)


