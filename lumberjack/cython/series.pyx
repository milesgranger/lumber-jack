# -*- coding: utf-8 -*-
# distutils: language = c++

import logging
import cloudpickle
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


cimport numpy as np
from cython.parallel import prange, parallel
from libcpp cimport bool
from cython cimport view
from lumberjack.cython.includes cimport free_data, DataPtr, DType, Tag, TagDataElement, DataElement, verify, copy_ptr, from_numpy_ptr
cimport lumberjack.cython.operators as ops

logger = logging.getLogger(__name__)

np.import_array()

ctypedef fused int_or_double:
    int
    double


# Presently not used, can't determine how to specialize it.
# here for reference.
def apply_func(bytes func, int[::1] series, double[::1] target):

    f = cloudpickle.loads(func)

    for i in range(len(series)):
        target[i] = f(series[i])

cpdef object rebuild(bytes data):
    cdef char* _ptr = data
    cdef DataPtr* ptr = <DataPtr*>_ptr
    series = LumberJackSeries.from_ptr(ptr[0])
    series.is_owner = False
    return series

cdef class LumberJackSeries(object):
    """
    LumberJackSeries

    Some implementations of Numpy / Pandas functionality with bindings to Rust.
    """
    cdef readonly bool is_owner
    cdef DType dtype

    # Possible array pointers for different dtypes
    cdef double*     vec_ptr_float64
    cdef np.int32_t* vec_ptr_int32

    # Static attrs across all dtypes of a DataPtr object.
    cdef readonly view.array array_view
    cdef readonly int len
    cdef DataPtr  data_ptr

    def _get_state(self):
        return <bytes>(<char*>&self.data_ptr)[:sizeof(DataPtr)]

    def __reduce__(self):
        data = self._get_state()
        return (rebuild, (data, ))

    @staticmethod
    cdef LumberJackSeries from_ptr(DataPtr ptr):
        cdef LumberJackSeries series
        series = LumberJackSeries()

        if ptr.tag == Tag.Tag_Float64:
            series.dtype = DType.Float64
            series.vec_ptr_float64 = ptr.float64.data_ptr
            series.array_view = <double[:ptr.float64.len]> ptr.float64.data_ptr
            series.len = ptr.float64.len

        elif ptr.tag == Tag.Tag_Int32:
            series.dtype = DType.Int32
            series.vec_ptr_int32 = ptr.int32.data_ptr
            series.array_view = <np.int32_t[:ptr.int32.len]> ptr.int32.data_ptr
            series.len = ptr.int32.len

        else:
            raise ValueError('Got unknown Dtype: {}'.format(ptr.tag))

        series.data_ptr = ptr
        series.is_owner = True
        return series

    def __dealloc__(self):
        if &self.data_ptr != NULL and self.is_owner:
            free_data(self.data_ptr)

    cpdef astype(self, type dtype):
        cdef DType _dtype
        if dtype == float:
            _dtype = DType.Float64
        else:
            raise ValueError('DType of "{}" not supported, please file an issue! :)'.format(dtype))
        ptr =  ops.astype(self.data_ptr, DType.Float64)
        series = LumberJackSeries.from_ptr(ptr)
        return series

    cdef np.ndarray _convert_byte_string_to_array(self, bytes string):
        cdef np.ndarray byte_array = np.fromstring(string, dtype=np.uint8)

        if not byte_array.flags['C_CONTIGUOUS']:
            byte_array = np.ascontiguousarray(byte_array) # Makes a contiguous copy of the numpy array.

        return byte_array


    cpdef LumberJackSeries map(self, object func, out_dtype: type=float):

        cdef:
            LumberJackSeries target = self.astype(out_dtype)
            int i

        for i, val in enumerate(self):
            target[i] = func(val)

        return target



    cpdef _scalar_arithmetic_factory(self, double scalar, str op, bool inplace):
        """
        Helper function to facilitate dunder methods requiring access to _DataPtr object
        which will not work inside of those.
        """
        cdef DataPtr ptr
        if op == 'mul':
            ptr = ops.multiply_by_scalar(self.data_ptr, scalar, inplace)
        elif op == 'add':
            ptr = ops.add_by_scalar(self.data_ptr, scalar, inplace)
        else:
            raise ValueError('Unknown operation: {}'.format(op))

        if inplace:
            self.is_owner = False
        _series = LumberJackSeries.from_ptr(ptr)
        return _series

    def __mul__(self, other):
        return self._scalar_arithmetic_factory(float(other), 'mul', False)

    def __imul__(self, other):
        self = self._scalar_arithmetic_factory(float(other), 'mul', True)
        return self

    def __add__(self, other):
        return self._scalar_arithmetic_factory(float(other), 'add', False)

    def __iadd__(self, other):
        self = self._scalar_arithmetic_factory(float(other), 'add', True)
        return self

    @staticmethod
    def arange(int start, int stop, type dtype=int):
        """
        This is ~2x faster than numpy's arange (tested 100000 times with range 0-100000)
        """
        cdef DType _dtype
        if dtype == float:
            _dtype = DType.Float64
        elif dtype == int:
            _dtype = DType.Int32
        else:
            raise ValueError('dtype "{}" not supported, please submit an issue on github!'.format(dtype))

        cdef DataPtr ptr = ops.arange(start, stop, _dtype)
        series = LumberJackSeries.from_ptr(ptr)
        return series

    def mean(self):
        cdef double avg
        avg = ops.mean(self.data_ptr)
        return avg

    def sum(self):
        cdef double result = ops.sum(self.data_ptr)
        return result

    def cumsum(self):
        cdef DataPtr ptr = ops.cumsum(self.data_ptr)
        series = LumberJackSeries.from_ptr(ptr)
        return series

    def to_cython_array_view(self):
        """
        Provide a cython array view to the data
        """
        return self.array_view

    def to_numpy(self):
        """
        Convert this to numpy array
        """
        cdef np.ndarray array = np.asarray(self.array_view)
        return array

    def __len__(self):
        return self.len

    def __iter__(self):
        return (self.array_view[i] for i in range(self.len))

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

        cdef DataPtr ptr = from_numpy_ptr(&arr_view[0], array.shape[0])
        series = LumberJackSeries.from_ptr(ptr)
        return series

    def __delitem__(self, key):
        raise NotImplementedError('Unable to delete individual elements right now!')

    def __getitem__(self, idx):
        return self.array_view[idx]

    def __setitem__(self, idx, value):
        #self.array_view[idx] = value
        ops.set_item(self.data_ptr, np.uint32(idx), float(value))

    def __repr__(self):
        return 'LumberJackSeries(length: {})'.format(self.len)


