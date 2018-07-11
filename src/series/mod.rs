#![allow(dead_code)]

use super::ndarray::prelude::*;
use std::{fmt, mem};
use std::iter::{IntoIterator};
use num::Zero;
use ndarray::{OwnedRepr, ArrayBase};

pub mod series_funcs;

#[repr(C)]
pub struct LumberJackSeriesPtr {
    data_ptr: *mut LumberJackData,
    len: usize,
}

fn ptr_from_vec<T: LumberJackData>(mut vec: Vec<T>) -> *mut T {
    vec.shrink_to_fit();
    let ptr = vec.as_mut_ptr();
    mem::forget(vec);
    ptr
}

impl LumberJackSeriesPtr
{

    fn from_vec<T>(mut vec: Vec<T>) -> Self
        where T: LumberJackData + 'static
    {
        vec.shrink_to_fit();
        let series_ptr = LumberJackSeriesPtr {
            data_ptr: vec.as_mut_ptr() as *mut T,
            len: vec.len(),
        };
        mem::forget(vec);
        series_ptr
    }
}

pub enum LJData {
    Float64 {
        data: Vec<f64>,
        dtype: DType
    },
    Int32 {
        data: Vec<i32>,
        dtype: DType
    }
}

#[repr(C)]
#[derive(Debug)]
pub enum DType {
    Float64,
    Int32
}

/// Trait to define supported dtypes.
pub trait LumberJackData {
    fn kind(&self) -> DType;
}

/// Support the usize dtype
impl LumberJackData for f64 {
    fn kind(&self) -> DType {
        DType::Float64
    }
}

/// Support the usize dtype
impl LumberJackData for i32 {
    fn kind(&self) -> DType {
        DType::Int32
    }
}

/// Implement Debug for the LumberJackData trait
impl fmt::Debug for LumberJackData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LumberJackData[{:?}]", self.kind())
    }
}


/// Series struct, with four type parameters where T1 & D1 represent the data type and
/// data dimension respectively for the index array, and T2 & D2 for the series values
#[derive(Debug)]
pub struct Series<T>
    where T: LumberJackData
{
    index: ArrayBase<OwnedRepr<T>, Dim<[usize; 1]>>,
    values: ArrayBase<OwnedRepr<T>, Dim<[usize; 1]>>,
    dtype: DType
}


impl<T> Series<T>
    where T: LumberJackData + Clone + Zero
{

    pub fn new<I>(index: I, values: I) -> Self
        where I: IntoIterator<Item=T>
    {
        let index = Array::from_iter(index.into_iter());
        let values = Array::from_iter(values.into_iter());
        let val = values[0].clone();
        let dtype = val.kind();
        Series{index, values, dtype}
    }


    pub fn map<F>(&self, func: F) -> Self
        where F: Fn(&T) -> T
    {
        /*
            Map an arbitrary function over the values of a series and return a new series result.
        */
        let mut values = Vec::new();
        for val in self.values.iter() {
            let result = func(val);
            values.push(result);
        }
        Series::new(self.index.to_vec(), values)
    }


    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn sum(&self) -> T {
        self.values.scalar_sum()
    }

    pub fn append<I: IntoIterator<Item=T>>(&mut self, index: I, values: I, inplace: bool) -> Option<Self>
        where T: Clone
    {
        // Append an iterable to Self or return a copy
        let mut new_index = self.index.to_vec();
        let mut new_values = self.values.to_vec();
        for (idx, value) in index.into_iter().zip(values.into_iter()) {
            new_values.push(value);
            new_index.push(idx);
        }

        if inplace {
            self.index = Array::from_vec(new_index);
            self.values = Array::from_vec(new_values);
            None
        } else {
            Some(Series::new(new_index, new_values))
        }
    }
}






