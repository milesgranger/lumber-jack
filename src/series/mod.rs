#![allow(dead_code)]

use super::ndarray::prelude::*;
use std::fmt;
use std::iter::FromIterator;

use ndarray::OwnedRepr;


#[derive(Debug)]
pub enum DType {
    Float64,
    String,
    Int64
}

/// Trait to define supported dtypes.
pub trait LumberJackData {
    fn dtype(&self) -> DType;
}


/// Support the f64 dtype
impl LumberJackData for f64 {
    fn dtype(&self) -> DType {
        DType::Float64
    }
}

impl LumberJackData for i64 {
    fn dtype(&self) -> DType {
        DType::Int64
    }
}

/// Implement Debug for the LumberJackData trait
impl fmt::Debug for LumberJackData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LumberJackArray[{:?}]", self.dtype())
    }
}

/// Series struct, with four type parameters where T1 & D1 represent the data type and
/// data dimension respectively for the index array, and T2 & D2 for the series values
#[derive(Debug)]
pub struct Series<T: LumberJackData>
{
    index: ArrayBase<OwnedRepr<T>, Dim<[usize; 1]>>,
    values: ArrayBase<OwnedRepr<T>, Dim<[usize; 1]>>
}


impl<T> Series<T> where T: LumberJackData {

    pub fn new(index: Vec<T>, values: Vec<T>) -> Self
    {
        let index = Array::from_vec(index);
        let values = Array::from_vec(values);
        Series{index, values}
    }


    pub fn len(&self) -> usize {
        self.values.len()
    }
}





