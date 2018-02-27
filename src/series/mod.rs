#![allow(dead_code)]

use super::ndarray::prelude::*;
use num;
use std::fmt;
use std::iter::{IntoIterator};
use std::ops::Mul;
use std::marker::Sized;
use ndarray::{OwnedRepr};
use ndarray::ArrayBase;


#[derive(Debug)]
pub enum DType {
    Float64,
    String,
    Int64,
    USize
}

/// Trait to define supported dtypes.
pub trait LumberJackData {
    fn kind(&self) -> DType;
}

/// Support the f64 dtype
impl LumberJackData for usize {
    fn kind(&self) -> DType {
        DType::USize
    }
}

/// Implement Debug for the LumberJackData trait
impl fmt::Debug for LumberJackData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LumberJackData[{:?}]", self.kind())
    }
}

type LumberJackArray = ArrayBase<OwnedRepr<Box<LumberJackData>>, Dim<[usize; 1]>>;

/// Series struct, with four type parameters where T1 & D1 represent the data type and
/// data dimension respectively for the index array, and T2 & D2 for the series values
#[derive(Debug)]
pub struct Series
{
    index: LumberJackArray,
    values: LumberJackArray
}


impl Series {

    pub fn new<I>(index: I, values: I) -> Self
        where I: IntoIterator, I::Item: LumberJackData
    {
        let index = Array::from_iter(index.into_iter());
        let values = Array::from_iter(values.into_iter());
        Series{index, values}
    }


    pub fn len(&self) -> usize {
        self.values.len()
    }
}






