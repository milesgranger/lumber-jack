#![allow(dead_code)]

use super::ndarray::prelude::*;
use super::ndarray::{OwnedRepr, Data};
use super::num;

#[derive(Debug, Clone)]
pub enum DataType {
    Usize(usize),
    Float64(f64),
    Int64(i64),
    String(String)
}


/// Series struct, with four type parameters where T1 & D1 represent the data type and
/// data dimension respectively for the index array, and T2 & D2 for the series values
#[derive(Debug)]
pub struct Series<I, V>
{
    index: ArrayBase<OwnedRepr<I>, Dim<[usize; 1]>>,
    values: ArrayBase<OwnedRepr<V>, Dim<[usize; 1]>>
}


impl<I, V> Series<I, V> {

    pub fn new(index: Vec<I>, values: Vec<V>) -> Self
    {
        let index = Array::from_vec(index);
        let values = Array::from_vec(values);
        Series{index, values}
    }
}





