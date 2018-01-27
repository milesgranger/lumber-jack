#![allow(dead_code)]

use super::ndarray::prelude::*;
use super::ndarray::{Data, OwnedRepr};

/// Series struct, with four type parameters where T1 & D1 represent the data type and
/// data dimension respectively for the index array, and T2 & D2 for the series values
pub struct Series<T1, T2>
    where T1: Data, T2: Data
{
    index: ArrayBase<OwnedRepr<T1>, Dim<[usize; 1]>>,
    values: ArrayBase<OwnedRepr<T2>, Dim<[usize; 1]>>
}

/// Series implementation where the values of the vector must have the Clone trait.
/// this is in order to be able to return copies of the underlying values within the series.
impl<T1, T2> Series<T1, T2>
    where T1: Data, T2: Data
{
    /// Create a new series from two vectors
    pub fn new(index: Vec<T1>, values: Vec<T2>) -> Self {
        let index = Array::from_vec(index);
        let values = Array::from_vec(values);

        Series {index, values}
    }

}


