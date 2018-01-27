#![allow(dead_code)]

use super::ndarray::prelude::*;
use super::ndarray::{OwnedRepr};
use super::num;

/// Series struct, with four type parameters where T1 & D1 represent the data type and
/// data dimension respectively for the index array, and T2 & D2 for the series values
pub struct Series<T1, T2>
{
    index: ArrayBase<OwnedRepr<T1>, Dim<[usize; 1]>>,
    values: ArrayBase<OwnedRepr<T2>, Dim<[usize; 1]>>
}


/// Series implementation where the values of the vector must have the Clone trait.
/// this is in order to be able to return copies of the underlying values within the series.
impl<T1, T2> Series<T1, T2>
    where T1: num::Float + num::FromPrimitive
{
    /// Create a new series where index is optional. In such case 0..n is defined as the index.
    pub fn new(index: Option<Vec<T1>>, values: Vec<T2>) -> Self {
        match index {
            Some(idx) => Series::new_with_index(idx, values),
            None => Series::new_without_index(values)
        }
    }

    // Create the series when an index wasn't passed, just a 1 step range to the length of values
    fn new_without_index(values: Vec<T2>) -> Self {
        use num::cast::FromPrimitive;
        let index = Array::range(
            FromPrimitive::from_usize(0 as usize).unwrap(),
            FromPrimitive::from_usize(values.len()).unwrap(),
            FromPrimitive::from_usize(1 as usize).unwrap()
        );
        let values = Array::from_vec(values);
        Series {index, values}
    }

    // Create the series given that there was an index passed.
    fn new_with_index(index: Vec<T1>, values: Vec<T2>) -> Self {

        let index = Array::from_vec(index);
        let values = Array::from_vec(values);
        Series {index, values}
    }

}



