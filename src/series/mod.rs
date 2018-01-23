extern crate num;

use self::num::{Num, NumCast, Float};
use std::vec::Vec;
use std::option::Option;
use std::result::Result;
use std::iter;

#[derive(Debug, PartialEq, Clone)]
pub enum DataType {
    Numeric,
    Object
}


#[derive(Debug, PartialEq, Clone)]
pub struct DataCell<T> {
    value: T,
    data_type: DataType
}

impl<T> DataCell<T> {
    pub fn new(value: T, data_type: DataType) -> DataCell<T> {
        DataCell{
            value,
            data_type
        }
    }
}


#[derive(Debug, PartialEq, Clone)]
pub struct Series<T>
{
    index: Vec<DataCell<T>>,
    values: Vec<DataCell<T>>,
}


impl<T> Series<T> {

    pub fn new(index: Vec<T>, values: Vec<T>) -> Series<T> {
        /*
        Return new Series object, but only if index and values are the same length
        return None if failure due to index and values not being same length.
        */

        let mut _index: Vec<DataCell<T>> = Vec::with_capacity(index.len());
        let mut _values: Vec<DataCell<T>> = Vec::with_capacity(values.len());
        for val in index {
            _index.push(DataCell::new(val, DataType::Object));
        }
        for val in values {
            _values.push(DataCell::new(val, DataType::Object));
        }

        Series {
            index: _index,
            values: _values
        }



    }

    pub fn length(&self) -> usize {
        /*
        Return the length of the Series
        */
        self.values.len()
    }

}