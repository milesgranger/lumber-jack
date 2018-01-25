#![allow(dead_code)]

use std::vec::Vec;

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
pub struct Series<T1, T2>
{
    index: Vec<DataCell<T1>>,
    values: Vec<DataCell<T2>>,
}

pub trait Shape {
    fn shape(&self) -> (u32,);
}

impl<T1, T2> Shape for Series<T1, T2> {
    fn shape(&self) -> (u32,) {
        (self.values.len() as u32,)
    }
}

impl<T1, T2> Series<T1, T2> {

    pub fn new(index: Vec<T1>, values: Vec<T2>) -> Series<T1, T2> {
        /*
        Return new Series object, but only if index and values are the same length
        return None if failure due to index and values not being same length.
        */

        let mut _index: Vec<DataCell<T1>> = Vec::with_capacity(index.len());
        let mut _values: Vec<DataCell<T2>> = Vec::with_capacity(values.len());
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
