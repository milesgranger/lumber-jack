#![allow(dead_code)]

extern crate ndarray;

use std::vec::Vec;
use std::result::Result;
use std::fmt;


#[derive(Clone, PartialEq, Debug)]
pub enum Value {
    String(String),
    Float64(f64),
    Float32(f32),
    U64(u64),
}

impl Value {
    pub fn new<T>(val: T) -> Value where T: ToString {
        let v = val.to_string().parse::<f64>().unwrap_or(0.0);  // TODO: Replace 0.0 with NaN type
        Value::Float64(v)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}


#[derive(Clone, Debug, PartialEq)]
pub struct Array(Vec<Value>);

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.len())
    }
}

impl Array {

    fn new<T>(vec: Vec<T>) -> Array where T: ToString {
        let mut array: Vec<Value> = Vec::with_capacity(vec.len());
        for v in vec {
            array.push(Value::new(v));
        }
        Array(array)
    }

    fn with_capacity(capacity: usize) -> Array {
        let array: Vec<Value> = Vec::with_capacity(capacity);
        Array(array)
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn push<T>(&mut self, value: T) where T: ToString {
        self.0.push(Value::new(value))
    }

}


/// Create a new Series object
///
/// # Example
///
/// ```
/// use lumberjack::prelude::*;
/// let series = Series::new(None, vec![1, 2, 3]);
/// assert_eq!(series.length(), 4);
/// ```
#[derive(Debug, PartialEq, Clone)]
pub struct Series
{
    index: Array,
    values: Array,
}


impl Series {

    pub fn new<T1, T2>(index: Option<Vec<T1>>, values: Vec<T2>) -> Series
        where T1: ToString, T2: ToString
    {
        /*
        Return new Series object, but only if index and values are the same length
        return None if failure due to index and values not being same length.
        */

        // Deal with the possibility an index vector wasn't passed.
        let mut _index = Array::with_capacity(values.len());
        if let Some(index) = index {

            // We don't know what dtype the values of index is
            for v in index {
                _index.push(Value::new(v));
            }
        } else {

            // No index passed, we know these will be u64 dtypes
            for i in 0..values.len() {
                _index.push(Value::new(i as u64));
            }
        }

        // Place unknown values into the vector for series values
        let mut _values = Array::with_capacity(values.len());
        for val in values {
            _values.push(Value::new(val));
        }

        Series {
            index: _index,
            values: _values
        }
    }
}


pub trait Shape {
    fn shape(&self) -> (u32,);
    fn length(&self) -> u32;
}

impl Shape for Series {

    fn shape(&self) -> (u32,) {
        (self.values.len() as u32,)
    }

    fn length(&self) -> u32 {
        self.values.len() as u32
    }
}


pub trait CastAsFloat64 {
    fn as_float64(&mut self, coerce: bool) -> Result<bool, String>;
}

impl CastAsFloat64 for Series {

    fn as_float64(&mut self, coerce: bool) -> Result<bool, String> {
        if coerce {
            Ok(true)
        } else {
            Ok(true)
        }

    }
}


