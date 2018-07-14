#![allow(dead_code)]

use std::mem;
use std::iter::FromIterator;

pub mod series_funcs;


/// This enum is what Cython will use to read the data created from Rust
#[repr(C)]
pub enum DataPtr {
    Float64 {
        data_ptr: *mut f64,
        len: usize
    },
    Int32 {
        data_ptr: *mut i32,
        len: usize
    }
}


/// Container for various supported data types
#[derive(Debug)]
pub enum Data {
    Float64(Vec<f64>),
    Int32(Vec<i32>)
}


/// Define which data types can be requested or cast to.
/// to serve as flags between Cython and Rust for data type conversions / creations
#[repr(C)]
#[derive(Debug)]
pub enum DType {
    Float64,
    Int32
}

impl FromIterator<i32> for Data {
    fn from_iter<I: IntoIterator<Item=i32>>(iter: I) -> Data {
        let mut vec = Vec::new();
        for v in iter {
            vec.push(v)
        }
        Data::Int32(vec)
    }
}

impl FromIterator<f64> for Data {
    fn from_iter<I: IntoIterator<Item=f64>>(iter: I) -> Data {
        let mut vec = Vec::new();
        for v in iter {
            vec.push(v)
        }
        Data::Float64(vec)
    }
}

/// Core data structure of LumberJack
#[derive(Debug)]
pub struct Series
{
    data: Data
}

/// Return a vector from a pointer
pub unsafe fn vec_from_raw<T>(ptr: *mut T, n_elements: usize) -> Vec<T> {
    Vec::from_raw_parts(ptr, n_elements, n_elements)
}


impl Series
{
    /// Create Series from a range of i32 values cast as DType
    pub fn from_arange(start: i32, stop: i32, dtype: DType) -> Self
    {
        let data = match dtype {
            DType::Float64 => (start..stop).map(|v| v as f64).collect(),
            DType::Int32 => (start..stop).map(|v| v as i32).collect()
        };

        Self { data }
    }

    /// Return a series from DataPtr
    pub fn from_data_ptr(ptr: DataPtr) -> Self {
        match ptr {
            DataPtr::Float64 { data_ptr, len } => {
                Self {
                    data: Data::Float64(unsafe { vec_from_raw(data_ptr, len)})
                }
            },
            DataPtr::Int32 { data_ptr, len } => {
                Self {
                    data: Data::Int32(unsafe { vec_from_raw(data_ptr, len)})
                }
            }
        }
    }

    /// Build a Series object from the DataPtr enum
    pub fn into_data_ptr(self) -> DataPtr {

        // Create a pointer which has the raw vector pointer and does not let it fall out of
        // scope by forgetting it, as it will be used later, and 'self' will be dropped.
        // TODO: Consider boxing self and returning that along with the DataPtr instead
        let data_ptr = match self.data {

            Data::Float64(mut vec) => {
                vec.shrink_to_fit();
                let ptr = DataPtr::Float64 {
                    data_ptr: vec.as_mut_ptr(),
                    len: vec.len()
                };
                mem::forget(vec);
                ptr
            },

            Data::Int32(mut vec) => {
                vec.shrink_to_fit();
                let ptr = DataPtr::Int32 {
                    data_ptr: vec.as_mut_ptr(),
                    len: vec.len()
                };
                mem::forget(vec);
                ptr
            }
        };

        data_ptr
    }
}


/*
    Function to be exposed to C below here and to be moved elsewhere later.
*/

/// Create Series from arange and pass back as DataPtr
#[no_mangle]
pub extern "C" fn arange(start: i32, stop: i32, dtype: DType) -> DataPtr {
    Series::from_arange(start, stop, dtype).into_data_ptr()
}

/// Reconstruct Series from DataPtr and let it fall out of scope to clear from memory.
#[no_mangle]
pub extern "C" fn free_series(data_ptr: DataPtr) {
    let _series = Series::from_data_ptr(data_ptr);
    println!("Got series: {:?}, letting it fall out of scope!", &_series);
}
