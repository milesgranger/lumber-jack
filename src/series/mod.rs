#![allow(dead_code)]

mod operators;
use containers::{DataPtr, DType, Data, into_data_ptr, from_data_ptr};
pub use series::operators::*;

/*
    Functions exposed to C which create lumberjack series or deal with existing ones
*/

/// Create Series from arange and pass back as DataPtr
#[no_mangle]
pub extern "C" fn arange(start: i32, stop: i32, dtype: DType) -> DataPtr {
    let ptr = match dtype {
            DType::Float64 => {
                let mut data = (start..stop).map(|v| v as f64).collect::<Vec<f64>>();
                let ptr = into_data_ptr(Data::Float64(data));
                ptr
            }

            DType::Int32 => {
                let mut data = (start..stop).map(|v| v as i32).collect::<Vec<i32>>();
                let ptr = into_data_ptr(Data::Int32(data));
                ptr
            }
        };
    ptr
}

/// Reconstruct Series from DataPtr and let it fall out of scope to clear from memory.
#[no_mangle]
pub extern "C" fn free_data(data_ptr: DataPtr) {
    // TODO: Replace this with dropping a pointer instead of passing the entire DataPtr struct back
    let _data = from_data_ptr(data_ptr);
    //println!("Got data letting it fall out of scope!");
}
