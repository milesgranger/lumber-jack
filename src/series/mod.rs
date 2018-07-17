#![allow(dead_code)]

use std::mem;

mod operators;
use containers::{DataPtr, DType, Data, into_data_ptr, from_data_ptr};



/*
    Function to be exposed to C below here and to be moved elsewhere later.
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


#[no_mangle]
pub extern "C" fn sum(data_ptr: DataPtr) -> DataPtr {

    let data = from_data_ptr(data_ptr);

    match data {
        Data::Float64(vec) => {
            let mut result = operators::sum_vec(&vec);
            let ptr = into_data_ptr(Data::Float64(result));
            mem::forget(vec);
            ptr
        },
        Data::Int32(vec) => {
            let mut result = operators::sum_vec(&vec);
            let ptr = into_data_ptr(Data::Int32(result));
            mem::forget(vec);
            ptr
        }
    }
}

#[no_mangle]
pub extern "C" fn cumsum(data_ptr: DataPtr) -> DataPtr {
    let data = from_data_ptr(data_ptr);

    match data {
        Data::Float64(vec) => {
            let mut result= Vec::with_capacity(vec.len());
            let mut cumsum = 0_f64;
            for val in vec.iter() {
                cumsum += val;
                result.push(cumsum);
            }
            let ptr = into_data_ptr(Data::Float64(result));
            mem::forget(vec);
            ptr
        },
        Data::Int32(vec) => {
            let mut result = Vec::with_capacity(vec.len());
            let mut cumsum = 0_i32;
            for val in vec.iter() {
                cumsum += val;
                result.push(cumsum);
            }
            let ptr = into_data_ptr(Data::Int32(result));
            mem::forget(vec);
            ptr
        }
    }
}

/// Reconstruct Series from DataPtr and let it fall out of scope to clear from memory.
#[no_mangle]
pub extern "C" fn free_data(data_ptr: DataPtr) {
    // TODO: Replace this with dropping a pointer instead of passing the entire DataPtr struct back
    let _data = from_data_ptr(data_ptr);
    //println!("Got data letting it fall out of scope!");
}
