use std::mem;
use super::{LumberJackSeriesPtr, DType, LumberJackData};

#[no_mangle]
pub extern "C" fn arange(start: i32, stop: i32, dtype: DType) -> LumberJackSeriesPtr
{
    match dtype {
        DType::Float64 => LumberJackSeriesPtr::from_vec((start..stop).map(|v| v as f64).collect(), dtype),
        DType::Int32 => LumberJackSeriesPtr::from_vec((start..stop).map(|v| v as i32).collect(), dtype),
    }
}

#[no_mangle]
pub extern "C" fn from_numpy_ptr<T>(array_ptr: *mut T, n_elements: usize) -> LumberJackSeriesPtr
    where T: LumberJackData + 'static
{
    let array = unsafe { create_vec_from_ptr(array_ptr, n_elements) };
    LumberJackSeriesPtr::from_vec(array, DType::Float64)
}

#[no_mangle]
pub extern "C" fn double_array(array_ptr: *mut f64) {
    println!("RUST: Have taken array pointer! {:?}", &array_ptr);
    let mut array = unsafe { create_vec_from_ptr(array_ptr, 10) };
    println!("RUST: Before doubling: {:?}", &array);
    double(&mut array);
    println!("RUST: Array after doubling is: {:?}", &array);
    mem::forget(array);
}

#[no_mangle]
// Create an array from a pointer and then let it fall out of scope to remove from memory.
pub extern "C" fn free_vector(array_ptr: *mut f64, n_elements: usize) {
    let _vector = unsafe { create_vec_from_ptr(array_ptr,  n_elements) };
    //println!("Created array and letting it fall out of scope!")
}

pub unsafe fn create_vec_from_ptr<T>(input: *mut T, n_elements: usize) -> Vec<T>
    where T: LumberJackData + 'static
{
    assert!(!input.is_null());
    let v1 = Vec::from_raw_parts(input, n_elements, n_elements);
    v1
}

fn double(array: &mut Vec<f64>) {
    for v in array.iter_mut() {
        *v *= 2.0;
    }
}