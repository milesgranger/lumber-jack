//#![feature(proc_macro, specialization)]

extern crate ndarray;
extern crate num;
extern crate libc;

use std::mem;

#[cfg(test)]
mod tests;
pub mod series;
pub mod dataframe;
pub mod prelude;
pub mod alterations;


#[repr(C)]
pub struct LumberJackSeriesPtr {
    ptr: *mut f64,
    len: usize,

}

impl LumberJackSeriesPtr {
    fn from_vec<T>(mut vec: Vec<T>) -> LumberJackSeriesPtr {
        vec.shrink_to_fit();
        let series_ptr = LumberJackSeriesPtr {
            ptr: vec.as_ptr() as *mut f64,
            len: vec.len(),
        };
        mem::forget(vec);
        series_ptr
    }

    #[no_mangle]
    pub extern "C" fn return_two(&self) -> i64 {
        2
    }
}

#[no_mangle]
pub extern "C" fn add_two_in_rust(a: f64, b: f64) -> f64 {
    a + b
}

#[no_mangle]
pub extern "C" fn create_lumberjack_series() -> LumberJackSeriesPtr {
    let vec = vec![1., 2., 3., 4.];
    LumberJackSeriesPtr::from_vec(vec)
}

#[no_mangle]
pub extern "C" fn from_numpy_ptr(array_ptr: *mut f64, n_elements: usize) -> LumberJackSeriesPtr {
    let array = unsafe { create_vec_from_ptr(array_ptr, n_elements) };
    LumberJackSeriesPtr::from_vec(array)
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
    println!("Created array and letting it fall out of scope!")
}

pub unsafe fn create_vec_from_ptr<T>(input: *mut T, n_elements: usize) -> Vec<T> {
    assert!(!input.is_null());
    let v1 = Vec::from_raw_parts(input, n_elements, n_elements);
    v1
}

fn double(array: &mut Vec<f64>) {
    for v in array.iter_mut() {
        *v *= 2.0;
    }
}
