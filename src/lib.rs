#![feature(proc_macro, specialization)]

extern crate ndarray;
extern crate num;

use std::mem;

#[cfg(test)]
mod tests;
pub mod series;
pub mod dataframe;
pub mod prelude;
pub mod alterations;

#[no_mangle]
pub extern "C" fn add_two_in_rust(a: f64, b: f64) -> f64 {
    a + b
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


#[no_mangle]
pub extern "C" fn double_array(array_ptr: *mut f64) {
    println!("RUST: Have taken array pointer! {:?}", &array_ptr);
    let mut array = unsafe { create_vec_from_ptr(array_ptr, 10) };
    println!("RUST: Before doubling: {:?}", &array);
    double(&mut array);
    println!("RUST: Array after doubling is: {:?}", &array);
    mem::forget(array);

}