#![feature(proc_macro, specialization)]

extern crate ndarray;
extern crate num;

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