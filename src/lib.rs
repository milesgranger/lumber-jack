//#![feature(proc_macro, specialization)]

extern crate ndarray;
extern crate num;
extern crate libc;

#[cfg(test)]
mod tests;
pub mod series;
pub mod dataframe;
pub mod prelude;
pub mod alterations;


pub use series::series_funcs::*;
pub use series::*;