#![feature(use_extern_macros, specialization, pro_macro)]

extern crate ndarray;
extern crate num;
extern crate libc;
extern crate pyo3;

#[macro_use]
pub mod macros;

#[cfg(test)]
mod tests;

pub mod series;
pub mod dataframe;
pub mod prelude;
pub mod alterations;
pub mod containers;

pub use macros::*;
pub use containers::*;
pub use series::*;

