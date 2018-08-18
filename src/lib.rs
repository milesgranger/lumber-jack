
//! LumberJack is a crate is a lightweight alternative to Python's
//! "Pandas" package.
//! 
//! Its main intention is to have a Python wrapper, but feel free
//! to make use of it in other settings!
//! 
//! # Examples
//! 
//! ```
//! use lumberjack::prelude::*;
//! 
//! let mut df = DataFrame::new():
//! let series = Series::arange(0, 10);
//! df["column1"] = series;
//! 
//! df.sum();
//! ```

extern crate ndarray;
extern crate num;
extern crate libc;

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
