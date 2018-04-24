#![feature(proc_macro, specialization)]

extern crate ndarray;
extern crate num;
extern crate pyo3;

use pyo3::prelude::*;

#[cfg(test)]
mod tests;
pub mod series;
pub mod dataframe;
pub mod prelude;
pub mod alterations;


#[py::modinit(alterations)]
// Initialize the 'alterations' module of lumberjack
fn init_mod(py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "split_n_one_hot_encode")]
    fn split_n_one_hot_encode_py(raw_texts: Vec<String>, sep: String, cutoff: usize) -> PyResult<(Vec<String>,  Vec<Vec<u8>>)> {
        let (words, one_hot) = alterations::split_n_hot_encode(raw_texts, sep, cutoff);
        Ok((words, one_hot))
    }

    Ok(())
}


// TODO: build and initialize a dataframe like module