use std::any::Any;
use ndarray::{Array, ArrayBase, Array1};
use csv_reader::read_csv;

pub struct DataFrame {
    /*
        DataFrame

        TODO: Figure out how to make arrays from unknown types...
    */
    index: Array1<u64>,
    columns: Array1<&'static str>
}

pub trait DataFrameOps {
    /*
        Methods for DataFrame
    */

    #[allow(dead_code)]
    fn new() -> DataFrame {
        DataFrame {
            index: array![0, 1],
            columns: array!["col1", "col2"]
        }
    }

    #[allow(dead_code)]
    fn from_csv(file_path: &str) -> DataFrame {
        DataFrame {
            index: array![0, 1],
            columns: array!["col1", "col2"]
        }
    }

    #[allow(dead_code)]
    fn length(&self) -> u64 {
        65
    }

}

impl DataFrameOps for DataFrame {}