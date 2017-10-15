#[macro_use(array)]
extern crate ndarray;

mod csv_reader;
mod dataframe;

use dataframe::{DataFrameOps, DataFrame};

fn main(){
    println!("Hello world!");

    let mut df: DataFrame = DataFrame::new();

    println!("No problame-O!");
}