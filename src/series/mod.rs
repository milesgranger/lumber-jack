use std::vec::Vec;
use std::option::Option;


pub struct Series {
    index: Vec<usize>,
    values: Vec<usize>
}

impl Series {

    pub fn new(index: Vec<usize>, values: Vec<usize>) -> Option<Series>  {
        /*
        Return new Series object, but only if index and values are the same length
        return None if failure due to index and values not being same length.
        */
        if index.len() != values.len() { None }
        else { Some(Series {index, values}) }
    }

    pub fn length(&self) -> usize {
        /*
        Return the length of the Series
        */
        self.values.len()
    }

}