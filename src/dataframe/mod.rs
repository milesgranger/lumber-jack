use containers::Data;
use std::ops::{Index, IndexMut};


struct Series<'a> {
    name: &'a str,
    data: Data
}

pub struct DataFrame<'a> {
    data: Vec<Series<'a>>
}

impl<'a> Index<&'a str> for DataFrame<'a> {
    type Output = Data;
    fn index<'b>(&'b self, index: &'a str) -> &'b Data {
        for series in self.data.iter() {
            if series.name == index {
                return &series.data
            }
        }
        panic!("Not found");
    }
}

impl<'a> IndexMut<&'a str> for DataFrame<'a> {
    fn index_mut<'b>(&'b mut self, index: &'a str) -> &'b mut Data {

        // Data enum type doesn't matter as it's going to be overwritten
        // on assignment.
        let new_series = Series{name: index, data: Data::Float64(Vec::new())};
        self.data.push(new_series);
        let idx = self.data.len() - 1 as usize;
        &mut self.data[idx].data
    }
}

impl<'a> DataFrame<'a> {
    pub fn new() -> Self {
        let data = Vec::new();
        DataFrame { data }
    }
}


