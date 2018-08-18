use containers::Data;
use std::ops::{Index, IndexMut};


struct Series {
    name: String,
    data: Data
}

pub struct DataFrame {
    data: Vec<Series>
}

impl Index<String> for DataFrame {
    type Output = Data;
    fn index<'a>(&'a self, index: String) -> &'a Data {
        for series in self.data.iter() {
            if series.name == index {
                return &series.data
            }
        }
        panic!("Not found");
    }
}

impl IndexMut<String> for DataFrame {
    fn index_mut<'a>(&'a mut self, index: String) -> &'a mut Data {

        // Data enum type doesn't matter as it's going to be overwritten
        // on assignment.
        let new_series = Series{name: index, data: Data::Float64(Vec::new())};
        self.data.push(new_series);
        let idx = self.data.len() - 1 as usize;
        &mut self.data[idx].data
    }
}

impl DataFrame {
    pub fn new() -> Self {
        let data = Vec::new();
        DataFrame {data}
    }
}


