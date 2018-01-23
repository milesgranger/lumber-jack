

use self::super::series::Series;
use self::super::alterations;
use std::option::Option;

#[test]
fn series_construction() {
    let _series = Series::new(vec![0, 1, 2], vec![1, 2, 3]);
    println!("Got series: {:?}", _series);
}

