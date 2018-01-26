use self::super::prelude::*;


#[test]
fn series_construction_integer() {
    let _series = Series::new(Some(vec![0, 1, 2]), vec![1, 2, 3]);
    println!("Got series: {:?}", _series);
}

#[test]
fn series_construction_string() {
    let _series = Series::new(Some(vec![0, 1, 2]), vec!["Hi".to_string(), "Hello".to_string()]);
    println!("Got series: {:?}", _series);
}

