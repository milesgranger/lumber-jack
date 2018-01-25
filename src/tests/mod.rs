use self::super::series::Series;
use self::super::prelude::*;


#[test]
fn series_construction_integer() {
    let _series = Series::new(vec![0, 1, 2], vec![1, 2, 3]);
    println!("Got series: {:?}", _series);
}

#[test]
fn series_length_n_shape_integer() {
    let _series = Series::new(vec![0, 1, 2], vec![1, 2, 3]);
    assert_eq!(3, _series.length());
    assert_eq!((3,), _series.shape());
}

#[test]
fn series_construction_str() {
    let _series = Series::new(vec![0, 1], vec!["Hi".to_string(), "Hello".to_string()]);
    println!("Got series: {:?}", _series);
}

#[test]
fn series_length_n_shape_str() {
    let _series = Series::new(vec![0, 1], vec!["Hi".to_string(), "Hello".to_string()]);
    assert_eq!(2, _series.length());
    assert_eq!((2,), _series.shape());
}

