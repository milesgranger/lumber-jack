use self::super::prelude::*;


#[test]
fn series_construction_integer() {
    let _series = Series::new(vec![0, 1, 2], vec![1, 2, 3]);
    println!("Got series: {:?}", _series);

    // Length check
    assert_eq!(_series.len(), 3);

    // multiplication check
}


#[test]
fn series_append_value() {
    let mut _series = Series::new(vec![0, 1, 2], vec![1, 2, 3]);
    println!("Got series: {:?}", _series);

    // append a value asking for a copy of the series
    _series = _series.append( vec![3, 4], vec![5, 6], false).unwrap();

    // Length check
    assert_eq!(_series.len(), 5);
}

#[test]
fn series_append_value_inplace() {
    let mut _series = Series::new(vec![0, 1, 2], vec![1, 2, 3]);
    println!("Got series: {:?}", _series);

    // append a value inplace
    _series.append( vec![3, 4], vec![5, 6], true);

    // Length check
    assert_eq!(_series.len(), 5);
}

