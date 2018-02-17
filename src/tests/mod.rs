use self::super::prelude::*;


#[test]
fn series_construction_integer() {
    let _series = Series::new_with_index(vec![0, 1, 2], vec![1, 2, 3]);
    println!("Got series: {:?}", _series);
}

#[test]
fn series_construction_string() {
    let _series: Series<usize, String> = Series::new_without_index(vec!["Hi".to_string(), "Hello".to_string()]);
    println!("Got series: {:?}", _series);
}

