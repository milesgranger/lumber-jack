use self::super::prelude::*;


#[test]
fn series_construction_integer() {
    let _series = Series::new(vec![0, 1, 2], vec![1, 2, 3]);
    println!("Got series: {:?}", _series);

    assert_eq!(_series.len(), 3);
}


