use self::super::prelude::*;


#[test]
fn test_alterations_one_hot() {
    /*
        Test the alterations split_n_hot_encode function
    */
    let words = vec!["hi, hello".to_string(), "hi, bye".to_string()];
    let (unique_words, one_hot) = alterations::split_n_hot_encode(
        words.clone(), ",".to_string(), 0
    );

    // Test that the first vector of unique words found contains "hi"
    println!("Got unique words: {:?}", unique_words);
    assert_eq!(unique_words.contains(&"hi".to_string()), true);

    // Test that the first one-hot array sums to 2, for "hi" and "hello"
    println!("Got one-hot: {:?}", one_hot);
    assert_eq!(one_hot[0].iter().fold(0, |sum, &val| sum + val), 2);
}


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

#[test]
fn series_map_func() {
    let _series = Series::new(vec![0, 1, 2], vec![0, 1, 2]);
    let _new_series = _series.map(|x| x * 2);

    // Assert that mapping the x2 equals 6 -> sum(x * 2 for x in [0, 1, 2])
    println!("Sum is: {}", _new_series.sum());
    assert_eq!(_new_series.sum(), 6);

}

