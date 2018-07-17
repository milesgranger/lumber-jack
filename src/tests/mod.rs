use std::iter::Sum;
use self::super::prelude::*;
use self::super::{series, containers};


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
fn test_cumsum() {
    let vec = vec![0, 1, 2, 3, 4];
    let ptr = containers::into_data_ptr(containers::Data::Int32(vec));
    let result = series::cumsum(ptr);
    let vec = match result {
        containers::DataPtr::Int32 { data_ptr, len } => {
            unsafe { Vec::from_raw_parts(data_ptr, len, len)}
        },
        _ => panic!("Expected to get DataPtr::Int32!")
        };
    println!("Got vec: {:?}", &vec);
    assert_eq!(vec.last().expect("Vector was empty!"), &10_i32);
}