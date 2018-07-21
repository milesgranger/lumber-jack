use self::super::prelude::*;
use self::super::{series, containers};

#[macro_use]
use macros;

#[test]
fn test_scalar_multiply() {
    // Test multiplication by creating a copy
    let data = vec![1, 2, 3];
    let new_data = multiply_vec_by_scalar!(!inplace &data, 2);
    assert_eq!(&vec![1, 2, 3], &data);
    assert_eq!(&vec![2, 4, 6], &new_data);
    println!("Original: {:?}, New: {:?}", data, new_data);

    // Test multiplication by inplace
    let mut data = vec![1, 2, 3];
    multiply_vec_by_scalar!(inplace &mut data, 2);
    assert_eq!(vec![2, 4, 6], data)
}

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

#[test]
fn test_sum() {
    let vec = vec![1, 2, 3, 4];
    let ptr = containers::into_data_ptr(containers::Data::Int32(vec));
    let result = series::sum(ptr);
    if let containers::DataPtr::Int32 { data_ptr, len } = result {
        let v = unsafe { Vec::from_raw_parts(data_ptr, len, len)};
        assert_eq!(v.last().expect("Vector was empty!"), &10_i32)
    } else {
        panic!("Expected to get a DataPtr::Int32 but we did not!");
    };

}

#[test]
fn test_mean() {
    let vec = vec![1, 1, 1, 1, 1];
    let ptr = containers::into_data_ptr(containers::Data::Int32(vec));
    let result = series::mean(ptr);
    assert_eq!(result, 1_f64);
}

#[test]
fn test_arange() {
    let v = series::arange(0, 5, containers::DType::Int32);
    let data = containers::from_data_ptr(v);
    if let containers::Data::Int32(vec) = data {
        assert_eq!(vec.iter().sum::<i32>(), 10);
    } else {
        panic!("Expected Data::Int32 but got {:?} instead!", data);
    }
}