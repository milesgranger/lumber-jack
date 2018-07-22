use self::super::prelude::*;
use self::super::{series};
use self::super::containers::{Data, into_data_ptr, from_data_ptr, DataPtr, DType};

#[test]
fn test_scalar_multiply_i32vec_i32scalar_not_inplace() {
    // Test multiplication by creating a copy
    let data = Data::Int32(vec![1, 2, 3]);
    let new_data = operate_on_vec_by_scalar!(!inplace &data, 2);
    assert_eq!(&Data::Int32(vec![1, 2, 3]), &data);
    assert_eq!(&Data::Int32(vec![2, 4, 6]), &new_data);
    println!("Original: {:?}, New: {:?}", data, new_data);
}

#[test]
fn test_scalar_multiply_i32vec_i32scalar_inplace() {
    // Test multiplication by inplace, all i32
    let mut data = Data::Int32(vec![1_i32, 2_i32, 3_i32]);
    data = operate_on_vec_by_scalar!(inplace data, 2_i32);
    assert_eq!(&Data::Int32(vec![2, 4, 6]), &data);
}

#[test]
fn test_scalar_multiply_i32vec_f64scalar_inplace() {
    // Test multiplication by inplace, vec of i32 * f64
    let mut data = Data::Int32(vec![1_i32, 2_i32, 3_i32]);
    data = operate_on_vec_by_scalar!(inplace data, 2_f64);
    assert_eq!(&Data::Float64(vec![2., 4., 6.]), &data);
}

#[test]
fn test_scalar_multiply_i32vec_f64scalar_not_inplace() {
    // Test multiplication by creating a copy
    let data = Data::Int32(vec![1, 2, 3]);
    let orig = data.clone();
    let new_data = operate_on_vec_by_scalar!(inplace data, 2_f64);
    assert_eq!(&Data::Int32(vec![1, 2, 3]), &orig);
    assert_eq!(&Data::Float64(vec![2., 4., 6.]), &new_data);
    println!("Original: {:?}, New: {:?}", orig, new_data);
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
    let ptr = into_data_ptr(Data::Int32(vec));
    let result = series::cumsum(ptr);
    let vec = match result {
        DataPtr::Int32 { data_ptr, len } => {
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
    let ptr = into_data_ptr(Data::Int32(vec));
    let result = series::sum(ptr);
    if let DataPtr::Int32 { data_ptr, len } = result {
        let v = unsafe { Vec::from_raw_parts(data_ptr, len, len)};
        assert_eq!(v.last().expect("Vector was empty!"), &10_i32)
    } else {
        panic!("Expected to get a DataPtr::Int32 but we did not!");
    };

}

#[test]
fn test_mean() {
    let vec = vec![1, 1, 1, 1, 1];
    let ptr = into_data_ptr(Data::Int32(vec));
    let result = series::mean(ptr);
    assert_eq!(result, 1_f64);
}

#[test]
fn test_arange() {
    let v = series::arange(0, 5, DType::Int32);
    let data = from_data_ptr(v);
    if let Data::Int32(vec) = data {
        assert_eq!(vec.iter().sum::<i32>(), 10);
    } else {
        panic!("Expected Data::Int32 but got {:?} instead!", data);
    }
}