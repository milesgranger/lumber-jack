use dataframe::DataFrame;
use containers::Data;

#[test]
fn test_create_dataframe() {
    let df = DataFrame::new();
}

#[test]
fn test_assign_column() {
    let data   = Data::Int32(vec![1, 2, 3]);
    let mut df = DataFrame::new();
    df["col1".to_string()] = data.clone();
    assert_eq!(&df["col1".to_string()], &data);
}