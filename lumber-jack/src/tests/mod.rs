
#[cfg(test)]
mod tests {

    use self::super::series::Series;
    use std::option::Option;

    #[test]
    fn series_contruction() {
        // Test construction of a series via new()
        let _series: Option<Series> = Series::new(vec![1, 2, 3], vec![1, 2, 3]);
        match _series {
            Some(s) => { /* Successfully made series, test passes. */ },
            _ => { panic!("Unable to create new series!"); }
        }
    }

    #[test]
    fn series_length() {
        // Test the length method of series
        let _series: Option<Series> = Series::new(vec![1, 2, 3], vec![1, 2, 3]);
        if let Some(_series) = _series {
            assert_eq!(3, _series.length());
        } else {
            panic!("Unable to create new series!");
        }
    }

}