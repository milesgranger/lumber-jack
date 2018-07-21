#[macro_export]
macro_rules! multiply_vec_by_scalar {

    // Multiply a vector inplace
    // ie: multiply_vec_by_scalar(vec, 2.0);
    (inplace $vec:expr, $scalar:expr) => {

        {
            use ndarray::ArrayViewMut;

            let mut array = ArrayViewMut::from_shape(($vec.len(),), $vec)
                .expect("Failed to build array view!");

            array *= $scalar;
            array.to_vec()
        }
    };

    // Multiply a vector by creating a copy
    // ie: multiply_vec_by_scalar(ref &vec, 2.0);
    (!inplace $vec:expr, $scalar:expr) => {
        {
            use std::mem;
            use ndarray::Array;

            let v = $vec.clone();
            mem::forget($vec);
            let mut array = Array::from_vec(v);
            array *= $scalar;
            array.to_vec()
        }
    }
}