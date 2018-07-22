#[macro_export]
macro_rules! multiply_vec_by_scalar {

    // Multiply a vector inplace
    // ie: multiply_vec_by_scalar(vec, 2.0);
    (inplace $data:expr, $scalar:expr) => {

        {
            use containers::{Data};
            use std::any::Any;

            // Get vec which can be any primitive
            // ...and perform an inplace op on it by scalar which can be any primitive.

            // If scalar is f64...
            if let Some(scalar) = (&$scalar as &Any).downcast_ref::<f64>() {
                match $data {
                    Data::Float64(vec) => { Data::Float64(vec.into_iter().map(|mut v| {v *= scalar; v}).collect()) },
                    Data::Int32(vec) => { Data::Float64(vec.into_iter().map(|v| {let mut v = v as f64; v *= scalar; v}).collect()) }
                }

            // If scalar is i32...
            } else if let Some(scalar) = (&$scalar as &Any).downcast_ref::<i32>() {
                match $data {
                    Data::Float64(vec) => { Data::Float64(vec.into_iter().map(|mut v| {v *= *scalar as f64; v}).collect()) },
                    Data::Int32(vec) => { Data::Int32(vec.into_iter().map(|mut v| {v *= scalar; v}).collect()) }
                }
            } else {
                panic!("Unexpected Scalar type!");
            }
        }
    };


    // Multiply a vector by creating a copy
    // ie: multiply_vec_by_scalar(ref &vec, 2.0);
    (!inplace $data:expr, $scalar:expr) => {
        {
            use containers::{Data};
            use std::any::Any;

            // Get vec which can be any primitive
            // ...and perform an inplace op on it by scalar which can be any primitive.

            // If scalar is f64...
            if let Some(_) = (&$scalar as &Any).downcast_ref::<f64>() {
                match $data {
                    Data::Float64(ref vec) => { Data::Float64(vec.clone().into_iter().map(|mut v| {v *= $scalar as f64; v}).collect()) },
                    Data::Int32(ref vec) => { Data::Float64(vec.clone().into_iter().map(|v| { let mut val = v as f64; val *= $scalar as f64; val}).collect()) }
                }

            // If scalar is i32...
            } else if let Some(_) = (&$scalar as &Any).downcast_ref::<i32>() {
                match $data {
                    Data::Float64(ref vec) => { Data::Float64(vec.clone().into_iter().map(|mut v| {v *= $scalar as f64; v}).collect()) },
                    Data::Int32(ref vec) => { Data::Int32(vec.clone().into_iter().map(|mut v| {v *= $scalar; v}).collect()) }
                }
            } else {
                panic!("Unexpected Scalar type!");
            }

        }
    };
}



