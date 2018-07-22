
#[macro_export]
macro_rules! operate_on_vec_by_scalar {

    // Multiply a vector inplace
    // ie: multiply_vec_by_scalar(vec, 2.0);
    (inplace $data:expr, $op:tt, $scalar:expr) => {

        {
            use std::mem;
            use containers::{Data, DType, AsType};
            use std::any::Any;

            // Get vec which can contain any primitive
            // ...and perform an inplace op on it by scalar which can be any primitive.

            // If scalar is f64...
            if let Some(scalar) = (&$scalar as &Any).downcast_ref::<f64>() {

                $data.astype(DType::Float64);

                match $data {
                    Data::Float64(ref mut vec) => {
                        for v in vec.iter_mut() {
                            *v = *v $op scalar;
                        }
                    },
                    _ => panic!("Multiplying any dtype by f64 should result in a vec of f64!")
                }

                mem::forget($data)

            // If scalar is i32...
            } else if let Some(scalar) = (&$scalar as &Any).downcast_ref::<i32>() {
                match $data {
                    Data::Float64(ref mut vec) => {
                            //vec.iter_mut().map(|v| *v $op *scalar as f64).collect()
                            for v in vec.iter_mut() {
                                *v = *v $op *scalar as f64;
                            }
                    },
                    Data::Int32(ref mut vec) => {
                            //vec.iter_mut().map(|v| *v $op *scalar).collect()
                            for v in vec.iter_mut() {
                                *v = *v $op *scalar
                            }
                    }
                }
            } else {
                panic!("Unexpected Scalar type!");
            }
        }
    };


    // Multiply a vector by creating a copy
    // ie: multiply_vec_by_scalar(ref &vec, 2.0);
    (!inplace $data:expr, $op:tt, $scalar:expr) => {
        {
            use containers::{Data};
            use std::any::Any;

            // Get vec which can be any primitive
            // ...and perform an inplace op on it by scalar which can be any primitive.

            // If scalar is f64...
            if let Some(scalar) = (&$scalar as &Any).downcast_ref::<f64>() {
                match $data {
                    Data::Float64(ref vec) => {
                        Data::Float64(
                            vec.clone().iter().map(|v| *v $op scalar).collect()
                        )
                    },
                    Data::Int32(ref vec) => {
                        Data::Float64(
                            vec.clone().iter().map(|v| *v as f64 $op scalar).collect()
                        )
                    }
                }

            // If scalar is i32...
            } else if let Some(scalar) = (&$scalar as &Any).downcast_ref::<i32>() {
                match $data {
                    Data::Float64(ref vec) => {
                        Data::Float64(
                            vec.clone().iter().map(|v| *v $op *scalar as f64).collect()
                        )
                    },
                    Data::Int32(ref vec) => {
                        Data::Int32(
                            vec.clone().iter().map(|v| *v $op scalar).collect()
                        )
                    }
                }
            } else {
                panic!("Unexpected Scalar type!");
            }

        }
    };
}
