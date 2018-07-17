/*

This module consists of vector operations; sum, multiply, divide, etc.

*/

use std::iter::Sum;

/// Sum a vector which consists of values allowed to be summed and return a Vec of size one
/// which plays well with DataPtr
pub fn sum_vec<'a, T>(vec: &'a Vec<T>) -> Vec<T>
    where T: Sum<&'a T>
{
    let mut result = Vec::with_capacity(1_usize);
    result.push(vec.iter().sum());
    result
}