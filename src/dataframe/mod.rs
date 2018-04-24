#![allow(dead_code)]  // TODO: Remove this after more work is put into dataframe impl.
use self::super::series::{Series, LumberJackData};

pub struct DataFrame<T>
    where T: LumberJackData
{
    index: Vec<Series<T>>,
    columns: Vec<Series<T>>
}