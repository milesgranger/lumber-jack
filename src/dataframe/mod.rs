
use self::super::series::{Series, LumberJackData};

pub struct DataFrame<T>
    where T: LumberJackData
{
    index: Vec<Series<T>>,
    columns: Vec<Series<T>>
}