
use std::mem;
use std::ffi::CString;
use pyo3::prelude::*;

use containers::{DataPtr, DType, Data, from_data_ptr};
use series::arange;


fn call_python_func(function: CString, source_data: &Data, out_dtype: DType) -> PyResult<()> {

    let gil = Python::acquire_gil();
    let py = gil.python();

    let sys = py.import("sys")?;
    let _version: String = sys.get("version")?.extract()?;

    let locals = PyDict::new(py);

    let function_py_bytes = PyByteArray::new(py, function.as_bytes());

    locals.set_item("os", py.import("os")?)?;
    locals.set_item("cloudpickle", py.import("cloudpickle")?)?;
    locals.set_item("pickled_func", Some(function_py_bytes))?;
    let func = py.eval("cloudpickle.loads(bytes(pickled_func))", None, Some(&locals))?;
    locals.set_item("func", Some(func))?;
    match source_data {
        Data::Float64(ref vec) => {
            let mut pyval;
            let val_dict = PyDict::new(py);
            val_dict.set_item("func", Some(func))?;
            for val in vec.iter() {
                pyval = PyFloat::new(py, *val);
                val_dict.set_item("val", Some(pyval))?;
                py.eval("func(val)", None, Some(&val_dict))?;
                //locals.del_item("val")?;
            }
        }
        Data::Int32(ref vec) => {
            println!("Got Int32 type!");
        }
    }
    println!("Ran with values!");
    //py.run("print(func(2.0))",  None, Some(&locals))?;

    Ok(())
}

#[no_mangle]
pub extern "C" fn series_map(func_ptr: *mut u8,
                             func_len: u32,
                             source_ptr: DataPtr,
                             out_dtype: DType) -> DataPtr {

    // Convert the pickled function pointer into CString - ie. b'/x02901/xc920...'
    let func_vec = unsafe { Vec::from_raw_parts(func_ptr, func_len as usize, func_len as usize) };
    let func = unsafe { CString::from_vec_unchecked(func_vec) };
    let data = from_data_ptr(source_ptr.clone());

    let _result = call_python_func(func, &data, out_dtype).unwrap();
    mem::forget(data);
    arange(0, 10, DType::Float64)

}