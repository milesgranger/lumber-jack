use libc::c_char;
use std::mem;
use std::ffi::CString;
use pyo3::prelude::*;
use pyo3::ffi;

use containers::{DataPtr, from_data_ptr};

fn call_python_func(function: *mut c_char) -> PyResult<()> {

    let gil = Python::acquire_gil();
    let py = gil.python();
    let sys = py.import("sys")?;
    let version: String = sys.get("version")?.extract()?;

    let locals = PyDict::new(py);
    let os_module = py.import("os")?;
    let function = unsafe { CString::from_raw(function) };
    let py_bytes = PyByteArray::new(py, function.as_bytes());
    println!("bytes: {:?}", &py_bytes);
    locals.set_item("os", os_module)?;
    locals.set_item("pickle", Some(py_bytes))?;
    py.run("print(locals())\nprint(locals())",  None, Some(&locals))?;

    Ok(())
}

#[no_mangle]
pub extern "C" fn series_map(data_ptr: DataPtr, function: *mut c_char) -> f64 {
    let _data = from_data_ptr(data_ptr);

    let _result = call_python_func(function).unwrap();

    mem::forget(_data);
    1.0

}