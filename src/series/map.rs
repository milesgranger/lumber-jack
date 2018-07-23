use libc::c_char;
use std::mem;
use std::ffi::CString;
use pyo3::prelude::*;

use containers::{DataPtr, from_data_ptr};

fn call_python_func() -> PyResult<()> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let sys = py.import("sys")?;

    let executable: String = sys.get("executable")?.extract()?;
    let locals = PyDict::new(py);
    py.eval("print('Hello from Python called by Rust!')", None, None)?;

    Ok(())
}

#[no_mangle]
pub extern "C" fn series_map(data_ptr: DataPtr, function: *mut c_char) -> f64 {
    let string = unsafe { CString::from_raw(function) };
    println!("Entering series map func! {:?}", string);
    let _data = from_data_ptr(data_ptr);

    let _result = call_python_func().unwrap();

    mem::forget(_data);
    1.0

}