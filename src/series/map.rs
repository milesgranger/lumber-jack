use libc::c_char;
use std::mem;
use std::ffi::CString;
use pyo3::prelude::*;
use pyo3::ffi;


use containers::{DataPtr, from_data_ptr};

fn call_python_func(function: CString) -> PyResult<()> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let sys = py.import("sys")?;
    let version: String = sys.get("version")?.extract()?;

    let locals = PyDict::new(py);
    let py_bytes = PyByteArray::new(py, function.as_bytes());
    println!("bytes: {:?}", &py_bytes);
    locals.set_item("os", py.import("os")?)?;
    locals.set_item("cloudpickle", py.import("cloudpickle")?)?;
    locals.set_item("pickled_func", Some(py_bytes))?;
    py.run("print(locals())\nfunc = cloudpickle.loads(bytes(pickled_func))\nprint(func())",  None, Some(&locals))?;

    Ok(())
}

#[no_mangle]
pub extern "C" fn series_map(data_ptr: DataPtr, func_ptr: *mut u8, len: u32) -> f64 {
    let vec = unsafe { Vec::from_raw_parts(func_ptr, len as usize, len as usize) };
    println!("Got vector of: {:?}", &vec);

    let func = unsafe { CString::from_vec_unchecked(vec) };
    println!("Got func bytes of: {:?}", &func);

    let _data = from_data_ptr(data_ptr);

    let _result = call_python_func(func).unwrap();

    mem::forget(_data);
    1.0

}