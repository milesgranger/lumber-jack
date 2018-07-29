
use std::ffi::CString;
use pyo3::prelude::*;


fn call_python_func(source_series: CString, target_series: CString, function: CString) -> PyResult<()> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let sys = py.import("sys")?;
    let _version: String = sys.get("version")?.extract()?;

    let locals = PyDict::new(py);
    let py_bytes = PyByteArray::new(py, function.as_bytes());
    //println!("bytes: {:?}", &py_bytes);
    locals.set_item("os", py.import("os")?)?;
    locals.set_item("cloudpickle", py.import("cloudpickle")?)?;
    locals.set_item("pickled_func", Some(py_bytes))?;
    locals.set_item("source_series", Some(PyByteArray::new(py, source_series.as_bytes())))?;
    py.run("func = cloudpickle.loads(bytes(pickled_func))",  None, Some(&locals))?;

    Ok(())
}

#[no_mangle]
pub extern "C" fn series_map(source_ptr: *mut u8, source_len: u32,
                             target_ptr: *mut u8, target_len: u32,
                             func_ptr: *mut u8,   func_len: u32) -> f64 {
    let func_vec = unsafe { Vec::from_raw_parts(func_ptr, func_len as usize, func_len as usize) };
    let source_vec = unsafe { Vec::from_raw_parts(source_ptr, source_len as usize, source_len as usize) };
    let target_vec = unsafe { Vec::from_raw_parts(target_ptr, target_len as usize, target_len as usize) };
    //println!("Got vector of: {:?}", &vec);

    let func = unsafe { CString::from_vec_unchecked(func_vec) };
    let source_series = unsafe { CString::from_vec_unchecked(source_vec) };
    let target_series = unsafe { CString::from_vec_unchecked(target_vec) };
    //println!("Got func bytes of: {:?}", &func);

    let _result = call_python_func(source_series, target_series, func).unwrap();
    1.0

}