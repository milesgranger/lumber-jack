// build.rs
extern crate cbindgen;

use std::env;

fn main() {
    //let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    cbindgen::generate(&"./".to_string())
      .unwrap()
      .write_to_file("./lumberjack/rust/rust_bindings.h");
}