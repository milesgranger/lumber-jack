// build.rs
extern crate cbindgen;

fn main() {
    cbindgen::generate(&"./".to_string())
      .unwrap()
      .write_to_file("./lumberjack/rust/liblumberjack.h");
}