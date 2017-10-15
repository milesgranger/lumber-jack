extern crate csv;

use std::error::Error;
use std::path::Path;
use self::csv::Reader;

pub fn read_csv(file_path: &str) -> Result<(), Box<Error>> {
    /*
        Read a file from a given path and return results
    */
    if Path::new(&file_path).exists() {

        let mut reader = Reader::from_path(&file_path)?;

        for result in reader.records(){
            let record = result?;
            println!("{:?}", record);
        }

    } else {
        print!("File does not exist!");
    }
    Ok(())
}

