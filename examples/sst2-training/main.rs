use anyhow::Result;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Row;
use parquet::record::RowAccessor;
use serde::{Deserialize, Serialize};

use reqwest;

/// Download the SST-2 data from huggingface
fn get_data() -> Result<Vec<(i32, String, bool)>> {
    let potential_urls =
        reqwest::blocking::get("https://huggingface.co/api/datasets/sst2/parquet/default/train")?
            .text()?;
    let urls: Vec<String> = serde_json::from_str(potential_urls.as_str())?;
    let data = reqwest::blocking::get(&urls[0])?.bytes()?;
    let reader = SerializedFileReader::new(data)?;

    let mut data = Vec::new();
    // But much for turning Vec<Result<..>> to Result<Vec<..>>
    let items: Result<Vec<Row>, _> = reader.get_row_iter(None)?.collect();
    for item in items? {
        let idx = item.get_int(0)?.to_owned();
        let content = item.get_string(1)?.to_owned();
        let label = item.get_long(2)?.to_owned() == 1;
        data.push((idx, content, label));
    }
    Ok(data)
}

fn main() -> Result<()> {
    let data = get_data()?;
    for d in data {
        println!("{:?}", d);
    }
    Ok(())
}
