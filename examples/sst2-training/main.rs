use std::iter::zip;

use anyhow::Result;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Row;
use parquet::record::RowAccessor;
use rsetlin::{tselin_machine::TsetlinMachine, vectorizer};

use reqwest;
use tokenizers::Tokenizer;

/// Download the SST-2 data from huggingface
fn get_data() -> Result<(Vec<i32>, Vec<String>, Vec<bool>)> {
    let potential_urls =
        reqwest::blocking::get("https://huggingface.co/api/datasets/sst2/parquet/default/train")?
            .text()?;
    let urls: Vec<String> = serde_json::from_str(potential_urls.as_str())?;
    let data = reqwest::blocking::get(&urls[0])?.bytes()?;
    let reader = SerializedFileReader::new(data)?;

    let mut idx = Vec::new();
    let mut content = Vec::new();
    let mut labels = Vec::new();
    // But much for turning Vec<Result<..>> to Result<Vec<..>>
    let items: Result<Vec<Row>, _> = reader.get_row_iter(None)?.collect();
    for item in items? {
        idx.push(item.get_int(0)?.to_owned());
        content.push(item.get_string(1)?.to_owned());
        labels.push(item.get_long(2)?.to_owned() == 1);
    }
    Ok((idx, content, labels))
}

fn main() -> Result<()> {
    let tok_model = include_bytes!("../../data/tokenizer.bin");
    let tokenizer = Tokenizer::from_bytes(tok_model).unwrap();
    let (idx, content, labels) = get_data()?;
    let vectors = vectorizer::vectorize(&content, &tokenizer)?;

    let mut model = TsetlinMachine::new(5000, 100, 15.0, 300.0, tokenizer.get_vocab_size(true));

    let mut correct = 0;
    for (i, (vector, label)) in zip(vectors.iter(), labels.iter()).enumerate().cycle() {
        if (i % 5) == 0 {
            if model.predict(&vector) == *label {
                correct += 1;
            }
        }
        if (i % 500) == 0 {
            println!("At step {} accuracy is {}", i, correct,);
            correct = 0;
        }
        model.fit(vector.clone(), *label);
    }

    Ok(())
}
