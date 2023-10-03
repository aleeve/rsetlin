use std::env;
use tokenizers::tokenizer::Tokenizer;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args[1].as_str();
    println!("Storing tokenizer at {}", &path);
    let tokenizer = Tokenizer::from_pretrained("bert-base-uncased", None).unwrap();
    tokenizer.save(path, false).unwrap();
}

#[test]
fn load() {
    // Just jot down how I intend to load tokenizers later
    let model = include_bytes!("../../data/tokenizer.bin");
    let token = Tokenizer::from_bytes(model).unwrap();
    token.encode("This is a test", true).unwrap();
}
