use anyhow::Result;
use tokenizers::tokenizer::Tokenizer;

fn tokenize(chunks: &Vec<String>, tokenizer: &Tokenizer) -> Result<Vec<Vec<u32>>> {
    let tokens = chunks
        .iter()
        .map(|chunk| {
            tokenizer
                .encode(chunk.as_str(), true)
                .unwrap()
                .get_ids()
                .to_owned()
        })
        .collect();
    Ok(tokens)
}

fn vectorize(chunks: &Vec<String>, tokenizer: &Tokenizer) -> Result<Vec<Vec<bool>>> {
    let chunks = tokenize(&chunks, &tokenizer)?;
    let mut batch = vec![];
    for chunk in chunks {
        let mut vector = vec![false; tokenizer.get_vocab_size(true)];
        for id in chunk {
            vector[id as usize] = true;
        }
        batch.push(vector);
    }
    Ok(batch)
}

#[test]
fn test_tokenize() {
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
    let tokens = tokenize(
        &vec!["This is a test".to_string(), "with two chunks".to_string()],
        &tokenizer,
    );

    tokens.expect("Oh no");
}

#[test]
fn test_vectorize() {
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
    let batch = vectorize(
        &vec!["This is a test".to_string(), "with two chunks".to_string()],
        &tokenizer,
    );

    batch.expect("Oh no");
}
