from transformers import BartTokenizer

def split_text_into_chunks(text, tokenizer, max_chunk_length=512):
    """
    Splits text into smaller chunks that can be processed by the model.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    current_chunk = []

    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= max_chunk_length:
            chunks.append(tokenizer.decode(current_chunk, clean_up_tokenization_spaces=True))
            current_chunk = []

    if current_chunk:
        chunks.append(tokenizer.decode(current_chunk, clean_up_tokenization_spaces=True))

    return chunks
