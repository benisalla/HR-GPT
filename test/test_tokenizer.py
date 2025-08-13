import pytest
from model.config import SPECIAL_TOKENS
from transformers import GPT2TokenizerFast
from dataset.tokenizer import MyTokenizer

@pytest.fixture(scope="module")
def tok():
    return MyTokenizer()

def test_special_tokens_ids_are_distinct(tok):
    ids = {tok.bos_id, tok.eos_id, tok.pad_id}
    assert len(ids) == 3, "Special tokens must have distinct IDs"

def test_vocab_size_consistency(tok):
    internal_vocab = tok.tokenizer.vocab_size
    assert tok.vocab_size == internal_vocab + len(SPECIAL_TOKENS), "vocab_size should match internal GPT2 tokenizer size with special tokens"

def test_special_tokens_present(tok):
    assert tok.tokenizer.bos_token == "<sost>"
    assert tok.tokenizer.eos_token == "<|endoftext|>"
    assert tok.tokenizer.pad_token == "<pad>"

def test_encode_without_special_tokens_excludes_bos_eos(tok):
    text = "plain text"
    ids = tok.encode(text, add_bos=False, add_eos=False)
    assert tok.bos_id not in ids
    assert tok.eos_id not in ids

def test_encode_with_special_tokens_includes_bos_eos(tok):
    text = "hello world"
    ids = tok.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id

def test_decode_with_and_without_skip(tok):
    text = "example"
    ids = tok.encode(text, add_bos=True, add_eos=True)
    decoded_with_skip = tok.decode(ids, skip_special_tokens=True)
    decoded_without_skip = tok.decode(ids, skip_special_tokens=False)
    assert decoded_with_skip.strip() == text
    assert "<sost>" in decoded_without_skip or tok.tokenizer.bos_token in decoded_without_skip

def test_batch_encode_decode(tok):
    texts = ["short", "a much longer sentence"]
    batch = tok(texts, padding=True, return_tensors="pt")
    decoded = tok.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
    assert decoded[0].strip() == "short"
    assert decoded[1].strip() == "a much longer sentence"

def test_truncation(tok):
    text = "word " * 100
    max_len = 10
    out = tok([text], padding=False, truncation=True, max_length=max_len, return_tensors="pt")
    assert out["input_ids"].shape[1] <= max_len

def test_save_and_reload_preserves_ids(tmp_path, tok):
    save_dir = tmp_path / "tok"
    tok.save(str(save_dir))
    reloaded = GPT2TokenizerFast.from_pretrained(str(save_dir))
    assert reloaded.bos_token_id == tok.bos_id
    assert reloaded.eos_token_id == tok.eos_id
    assert reloaded.pad_token_id == tok.pad_id
