# test/test_tokenizer.py
import os
import pytest
import torch

from dataset.tokenizer import MyTokenizer
from transformers import GPT2TokenizerFast

@pytest.fixture(scope="module")
def tok():
    return MyTokenizer("gpt2")

def test_special_tokens_present(tok):
    # ids exist
    assert tok.bos_id is not None
    assert tok.eos_id is not None
    assert tok.pad_id is not None

    # names are set on the underlying tokenizer
    assert tok.tokenizer.bos_token == "<sost>"
    assert tok.tokenizer.eos_token == "<eost>"
    assert tok.tokenizer.pad_token == "<pad>"
    assert "<ans>" in tok.tokenizer.additional_special_tokens

def test_encode_adds_bos_eos(tok):
    text = "hello world"
    base_ids = tok.tokenizer.encode(text, add_special_tokens=False)
    ids = tok.encode(text)  # defaults: add_bos=True, add_eos=True
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert ids[1:-1] == base_ids

def test_encode_without_specials(tok):
    text = "some text"
    ids = tok.encode(text, add_bos=False, add_eos=False)
    base_ids = tok.tokenizer.encode(text, add_special_tokens=False)
    assert ids == base_ids

def test_decode_roundtrip_ids(tok):
    text = "Tokenization test!"
    ids = tok.encode(text)  # has BOS/EOS
    # skip special tokens when decoding
    decoded = tok.decode(ids, skip_special_tokens=True)
    # Compare by re-tokenizing; avoids whitespace quirks in GPT-2 decoding
    assert tok.tokenizer.encode(decoded, add_special_tokens=False) == tok.tokenizer.encode(
        text, add_special_tokens=False
    )

def test_batch_api_padding(tok):
    batch = ["hi", "a bit longer"]
    out = tok(batch, padding=True, truncation=False, return_tensors="pt")
    # shapes
    assert "input_ids" in out and "attention_mask" in out
    assert out["input_ids"].ndim == 2
    assert out["attention_mask"].shape == out["input_ids"].shape
    # pad id should appear in the shorter sequence row
    row0 = out["input_ids"][0].tolist()
    assert tok.pad_id in row0

def test_save_and_reload(tmp_path, tok):
    save_dir = tmp_path / "tok"
    tok.save(str(save_dir))

    # load underlying tokenizer to ensure specials persisted
    reloaded = GPT2TokenizerFast.from_pretrained(str(save_dir))
    assert reloaded.bos_token == "<sost>"
    assert reloaded.eos_token == "<eost>"
    assert reloaded.pad_token == "<pad>"
    assert "<ans>" in reloaded.additional_special_tokens


# test/test_tokenizer.py (temporary)
if __name__ == "__main__":
    t = MyTokenizer()
    print("BOS/EOS/PAD ids:", t.bos_id, t.eos_id, t.pad_id)
    ids = t.encode("hello world")
    print("Encoded:", ids)
    print("Decoded:", t.decode(ids, True))