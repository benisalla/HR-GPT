from model.config import SPECIAL_TOKENS
from transformers import GPT2TokenizerFast

class MyTokenizer:
    def __init__(self, pretrained_model_name: str = "gpt2"):
        # Use the fast tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_model_name)

        # Keep GPT-2's original EOS
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
        
        # Convenience IDs
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

        # vocab size 
        self.vocab_size = self.tokenizer.vocab_size + len(SPECIAL_TOKENS)

        # info
        print(f"Tokenizer initialized with vocab size: {self.vocab_size}")

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True, **kwargs):
        ids = self.tokenizer.encode(text, add_special_tokens=False, **kwargs)
        if add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    def __call__(self, texts, padding=False, truncation=False, max_length=None, return_tensors=None):
        # Batch-friendly API
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            add_special_tokens=False,  
        )

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str):
        self.tokenizer.save_pretrained(path)