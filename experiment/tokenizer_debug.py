"""Notebook script: inspect Hugging Face tokenizer behavior on sample text."""
from __future__ import annotations

from transformers import AutoTokenizer

from src.utils import load_yaml


def main() -> None:
    """Run a quick tokenizer encode/decode sanity check."""
    config = load_yaml("configs/train_config.yaml")
    tokenizer_name = config["tokenizer"]["hf_tokenizer_name"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    sample_text = "Mini-GPT pretraining starts with high-quality cleaned text."

    token_ids = tokenizer.encode(sample_text, add_special_tokens=False)
    decoded_text = tokenizer.decode(token_ids)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print(f"Tokenizer: {tokenizer_name}")
    print(f"Input text: {sample_text}")
    print(f"Token ids: {token_ids}")
    print(f"Tokens: {tokens}")
    print(f"Decoded text: {decoded_text}")


if __name__ == "__main__":
    main()
