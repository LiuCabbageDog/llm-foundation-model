"""Preprocess collected JSONL corpus into fixed token blocks using Hugging Face AutoTokenizer."""
from __future__ import annotations

import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from utils import iter_jsonl, load_yaml, normalize_whitespace, save_json


def strip_markup_and_noise(text: str) -> str:
    """Remove common HTML/markdown/reference artifacts before tokenization."""
    no_html = re.sub(r"<[^>]+>", " ", text)
    no_markdown_links = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", no_html)
    no_inline_code = re.sub(r"`{1,3}[^`]*`{1,3}", " ", no_markdown_links)
    no_reference_markers = re.sub(r"\[\d+\]", " ", no_inline_code)
    no_url = re.sub(r"https?://\S+|www\.\S+", " ", no_reference_markers)
    return no_url


def normalize_document_text(text: str) -> str:
    """Normalize text by lowercasing, filtering symbols, and fixing whitespace."""
    lowercased_text = text.lower()
    no_symbols_text = re.sub(r"[^\w\s\.,;:!?\-\'\"]", " ", lowercased_text)
    normalized_text = normalize_whitespace(no_symbols_text)
    return normalized_text


def chunk_token_ids(token_ids: list[int], block_size: int) -> list[list[int]]:
    """Split token IDs into non-overlapping blocks of length block_size + 1."""
    chunked_sequences: list[list[int]] = []
    required_length = block_size + 1
    for start_index in range(0, len(token_ids) - required_length + 1, block_size):
        block = token_ids[start_index : start_index + required_length]
        if len(block) == required_length:
            chunked_sequences.append(block)
    return chunked_sequences


def main() -> None:
    """Run corpus cleaning, tokenization, chunking, and artifact export."""
    config = load_yaml("configs/train_config.yaml")

    raw_corpus_path = Path(config["data"]["raw_corpus_path"])
    processed_dataset_path = Path(config["data"]["processed_dataset_path"])
    tokenizer_output_dir = Path(config["data"]["tokenizer_output_dir"])

    min_words = int(config["preprocess"]["min_words"])
    block_size = int(config["training"]["block_size"])
    hf_tokenizer_name = config["tokenizer"]["hf_tokenizer_name"]

    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    unique_document_keys: set[str] = set()
    cleaned_documents: list[str] = []

    duplicate_count = 0
    short_document_count = 0

    for row in iter_jsonl(raw_corpus_path):
        raw_text = str(row.get("text", ""))

        # ===== 5.1 Remove duplicate documents. =====
        dedupe_key = normalize_whitespace(raw_text).lower()
        if not dedupe_key or dedupe_key in unique_document_keys:
            duplicate_count += 1
            continue
        unique_document_keys.add(dedupe_key)

        # ===== 5.2 Normalize text: lowercase, remove extra whitespace, strip irrelevant symbols. =====
        normalized_text = normalize_document_text(raw_text)

        # ===== 5.3 Remove low-quality or very short documents (e.g., fewer than 50 words). =====
        word_count = len(normalized_text.split())
        if word_count < min_words:
            short_document_count += 1
            continue

        # ===== 5.4 Optionally remove HTML tags, markdown, reference markers, and related artifacts. =====
        cleaned_text = strip_markup_and_noise(normalized_text)
        cleaned_text = normalize_whitespace(cleaned_text)
        if not cleaned_text:
            short_document_count += 1
            continue

        cleaned_documents.append(cleaned_text)

    all_sequences: list[list[int]] = []
    for cleaned_text in cleaned_documents:
        encoded_ids = tokenizer.encode(cleaned_text, add_special_tokens=False)
        all_sequences.extend(chunk_token_ids(encoded_ids, block_size=block_size))

    processed_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    required_sequence_length = block_size + 1
    if all_sequences:
        dataset_tensor = torch.tensor(all_sequences, dtype=torch.long)
    else:
        dataset_tensor = torch.empty((0, required_sequence_length), dtype=torch.long)
    torch.save(
        {
            "sequences": dataset_tensor,
            "vocab_size": int(tokenizer.vocab_size),
            "block_size": block_size,
            "tokenizer_name": hf_tokenizer_name,
            "pad_token_id": int(tokenizer.pad_token_id),
        },
        processed_dataset_path,
    )

    tokenizer_output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_output_dir)

    save_json(
        {
            "input_path": str(raw_corpus_path),
            "tokenizer": hf_tokenizer_name,
            "documents_kept": len(cleaned_documents),
            "documents_removed_as_duplicates_or_empty": duplicate_count,
            "documents_removed_as_short_or_low_quality": short_document_count,
            "num_sequences": len(all_sequences),
            "block_size": block_size,
            "output_dataset": str(processed_dataset_path),
            "tokenizer_output_dir": str(tokenizer_output_dir),
        },
        "outputs/logs/preprocess_summary.json",
    )

    print(f"Preprocessing complete. Saved {len(all_sequences)} sequences to {processed_dataset_path}.")


if __name__ == "__main__":
    main()
