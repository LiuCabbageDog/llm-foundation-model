"""Notebook script: explore collected corpus quality and size statistics."""
from __future__ import annotations

from pathlib import Path

from src.utils import iter_jsonl, load_yaml


def main() -> None:
    """Print lightweight exploratory statistics for the merged corpus."""
    config = load_yaml("configs/train_config.yaml")
    corpus_path = Path(config["data"]["raw_corpus_path"])

    document_count = 0
    total_words = 0
    min_words = 10**9
    max_words = 0
    preview_samples: list[str] = []

    for row in iter_jsonl(corpus_path):
        text = str(row.get("text", "")).strip()
        word_count = len(text.split())

        document_count += 1
        total_words += word_count
        min_words = min(min_words, word_count)
        max_words = max(max_words, word_count)

        if len(preview_samples) < 3 and text:
            preview_samples.append(text[:250])

    avg_words = total_words / max(document_count, 1)

    print(f"Corpus path: {corpus_path}")
    print(f"Documents: {document_count}")
    print(f"Avg words/doc: {avg_words:.2f}")
    print(f"Min words/doc: {min_words if document_count else 0}")
    print(f"Max words/doc: {max_words}")
    print("\nSample documents:")
    for sample_index, sample_text in enumerate(preview_samples, start=1):
        print(f"\n[{sample_index}] {sample_text}")


if __name__ == "__main__":
    main()
