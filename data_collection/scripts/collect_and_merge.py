"""Collect a ~1.3 GB raw-text corpus in a 4:3:3 ratio and merge it.

Sources:
- Wikipedia: encyclopedic text
- CNN/DailyMail: news text
- OpenWebText: general web text

The script streams each dataset from Hugging Face, applies light collection-
stage filtering, stops at the requested byte budget, writes one JSONL per
source, and then merges the three JSONL files into one combined corpus.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from datasets import load_dataset
from tqdm import tqdm

from config import (
    COLLECTION_SUMMARY_FILE,
    DATASET_SPECS,
    LOG_DIR,
    LOWERCASE,
    MAX_CHARS,
    MERGED_OUTPUT_FILE,
    MIN_WORDS,
    NORMALIZE_WHITESPACE,
    OUTPUT_DIR,
    TARGETS_MB,
    MB,
)


@dataclass
class SourceStats:
    source_name: str
    target_mb: int
    collected_docs: int = 0
    skipped_empty: int = 0
    skipped_short: int = 0
    skipped_too_long: int = 0
    written_bytes: int = 0
    output_file: str = ""

    @property
    def written_mb(self) -> float:
        return self.written_bytes / MB


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    text = text.strip()
    if LOWERCASE:
        text = text.lower()
    if NORMALIZE_WHITESPACE:
        text = re.sub(r"\s+", " ", text)
    return text


def text_size_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def extract_text(source_key: str, record: Dict) -> str:
    if source_key == "wikipedia":
        return record.get("text", "")
    if source_key == "news":
        # For pretraining, the article body is the cleanest news-style source.
        return record.get("article", "")
    if source_key == "web":
        return record.get("text", "")
    raise ValueError(f"Unsupported source_key: {source_key}")


def iter_streaming_dataset(dataset_name: str, config_name: Optional[str], split: str):
    kwargs = {
        "path": dataset_name,
        "split": split,
        "streaming": True,
    }
    if config_name is not None:
        kwargs["name"] = config_name
    return load_dataset(**kwargs)


def load_streaming_with_fallbacks(spec: Dict):
    """Load a streaming dataset, trying script-free fallbacks first."""
    candidates = spec.get(
        "dataset_candidates",
        [{"dataset_name": spec["dataset_name"], "config_name": spec["config_name"]}],
    )
    last_error: Optional[Exception] = None

    for candidate in candidates:
        dataset_name = candidate["dataset_name"]
        config_name = candidate.get("config_name")
        try:
            return iter_streaming_dataset(
                dataset_name=dataset_name,
                config_name=config_name,
                split=spec["split"],
            )
        except RuntimeError as err:
            message = str(err)
            if "Dataset scripts are no longer supported" in message:
                print(
                    f"[warn] Dataset '{dataset_name}' rejected by installed "
                    "datasets version (dataset script unsupported). Trying next fallback..."
                )
                last_error = err
                continue
            raise
        except Exception as err:  # noqa: BLE001
            last_error = err
            continue

    assert last_error is not None
    raise RuntimeError(
        "Failed to load dataset using all candidates. "
        "If this is a network/proxy issue, verify Hugging Face access. "
        f"Last error: {last_error}"
    ) from last_error


def collect_source(source_key: str) -> SourceStats:
    spec = DATASET_SPECS[source_key]
    target_mb = TARGETS_MB[source_key]
    target_bytes = target_mb * MB
    stats = SourceStats(
        source_name=source_key,
        target_mb=target_mb,
        output_file=str(spec["output_file"]),
    )

    ds = load_streaming_with_fallbacks(spec)

    progress = tqdm(
        total=target_bytes,
        desc=f"Collecting {source_key}",
        unit="B",
        unit_scale=True,
        unit_divisor=1000,
    )

    with open(spec["output_file"], "w", encoding="utf-8") as f:
        for record in ds:
            raw_text = extract_text(source_key, record)
            if not raw_text or not raw_text.strip():
                stats.skipped_empty += 1
                continue

            text = normalize_text(raw_text)
            if len(text.split()) < MIN_WORDS:
                stats.skipped_short += 1
                continue
            if len(text) > MAX_CHARS:
                stats.skipped_too_long += 1
                continue

            json_record = {
                "source": source_key,
                "domain": spec["source_label"],
                "text": text,
            }
            line = json.dumps(json_record, ensure_ascii=False) + "\n"
            line_bytes = len(line.encode("utf-8"))

            if stats.written_bytes + line_bytes > target_bytes:
                break

            f.write(line)
            stats.written_bytes += line_bytes
            stats.collected_docs += 1
            progress.update(line_bytes)

    progress.close()
    return stats


def merge_files(source_files: Iterable[Path], merged_output_file: Path) -> Dict[str, float]:
    merged_bytes = 0
    merged_docs = 0

    with open(merged_output_file, "w", encoding="utf-8") as out_f:
        for source_file in source_files:
            with open(source_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
                    merged_bytes += len(line.encode("utf-8"))
                    merged_docs += 1

    return {
        "merged_output_file": str(merged_output_file),
        "merged_docs": merged_docs,
        "merged_bytes": merged_bytes,
        "merged_mb": merged_bytes / MB,
    }


def write_summary(source_stats: Iterable[SourceStats], merge_stats: Dict[str, float]) -> None:
    payload = {
        "collection_stage": "raw_text_collection_and_merge",
        "size_unit": "decimal_mb_utf8_jsonl_bytes",
        "min_words": MIN_WORDS,
        "max_chars": MAX_CHARS,
        "sources": [
            {
                "source_name": s.source_name,
                "target_mb": s.target_mb,
                "written_mb": round(s.written_mb, 3),
                "written_bytes": s.written_bytes,
                "collected_docs": s.collected_docs,
                "skipped_empty": s.skipped_empty,
                "skipped_short": s.skipped_short,
                "skipped_too_long": s.skipped_too_long,
                "output_file": s.output_file,
            }
            for s in source_stats
        ],
        "merged": {
            "merged_output_file": merge_stats["merged_output_file"],
            "merged_docs": merge_stats["merged_docs"],
            "merged_bytes": merge_stats["merged_bytes"],
            "merged_mb": round(merge_stats["merged_mb"], 3),
        },
    }

    with open(COLLECTION_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    ensure_dirs()

    ordered_sources = ["wikipedia", "news", "web"]
    stats = [collect_source(source_key) for source_key in ordered_sources]

    merge_stats = merge_files(
        source_files=[DATASET_SPECS[key]["output_file"] for key in ordered_sources],
        merged_output_file=MERGED_OUTPUT_FILE,
    )
    write_summary(stats, merge_stats)

    print("\nCollection complete.")
    for s in stats:
        print(
            f"- {s.source_name}: {s.written_mb:.2f} MB, "
            f"{s.collected_docs} docs -> {s.output_file}"
        )
    print(
        f"- merged: {merge_stats['merged_mb']:.2f} MB, "
        f"{merge_stats['merged_docs']} docs -> {merge_stats['merged_output_file']}"
    )
    print(f"- summary: {COLLECTION_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
