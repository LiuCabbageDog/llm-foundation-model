"""Configuration for Assignment 1 data collection.

The corpus is collected in a 4:3:3 ratio with a target merged raw-text size
of about 1.3 GB, measured as UTF-8 encoded text bytes written to JSONL.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"

# Use decimal megabytes so the total lands close to 1.3 GB in the report.
MB = 1_000_000
TOTAL_TARGET_MB = 1300

TARGETS_MB = {
    "wikipedia": 520,   # 40%
    "news": 390,        # 30%
    "web": 390,         # 30%
}

DATASET_SPECS = {
    "wikipedia": {
        # Prefer script-free parquet exports first (datasets>=3 removed
        # loading dataset scripts such as wikipedia.py).
        "dataset_candidates": [
            {"dataset_name": "wikimedia/wikipedia", "config_name": "20231101.en"},
            {"dataset_name": "wikipedia", "config_name": "20220301.en"},
        ],
        "dataset_name": "wikimedia/wikipedia",
        "config_name": "20231101.en",
        "split": "train",
        "source_label": "encyclopedic",
        "output_file": OUTPUT_DIR / "wikipedia_520mb.jsonl",
    },
    "news": {
        "dataset_name": "abisee/cnn_dailymail",
        "config_name": "3.0.0",
        "split": "train",
        "source_label": "news",
        "output_file": OUTPUT_DIR / "news_390mb.jsonl",
    },
    "web": {
        "dataset_name": "Skylion007/openwebtext",
        "config_name": None,
        "split": "train",
        "source_label": "general_web",
        "output_file": OUTPUT_DIR / "web_390mb.jsonl",
    },
}

MERGED_OUTPUT_FILE = OUTPUT_DIR / "corpus_merged_1300mb.jsonl"
COLLECTION_SUMMARY_FILE = LOG_DIR / "collection_summary.json"

# Initial quality filters for the collection stage.
MIN_WORDS = 50
MAX_CHARS = 200_000
NORMALIZE_WHITESPACE = True
LOWERCASE = False  # Keep case for a more natural pretraining corpus.
