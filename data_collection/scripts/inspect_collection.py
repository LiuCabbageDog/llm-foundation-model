"""Quick inspection utility for collected JSONL files."""
from __future__ import annotations

import json
from pathlib import Path

from config import OUTPUT_DIR


def inspect_jsonl(path: Path, max_rows: int = 3) -> None:
    print(f"\n== {path.name} ==")
    if not path.exists():
        print("File does not exist.")
        return

    size_mb = path.stat().st_size / 1_000_000
    print(f"size_mb={size_mb:.2f}")

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_rows:
                break
            obj = json.loads(line)
            preview = obj["text"][:160].replace("\n", " ")
            print(f"[{idx}] source={obj['source']} preview={preview}...")


if __name__ == "__main__":
    for file in sorted(OUTPUT_DIR.glob("*.jsonl")):
        inspect_jsonl(file)
