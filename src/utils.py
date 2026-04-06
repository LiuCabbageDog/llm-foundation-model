"""Utility helpers shared by preprocessing, training, and notebook scripts."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return it as a Python dictionary."""
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def save_json(payload: dict, path: str | Path) -> None:
    """Write JSON data to disk, creating parent directories if needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace into single spaces and strip boundaries."""
    return re.sub(r"\s+", " ", text).strip()


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    """Yield one parsed JSON object per non-empty line from a JSONL file."""
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            clean_line = line.strip()
            if clean_line:
                yield json.loads(clean_line)
