"""Notebook script: plot training loss and perplexity from logged metrics."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt



def main() -> None:
    """Load saved training metrics and generate figure files."""
    metrics_path = Path("outputs/logs/train_metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError("Run training first so outputs/logs/train_metrics.json exists.")

    import json
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    history = metrics.get("history", [])
    if not history:
        raise ValueError("No logged history found in train_metrics.json.")

    steps = [entry["step"] for entry in history]
    losses = [entry["loss"] for entry in history]
    perplexities = [entry["perplexity"] for entry in history]

    figures_dir = Path("outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    loss_path = figures_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(steps, perplexities, marker="o", color="orange")
    plt.title("Training Perplexity")
    plt.xlabel("Step")
    plt.ylabel("Perplexity")
    plt.grid(True, alpha=0.3)
    ppl_path = figures_dir / "perplexity_curve.png"
    plt.tight_layout()
    plt.savefig(ppl_path)
    plt.close()

    print(f"Saved loss curve to {loss_path}")
    print(f"Saved perplexity curve to {ppl_path}")


if __name__ == "__main__":
    main()
