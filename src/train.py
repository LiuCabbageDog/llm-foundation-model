"""Training entrypoint for transformer-based Mini-GPT language modeling."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.optim import AdamW

from dataset import create_dataloader
from model import MiniGPT
from utils import load_yaml, save_json, set_seed


def main() -> None:
    """Load processed data, train Mini-GPT, and save checkpoint + metrics."""
    config = load_yaml("configs/train_config.yaml")
    training_config = config["training"]
    model_config = config["model"]

    set_seed(int(training_config["seed"]))

    data_blob = torch.load(config["data"]["processed_dataset_path"], map_location="cpu")
    sequences = data_blob["sequences"]
    vocab_size = int(data_blob["vocab_size"])
    block_size = int(data_blob["block_size"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # PyTorch DataLoader for shuffled mini-batches.
    train_loader = create_dataloader(
        sequences=sequences,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
    )

    # Transformer-based Mini-GPT model.
    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        embed_dim=int(model_config["embed_dim"]),
        n_heads=int(model_config["n_heads"]),
        n_layers=int(model_config["n_layers"]),
        dropout=float(model_config.get("dropout", 0.1)),
        mlp_ratio=int(model_config.get("mlp_ratio", 4)),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=float(training_config["learning_rate"]))

    training_history: list[dict[str, float | int]] = []
    global_step = 0
    model.train()

    for epoch_index in range(int(training_config["epochs"])):
        for input_ids, target_ids in train_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            _, loss = model(input_ids, target_ids)
            assert loss is not None

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_config.get("grad_clip", 1.0)))
            optimizer.step()

            global_step += 1
            if global_step % int(training_config["log_interval"]) == 0:
                loss_value = float(loss.detach().cpu())
                perplexity_value = math.exp(min(loss_value, 20.0))

                training_history.append(
                    {
                        "step": global_step,
                        "epoch": epoch_index + 1,
                        "loss": loss_value,
                        "perplexity": perplexity_value,
                    }
                )
                print(
                    f"step={global_step} epoch={epoch_index + 1} "
                    f"loss={loss_value:.4f} ppl={perplexity_value:.2f}"
                )

    outputs_dir = Path("outputs")
    (outputs_dir / "logs").mkdir(parents=True, exist_ok=True)

    checkpoint_path = outputs_dir / "mini_gpt.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "vocab_size": vocab_size,
            "block_size": block_size,
        },
        checkpoint_path,
    )

    save_json(
        {
            "history": training_history,
            "checkpoint": str(checkpoint_path),
            "num_logged_steps": len(training_history),
        },
        outputs_dir / "logs" / "train_metrics.json",
    )
    print(f"Training complete. Saved checkpoint to {checkpoint_path}.")


if __name__ == "__main__":
    main()
