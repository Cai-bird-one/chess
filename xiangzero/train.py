from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .game import ACTION_SIZE, XiangqiGame
from .model import AlphaZeroNet, load_model_checkpoint, save_model_checkpoint


class JsonlDataset(Dataset):
    def __init__(self, path: str):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        state = XiangqiGame(row["board"], row["side"]).encode()
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        policy[row["policy"]] = row["policy_values"]
        return state, policy, np.float32(row["value"])


def train_model(
    data_path: str,
    checkpoint_path: str,
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 1e-3,
    init_checkpoint: str | None = None,
    device: str | None = None,
    channels: int = 96,
    blocks: int = 6,
) -> list[float]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = JsonlDataset(data_path)
    if len(dataset) == 0:
        raise ValueError(f"no training rows found in {data_path}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = AlphaZeroNet(channels=channels, blocks=blocks).to(device)
    if init_checkpoint:
        load_model_checkpoint(model, init_checkpoint, map_location=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    losses: list[float] = []
    for epoch in range(epochs):
        total_loss = 0.0
        for state, target_policy, target_value in loader:
            state = state.to(device)
            target_policy = target_policy.to(device)
            target_value = target_value.to(device)
            logits, value = model(state)
            policy_loss = -(target_policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(value, target_value)
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(loader))
        losses.append(avg_loss)
        print(f"epoch {epoch + 1}: loss={avg_loss:.4f}")

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_model_checkpoint(model, str(path))
    return losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=6)
    args = parser.parse_args()

    train_model(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        init_checkpoint=args.init_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        channels=args.channels,
        blocks=args.blocks,
    )


if __name__ == "__main__":
    main()
