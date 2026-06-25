from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .game import ACTION_SIZE

DEFAULT_CHANNELS = 96
DEFAULT_BLOCKS = 6


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AlphaZeroNet(nn.Module):
    def __init__(self, channels: int = DEFAULT_CHANNELS, blocks: int = DEFAULT_BLOCKS):
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(15, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 10 * 9, ACTION_SIZE),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(16 * 10 * 9, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tower(self.stem(x))
        return self.policy(x), self.value_fc(self.value_conv(x)).squeeze(-1)


def checkpoint_config(path: str, map_location: str | torch.device = "cpu") -> dict[str, int]:
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config = checkpoint["config"]
        return {
            "channels": int(config.get("channels", DEFAULT_CHANNELS)),
            "blocks": int(config.get("blocks", DEFAULT_BLOCKS)),
        }
    return {"channels": DEFAULT_CHANNELS, "blocks": DEFAULT_BLOCKS}


def load_model_checkpoint(
    model: AlphaZeroNet,
    path: str,
    map_location: str | torch.device = "cpu",
) -> None:
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def save_model_checkpoint(model: AlphaZeroNet, path: str) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "channels": model.channels,
                "blocks": model.blocks,
            },
        },
        path,
    )
