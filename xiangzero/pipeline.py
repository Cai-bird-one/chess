from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from .mcts import MCTS
from .model import AlphaZeroNet, checkpoint_config, load_model_checkpoint
from .self_play import self_play_game
from .train import train_model


def generate_positions(
    checkpoint: Path | None,
    out_path: Path,
    games: int,
    sims: int,
    max_plies: int,
    device: str,
    channels: int,
    blocks: int,
) -> int:
    model = AlphaZeroNet(channels=channels, blocks=blocks).to(device)
    if checkpoint and checkpoint.exists():
        load_model_checkpoint(model, str(checkpoint), map_location=device)
    model.eval()
    mcts = MCTS(model, simulations=sims, device=device)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    positions = 0
    with out_path.open("w", encoding="utf-8") as f:
        for game_idx in range(games):
            samples = self_play_game(mcts, max_plies=max_plies)
            positions += len(samples)
            for sample in samples:
                f.write(json.dumps(sample, separators=(",", ":")) + "\n")
            print(f"self-play game {game_idx + 1}/{games}: {len(samples)} positions")
    return positions


def append_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as f_src, dst.open("a", encoding="utf-8") as f_dst:
        shutil.copyfileobj(f_src, f_dst)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def run_pipeline(
    iterations: int,
    games_per_iteration: int,
    sims: int,
    epochs: int,
    batch_size: int,
    lr: float,
    max_plies: int,
    data_dir: Path,
    checkpoint_dir: Path,
    replay_buffer: Path,
    initial_checkpoint: Path | None = None,
    channels: int = 96,
    blocks: int = 6,
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = checkpoint_dir / "latest.pt"
    current = initial_checkpoint if initial_checkpoint and initial_checkpoint.exists() else None
    if current is None and latest.exists():
        current = latest
    if current is not None:
        config = checkpoint_config(str(current), map_location=device)
        if config["channels"] != channels or config["blocks"] != blocks:
            raise ValueError(
                f"checkpoint {current} uses channels={config['channels']} blocks={config['blocks']}, "
                f"but this run requested channels={channels} blocks={blocks}; use matching "
                "--channels/--blocks or choose a fresh --checkpoint-dir"
            )

    for iteration in range(1, iterations + 1):
        print(f"\n=== iteration {iteration}/{iterations} ===")
        selfplay_path = data_dir / f"selfplay_iter_{iteration:04d}.jsonl"
        positions = generate_positions(
            checkpoint=current,
            out_path=selfplay_path,
            games=games_per_iteration,
            sims=sims,
            max_plies=max_plies,
            device=device,
            channels=channels,
            blocks=blocks,
        )
        append_file(selfplay_path, replay_buffer)
        total_rows = count_jsonl(replay_buffer)
        print(f"generated {positions} positions; replay buffer now has {total_rows} rows")

        next_checkpoint = checkpoint_dir / f"model_iter_{iteration:04d}.pt"
        train_model(
            data_path=str(replay_buffer),
            checkpoint_path=str(next_checkpoint),
            init_checkpoint=str(current) if current else None,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            channels=channels,
            blocks=blocks,
        )
        shutil.copyfile(next_checkpoint, latest)
        current = latest
        print(f"saved checkpoint: {next_checkpoint}")

    return latest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games-per-iteration", type=int, default=1)
    parser.add_argument("--sims", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--replay-buffer", default="data/replay.jsonl")
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=6)
    args = parser.parse_args()

    latest = run_pipeline(
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        sims=args.sims,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_plies=args.max_plies,
        data_dir=Path(args.data_dir),
        checkpoint_dir=Path(args.checkpoint_dir),
        replay_buffer=Path(args.replay_buffer),
        initial_checkpoint=Path(args.initial_checkpoint) if args.initial_checkpoint else None,
        channels=args.channels,
        blocks=args.blocks,
    )
    print(f"\nlatest checkpoint: {latest}")


if __name__ == "__main__":
    main()
