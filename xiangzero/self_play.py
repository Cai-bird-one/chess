from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .game import RED, XiangqiGame
from .mcts import MCTS
from .model import AlphaZeroNet, load_model_checkpoint


def self_play_game(mcts: MCTS, max_plies: int = 240) -> list[dict]:
    game = XiangqiGame()
    history: list[dict] = []

    for ply in range(max_plies):
        move, policy = mcts.run(game, temperature=1.0 if ply < 20 else 0.1)
        history.append(
            {
                "board": game.board,
                "side": game.side,
                "policy": policy.nonzero()[0].tolist(),
                "policy_values": policy[policy.nonzero()[0]].tolist(),
            }
        )
        game = game.apply(move)
        terminal, value = game.is_terminal()
        if terminal:
            winner = -game.side if value < 0 else 0
            return _attach_values(history, winner)
    return _attach_values(history, 0)


def _attach_values(history: list[dict], winner: int) -> list[dict]:
    for item in history:
        if winner == 0:
            item["value"] = 0.0
        else:
            item["value"] = 1.0 if item["side"] == winner else -1.0
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--sims", type=int, default=80)
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", default="data/selfplay.jsonl")
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=6)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AlphaZeroNet(channels=args.channels, blocks=args.blocks).to(device)
    if args.checkpoint:
        load_model_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()

    mcts = MCTS(model, simulations=args.sims, device=device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for game_idx in range(args.games):
            samples = self_play_game(mcts)
            for sample in samples:
                f.write(json.dumps(sample, separators=(",", ":")) + "\n")
            print(f"game {game_idx + 1}: wrote {len(samples)} positions")


if __name__ == "__main__":
    main()
