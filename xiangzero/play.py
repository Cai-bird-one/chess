from __future__ import annotations

import argparse

import torch

from .game import XiangqiGame
from .mcts import MCTS
from .model import AlphaZeroNet, checkpoint_config, load_model_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=80)
    parser.add_argument("--checkpoint")
    parser.add_argument("--moves", type=int, default=20)
    parser.add_argument("--channels", type=int)
    parser.add_argument("--blocks", type=int)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = checkpoint_config(args.checkpoint, map_location=device) if args.checkpoint else {}
    channels = args.channels or config.get("channels", 96)
    blocks = args.blocks or config.get("blocks", 6)
    model = AlphaZeroNet(channels=channels, blocks=blocks).to(device)
    if args.checkpoint:
        load_model_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()
    mcts = MCTS(model, simulations=args.sims, device=device)
    game = XiangqiGame()

    print(game.render())
    for idx in range(args.moves):
        move, _ = mcts.run(game, temperature=0.1)
        print(f"\n{idx + 1}. {'red' if game.side == 1 else 'black'} {move.ucci()}")
        game = game.apply(move)
        print(game.render())
        terminal, value = game.is_terminal()
        if terminal:
            print(f"game over: value={value}")
            break


if __name__ == "__main__":
    main()
