#!/usr/bin/env bash
set -euo pipefail

cd /home/birdcly/chess
mkdir -p logs checkpoints_128x8 data

python3 -u -m xiangzero.pipeline \
  --channels 128 \
  --blocks 8 \
  --iterations 30 \
  --games-per-iteration 10 \
  --sims 64 \
  --epochs 2 \
  --batch-size 32 \
  --max-plies 160 \
  --checkpoint-dir checkpoints_128x8 \
  --replay-buffer data/replay_128x8.jsonl \
  --data-dir data \
  > logs/train_128x8.log 2>&1
