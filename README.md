# XiangZero

一个最小可运行的 AlphaZero 风格中国象棋 AI 项目。

## 功能

- 中国象棋棋盘、合法走法生成、将帅照面检测
- 竞赛规则中的将死、困毙、三次循环作和、60 回合无吃子自然限着
- `90 x 90` 起点到终点策略空间
- PyTorch 残差网络：策略头 + 价值头
- MCTS 搜索
- 自我对弈数据生成
- 生成局面 -> 训练 -> 保存 checkpoint 的完整循环

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m xiangzero.play --sims 40
python -m xiangzero.pipeline --iterations 3 --games-per-iteration 4 --sims 40 --epochs 2
```

如果本机已经安装了 `torch` 和 `numpy`，也可以不创建虚拟环境，直接运行上面的 `python -m ...` 命令。

## 训练流程

完整训练入口会在每一轮执行：

1. 用当前 checkpoint 加载模型。
2. 通过 MCTS 自我对弈生成新局面。
3. 追加到 replay buffer。
4. 用累计局面继续训练模型。
5. 保存 `checkpoints/model_iter_XXXX.pt`，并更新 `checkpoints/latest.pt`。

示例：

```bash
python -m xiangzero.pipeline \
  --iterations 10 \
  --games-per-iteration 20 \
  --sims 100 \
  --epochs 2 \
  --batch-size 64 \
  --channels 128 \
  --blocks 8
```

如果要从已有模型继续：

```bash
python -m xiangzero.pipeline --initial-checkpoint checkpoints/model.pt
```

注意：checkpoint 的网络结构必须和 `--channels/--blocks` 一致。旧的默认模型是 `--channels 96 --blocks 6`；如果要训练更大的 `128x8` 或 `192x12`，建议使用新的 `--checkpoint-dir` 和 `--replay-buffer` 从头开始，或者先准备同结构的初始 checkpoint。

建议训练档位：

```bash
# 原型验证：约 780 万参数
python -m xiangzero.pipeline --channels 96 --blocks 6 --iterations 5 --games-per-iteration 8 --sims 40 --epochs 2

# 入门增强：约 2600 万参数，适合 8GB GPU 小批量训练
python -m xiangzero.pipeline --channels 128 --blocks 8 --iterations 20 --games-per-iteration 20 --sims 100 --epochs 2 --batch-size 32

# 更认真训练：需要更长时间和更多数据
python -m xiangzero.pipeline --channels 192 --blocks 12 --iterations 50 --games-per-iteration 50 --sims 200 --epochs 2 --batch-size 16
```

仍然可以单独运行自我对弈或训练：

```bash
python -m xiangzero.self_play --games 2 --sims 40 --out data/selfplay.jsonl
python -m xiangzero.train --data data/selfplay.jsonl --epochs 2 --checkpoint checkpoints/model.pt
```

## 说明

这是一套工程骨架，不是预训练强棋力模型。棋力来自大量自我对弈训练、更多 MCTS 模拟次数、更大的网络和更完善的训练调度。

棋规参考 XQBase 收录的 1999 年版《象棋竞赛规则》：https://www.xqbase.com/protocol/rule.htm

目前已覆盖行棋规则、王不见王、将死、困毙、循环三次和自然限着。复杂“棋例”中的长将、长捉、一将一杀等细分裁决还没有完全展开，后续可以单独加入局面威胁分类器。
