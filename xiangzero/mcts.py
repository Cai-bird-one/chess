from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from .game import ACTION_SIZE, Move, XiangqiGame


@dataclass
class Node:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return bool(self.children)


class MCTS:
    def __init__(self, model: torch.nn.Module, simulations: int = 100, c_puct: float = 1.5, device: str = "cpu"):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = torch.device(device)

    @torch.no_grad()
    def run(self, game: XiangqiGame, temperature: float = 1.0) -> tuple[Move, np.ndarray]:
        root = Node(0.0)
        self._expand(root, game)
        if root.children:
            self._add_dirichlet_noise(root)

        for _ in range(self.simulations):
            node = root
            scratch = game.clone()
            path = [node]

            while node.expanded():
                action, node = self._select_child(node)
                scratch = scratch.apply(Move.from_action(action))
                path.append(node)

            terminal, value = scratch.is_terminal()
            if not terminal:
                value = self._expand(node, scratch)
            self._backpropagate(path, value)

        visits = np.zeros(ACTION_SIZE, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        if visits.sum() == 0:
            legal = game.legal_moves()
            move = random.choice(legal)
            visits[move.action] = 1
            return move, visits / visits.sum()

        policy = _visit_policy(visits, temperature)
        action = int(np.random.choice(np.arange(ACTION_SIZE), p=policy))
        return Move.from_action(action), policy

    def _select_child(self, node: Node) -> tuple[int, Node]:
        total = math.sqrt(max(1, node.visit_count))

        def score(item: tuple[int, Node]) -> float:
            _, child = item
            prior_score = self.c_puct * child.prior * total / (1 + child.visit_count)
            return -child.value + prior_score

        return max(node.children.items(), key=score)

    @torch.no_grad()
    def _expand(self, node: Node, game: XiangqiGame) -> float:
        legal = game.legal_moves()
        if not legal:
            terminal, value = game.is_terminal()
            return value if terminal else 0.0

        state = torch.from_numpy(game.encode()).unsqueeze(0).to(self.device)
        logits, value = self.model(state)
        probs = F.softmax(logits[0], dim=0).detach().cpu().numpy()
        legal_actions = [m.action for m in legal]
        priors = probs[legal_actions]
        total = float(priors.sum())
        if total <= 0:
            priors = np.full(len(legal_actions), 1 / len(legal_actions), dtype=np.float32)
        else:
            priors = priors / total
        for action, prior in zip(legal_actions, priors):
            node.children[action] = Node(float(prior))
        return float(value.item())

    def _backpropagate(self, path: list[Node], value: float) -> None:
        for node in reversed(path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def _add_dirichlet_noise(self, root: Node, frac: float = 0.25, alpha: float = 0.3) -> None:
        actions = list(root.children)
        noise = np.random.dirichlet([alpha] * len(actions))
        for action, n in zip(actions, noise):
            child = root.children[action]
            child.prior = child.prior * (1 - frac) + float(n) * frac


def _visit_policy(visits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 1e-6:
        policy = np.zeros_like(visits)
        policy[int(visits.argmax())] = 1
        return policy
    adjusted = visits ** (1.0 / temperature)
    return adjusted / adjusted.sum()
