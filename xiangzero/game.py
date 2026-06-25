from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

BOARD_H = 10
BOARD_W = 9
BOARD_SIZE = BOARD_H * BOARD_W
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
NATURAL_DRAW_PLIES = 120

RED = 1
BLACK = -1

INITIAL_BOARD = (
    "rnbakabnr"
    "........."
    ".c.....c."
    "p.p.p.p.p"
    "........."
    "........."
    "P.P.P.P.P"
    ".C.....C."
    "........."
    "RNBAKABNR"
)

PIECE_PLANES = {
    "K": 0,
    "A": 1,
    "B": 2,
    "N": 3,
    "R": 4,
    "C": 5,
    "P": 6,
    "k": 7,
    "a": 8,
    "b": 9,
    "n": 10,
    "r": 11,
    "c": 12,
    "p": 13,
}


@dataclass(frozen=True, slots=True)
class Move:
    src: int
    dst: int

    @property
    def action(self) -> int:
        return self.src * BOARD_SIZE + self.dst

    @classmethod
    def from_action(cls, action: int) -> "Move":
        return cls(action // BOARD_SIZE, action % BOARD_SIZE)

    def ucci(self) -> str:
        return f"{_square_name(self.src)}{_square_name(self.dst)}"


@dataclass(frozen=True, slots=True)
class GameResult:
    terminal: bool
    value: float
    reason: str = "ongoing"


class XiangqiGame:
    def __init__(
        self,
        board: str = INITIAL_BOARD,
        side: int = RED,
        ply: int = 0,
        halfmove_clock: int = 0,
        history: tuple[str, ...] | None = None,
    ):
        if len(board) != BOARD_SIZE:
            raise ValueError("board must contain 90 squares")
        self.board = board
        self.side = side
        self.ply = ply
        self.halfmove_clock = halfmove_clock
        self.history = history if history is not None else (self.position_key(),)

    def clone(self) -> "XiangqiGame":
        return XiangqiGame(self.board, self.side, self.ply, self.halfmove_clock, self.history)

    def position_key(self) -> str:
        return f"{self.side}:{self.board}"

    def legal_moves(self) -> list[Move]:
        moves: list[Move] = []
        for src, piece in enumerate(self.board):
            if piece == "." or _piece_side(piece) != self.side:
                continue
            for dst in self._pseudo_destinations(src, piece):
                move = Move(src, dst)
                child = self.apply(move)
                if not child.is_in_check(-child.side):
                    moves.append(move)
        return moves

    def apply(self, move: Move) -> "XiangqiGame":
        piece = self.board[move.src]
        if piece == ".":
            raise ValueError("source square is empty")
        captured = self.board[move.dst]
        cells = list(self.board)
        cells[move.src] = "."
        cells[move.dst] = piece
        board = "".join(cells)
        side = -self.side
        halfmove_clock = 0 if captured != "." else self.halfmove_clock + 1
        key = f"{side}:{board}"
        return XiangqiGame(
            board,
            side,
            self.ply + 1,
            halfmove_clock,
            self.history + (key,),
        )

    def is_terminal(self) -> tuple[bool, float]:
        result = self.result()
        return result.terminal, result.value

    def result(self) -> GameResult:
        if self.history.count(self.position_key()) >= 3:
            return GameResult(True, 0.0, "threefold_repetition")
        if self.halfmove_clock >= NATURAL_DRAW_PLIES:
            return GameResult(True, 0.0, "natural_draw")

        legal = self.legal_moves()
        if legal:
            return GameResult(False, 0.0)
        if self.is_in_check(self.side):
            return GameResult(True, -1.0, "checkmate")
        return GameResult(True, -1.0, "stalemate")

    def is_in_check(self, side: int) -> bool:
        king = "K" if side == RED else "k"
        try:
            king_sq = self.board.index(king)
        except ValueError:
            return True
        opponent = -side
        for src, piece in enumerate(self.board):
            if piece == "." or _piece_side(piece) != opponent:
                continue
            if king_sq in self._pseudo_destinations(src, piece, validate_king=False):
                return True
        return self._kings_face()

    def encode(self) -> np.ndarray:
        planes = np.zeros((15, BOARD_H, BOARD_W), dtype=np.float32)
        for idx, piece in enumerate(self.board):
            plane = PIECE_PLANES.get(piece)
            if plane is not None:
                r, c = divmod(idx, BOARD_W)
                planes[plane, r, c] = 1.0
        planes[14, :, :] = 1.0 if self.side == RED else 0.0
        return planes

    def render(self) -> str:
        rows = [self.board[i : i + BOARD_W] for i in range(0, BOARD_SIZE, BOARD_W)]
        return "\n".join(rows)

    def _pseudo_destinations(
        self, src: int, piece: str, validate_king: bool = True
    ) -> Iterable[int]:
        row, col = divmod(src, BOARD_W)
        lower = piece.lower()
        side = _piece_side(piece)

        if lower == "k":
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                dst = _idx(row + dr, col + dc)
                if dst is not None and _in_palace(dst, side) and self._can_land(piece, dst):
                    yield dst

        elif lower == "a":
            for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                dst = _idx(row + dr, col + dc)
                if dst is not None and _in_palace(dst, side) and self._can_land(piece, dst):
                    yield dst

        elif lower == "b":
            for dr, dc in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
                eye = _idx(row + dr // 2, col + dc // 2)
                dst = _idx(row + dr, col + dc)
                if dst is None or eye is None or self.board[eye] != ".":
                    continue
                dst_row, _ = divmod(dst, BOARD_W)
                if side == RED and dst_row < 5:
                    continue
                if side == BLACK and dst_row > 4:
                    continue
                if self._can_land(piece, dst):
                    yield dst

        elif lower == "n":
            checks = (
                (-2, -1, -1, 0),
                (-2, 1, -1, 0),
                (2, -1, 1, 0),
                (2, 1, 1, 0),
                (-1, -2, 0, -1),
                (1, -2, 0, -1),
                (-1, 2, 0, 1),
                (1, 2, 0, 1),
            )
            for dr, dc, lr, lc in checks:
                leg = _idx(row + lr, col + lc)
                dst = _idx(row + dr, col + dc)
                if dst is not None and leg is not None and self.board[leg] == "." and self._can_land(piece, dst):
                    yield dst

        elif lower in ("r", "c"):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jumped = False
                nr, nc = row + dr, col + dc
                while True:
                    dst = _idx(nr, nc)
                    if dst is None:
                        break
                    target = self.board[dst]
                    if lower == "r":
                        if target == ".":
                            yield dst
                        else:
                            if _piece_side(target) != side:
                                yield dst
                            break
                    else:
                        if not jumped:
                            if target == ".":
                                yield dst
                            else:
                                jumped = True
                        else:
                            if target != ".":
                                if _piece_side(target) != side:
                                    yield dst
                                break
                    nr += dr
                    nc += dc

        elif lower == "p":
            forward = -1 if side == RED else 1
            directions = [(forward, 0)]
            crossed = row <= 4 if side == RED else row >= 5
            if crossed:
                directions.extend([(0, -1), (0, 1)])
            for dr, dc in directions:
                dst = _idx(row + dr, col + dc)
                if dst is not None and self._can_land(piece, dst):
                    yield dst

    def _can_land(self, piece: str, dst: int) -> bool:
        target = self.board[dst]
        return target == "." or _piece_side(target) != _piece_side(piece)

    def _clear_file(self, a: int, b: int) -> bool:
        ar, ac = divmod(a, BOARD_W)
        br, bc = divmod(b, BOARD_W)
        if ac != bc:
            return False
        lo, hi = sorted((ar, br))
        return all(self.board[r * BOARD_W + ac] == "." for r in range(lo + 1, hi))

    def _kings_face(self) -> bool:
        red = self.board.find("K")
        black = self.board.find("k")
        if red < 0 or black < 0:
            return True
        return red % BOARD_W == black % BOARD_W and self._clear_file(red, black)


def action_to_move(action: int) -> Move:
    return Move.from_action(action)


def _idx(row: int, col: int) -> int | None:
    if 0 <= row < BOARD_H and 0 <= col < BOARD_W:
        return row * BOARD_W + col
    return None


def _piece_side(piece: str) -> int:
    return RED if piece.isupper() else BLACK


def _in_palace(idx: int, side: int) -> bool:
    row, col = divmod(idx, BOARD_W)
    if col < 3 or col > 5:
        return False
    return row >= 7 if side == RED else row <= 2


def _square_name(idx: int) -> str:
    row, col = divmod(idx, BOARD_W)
    return f"{chr(ord('a') + col)}{9 - row}"
