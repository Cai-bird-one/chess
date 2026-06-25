from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .game import Move

PIECES = {
    "K": "帅",
    "A": "仕",
    "B": "相",
    "N": "马",
    "R": "车",
    "C": "炮",
    "P": "兵",
    "k": "将",
    "a": "士",
    "b": "象",
    "n": "马",
    "r": "车",
    "c": "炮",
    "p": "卒",
}


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["top_moves"] = top_moves(row, limit=8)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def top_moves(row: dict, limit: int = 8) -> list[dict]:
    pairs = sorted(zip(row["policy"], row["policy_values"]), key=lambda item: item[1], reverse=True)
    return [
        {
            "move": Move.from_action(int(action)).ucci(),
            "prob": float(prob),
        }
        for action, prob in pairs[:limit]
    ]


def build_html(rows: list[dict], title: str, source: Path) -> str:
    data = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    escaped_title = html.escape(title)
    escaped_source = html.escape(str(source))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --paper: #f6efe1;
      --ink: #30281f;
      --line: #8b5a2b;
      --red: #b42318;
      --black: #1f2933;
      --panel: #ffffff;
      --muted: #6b6258;
      --accent: #1d6f7a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ede7dc;
    }}
    main {{
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(360px, 640px) minmax(300px, 420px);
      gap: 24px;
      align-items: start;
      justify-content: center;
      padding: 24px;
    }}
    .board-wrap {{
      width: min(100%, 640px);
    }}
    .board {{
      position: relative;
      aspect-ratio: 8 / 9;
      background: var(--paper);
      border: 3px solid var(--line);
      box-shadow: 0 16px 40px rgba(58, 44, 29, 0.18);
      user-select: none;
    }}
    .hline, .vline {{
      position: absolute;
      background: var(--line);
      opacity: 0.82;
    }}
    .hline {{ height: 1.5px; left: 0; right: 0; }}
    .vline {{ width: 1.5px; top: 0; bottom: 0; }}
    .river {{
      position: absolute;
      left: 12.5%;
      right: 12.5%;
      top: 44.45%;
      height: 11.1%;
      display: flex;
      align-items: center;
      justify-content: space-around;
      color: rgba(139, 90, 43, 0.5);
      font-size: clamp(20px, 4vw, 34px);
      letter-spacing: 0.2em;
      pointer-events: none;
    }}
    .piece {{
      position: absolute;
      width: 9.5%;
      aspect-ratio: 1;
      transform: translate(-50%, -50%);
      border-radius: 999px;
      border: 2px solid currentColor;
      background: #fffaf0;
      display: grid;
      place-items: center;
      font-size: clamp(16px, 3.8vw, 30px);
      font-weight: 700;
      box-shadow: 0 3px 8px rgba(44, 32, 20, 0.18);
    }}
    .red {{ color: var(--red); }}
    .black {{ color: var(--black); }}
    .panel {{
      background: var(--panel);
      border: 1px solid rgba(48, 40, 31, 0.12);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(58, 44, 29, 0.08);
    }}
    h1 {{
      font-size: 22px;
      margin: 0 0 6px;
      letter-spacing: 0;
    }}
    .source {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 18px;
      word-break: break-all;
    }}
    .stats {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 18px;
    }}
    .stat {{
      border: 1px solid rgba(48, 40, 31, 0.12);
      border-radius: 6px;
      padding: 10px;
      min-height: 62px;
    }}
    .label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .value {{
      font-size: 18px;
      font-weight: 700;
    }}
    .controls {{
      display: grid;
      grid-template-columns: 44px 44px 1fr 44px 44px;
      gap: 8px;
      align-items: center;
      margin-bottom: 18px;
    }}
    button {{
      height: 40px;
      border: 1px solid rgba(48, 40, 31, 0.18);
      border-radius: 6px;
      background: #f8f5ee;
      color: var(--ink);
      font-size: 18px;
      cursor: pointer;
    }}
    button:hover {{ border-color: var(--accent); color: var(--accent); }}
    input[type="range"] {{ width: 100%; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid rgba(48, 40, 31, 0.1);
      padding: 8px 4px;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    @media (max-width: 900px) {{
      main {{
        grid-template-columns: 1fr;
        padding: 16px;
      }}
      .board-wrap {{ margin: 0 auto; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="board-wrap">
      <div id="board" class="board" aria-label="xiangqi board"></div>
    </section>
    <aside class="panel">
      <h1>{escaped_title}</h1>
      <div class="source">{escaped_source}</div>
      <div class="controls">
        <button id="first" title="第一步">|&lt;</button>
        <button id="prev" title="上一步">&lt;</button>
        <input id="slider" type="range" min="0" max="0" value="0">
        <button id="next" title="下一步">&gt;</button>
        <button id="last" title="最后一步">&gt;|</button>
      </div>
      <div class="stats">
        <div class="stat"><span class="label">局面</span><span id="index" class="value"></span></div>
        <div class="stat"><span class="label">走棋方</span><span id="side" class="value"></span></div>
        <div class="stat"><span class="label">价值</span><span id="value" class="value"></span></div>
        <div class="stat"><span class="label">策略候选</span><span id="policy-count" class="value"></span></div>
      </div>
      <table>
        <thead><tr><th>Top move</th><th>MCTS prob</th></tr></thead>
        <tbody id="moves"></tbody>
      </table>
    </aside>
  </main>
  <script>
    const rows = {data};
    const pieces = {json.dumps(PIECES, ensure_ascii=False)};
    const board = document.getElementById("board");
    const slider = document.getElementById("slider");
    const indexEl = document.getElementById("index");
    const sideEl = document.getElementById("side");
    const valueEl = document.getElementById("value");
    const policyCountEl = document.getElementById("policy-count");
    const movesEl = document.getElementById("moves");
    let current = 0;

    function pct(value) {{
      return (value * 100).toFixed(4) + "%";
    }}

    function drawGrid() {{
      board.innerHTML = "";
      for (let r = 0; r < 10; r++) {{
        const line = document.createElement("div");
        line.className = "hline";
        line.style.top = pct(r / 9);
        board.appendChild(line);
      }}
      for (let c = 0; c < 9; c++) {{
        const line = document.createElement("div");
        line.className = "vline";
        line.style.left = pct(c / 8);
        board.appendChild(line);
      }}
      const river = document.createElement("div");
      river.className = "river";
      river.textContent = "楚河  汉界";
      board.appendChild(river);
    }}

    function render(i) {{
      current = Math.max(0, Math.min(rows.length - 1, i));
      const row = rows[current];
      drawGrid();
      for (let idx = 0; idx < row.board.length; idx++) {{
        const piece = row.board[idx];
        if (piece === ".") continue;
        const el = document.createElement("div");
        el.className = "piece " + (piece === piece.toUpperCase() ? "red" : "black");
        el.textContent = pieces[piece] || piece;
        el.style.left = pct((idx % 9) / 8);
        el.style.top = pct(Math.floor(idx / 9) / 9);
        board.appendChild(el);
      }}
      slider.value = String(current);
      indexEl.textContent = `${{current + 1}} / ${{rows.length}}`;
      sideEl.textContent = row.side === 1 ? "红方" : "黑方";
      valueEl.textContent = Number(row.value).toFixed(3);
      policyCountEl.textContent = String(row.policy.length);
      movesEl.innerHTML = "";
      for (const move of row.top_moves) {{
        const tr = document.createElement("tr");
        const a = document.createElement("td");
        const p = document.createElement("td");
        a.textContent = move.move;
        p.textContent = (move.prob * 100).toFixed(2) + "%";
        tr.appendChild(a);
        tr.appendChild(p);
        movesEl.appendChild(tr);
      }}
    }}

    slider.max = String(Math.max(0, rows.length - 1));
    slider.addEventListener("input", () => render(Number(slider.value)));
    document.getElementById("first").addEventListener("click", () => render(0));
    document.getElementById("prev").addEventListener("click", () => render(current - 1));
    document.getElementById("next").addEventListener("click", () => render(current + 1));
    document.getElementById("last").addEventListener("click", () => render(rows.length - 1));
    window.addEventListener("keydown", (event) => {{
      if (event.key === "ArrowLeft") render(current - 1);
      if (event.key === "ArrowRight") render(current + 1);
    }});
    render(0);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="self-play JSONL file")
    parser.add_argument("--out", default="visualizations/selfplay.html")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--title", default="XiangZero Self-Play Viewer")
    args = parser.parse_args()

    source = Path(args.input)
    rows = load_rows(source, limit=args.limit)
    if not rows:
        raise ValueError(f"no rows found in {source}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(rows, args.title, source), encoding="utf-8")
    print(f"wrote {len(rows)} positions to {out}")


if __name__ == "__main__":
    main()
