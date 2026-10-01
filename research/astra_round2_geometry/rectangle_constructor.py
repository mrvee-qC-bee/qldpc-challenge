"""Frozen original rectangle constructor; no distance search.

Affinely punctured checkerboard grammar: @mathysrennela, notes/961-169-3.md.
Rectangular completion and this finite contraction campaign: @vprusso.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "verify/gf2.py").exists())
sys.path[:0] = [str(ROOT / "research/kit"), str(ROOT / "verify"), str(ROOT / "research/local2d")]

try:
    import gf2_fast as F
except ImportError:
    from gf2 import rref

    class F:
        gf2_rref = staticmethod(rref)


def dense(rows, n):
    h = np.zeros((len(rows), n), dtype=np.uint8)
    for i, support in enumerate(rows):
        h[i, list(support)] = 1
    return h


def initial(W, H, px, pz):
    rows, holes = [[], []], [[], []]
    for x in range(W - 1):
        for y in range(H - 1):
            side = (x + y) % 2
            support = (x * H + y, x * H + y + 1, (x + 1) * H + y, (x + 1) * H + y + 1)
            phase = (px, pz)[side]
            (holes if (x + (3 if side == 0 else 2) * y) % 5 == phase else rows)[side].append(support)
    return rows, holes


def construct(W, H, px, pz, order):
    n = W * H
    rows, holes = initial(W, H, px, pz)
    cols = [[0] * n for _ in range(2)]

    def append(side, support):
        bit = 1 << len(rows[side])
        rows[side].append(tuple(support))
        for q in support:
            cols[side][q] ^= bit

    for side in [0, 1]:
        for i, support in enumerate(rows[side]):
            for q in support:
                cols[side][q] ^= 1 << i
    rng = np.random.default_rng(924995100 + order)
    pairs = []
    for x in range(W):
        for y in range(H):
            for dx, dy in [(0, 1), (1, -1), (1, 0), (1, 1)]:
                xx, yy = x + dx, y + dy
                if 0 <= xx < W and 0 <= yy < H:
                    # Every supplied pin in the baseline touches this band.
                    if min(x, y, W - 1 - x, H - 1 - y, xx, yy, W - 1 - xx, H - 1 - yy) <= 1:
                        pairs.append((x * H + y, xx * H + yy))
    candidates = [(side, a, b) for side in [0, 1] for a, b in pairs]
    permutation = rng.permutation(len(candidates))
    priority = {tuple(candidates[i]): rank for rank, i in enumerate(permutation)}
    # Pure algebraic completion: commutation, boundary coverage, and local
    # independent rows. No logical operators are enumerated or classified.
    # Order0/1 favor one Pauli side, order2/3 use a mixed fixed-seed order.
    pin_count = [0, 0]
    for epoch in range(3):
        while True:
            available = []
            for side, a, b in candidates:
                if cols[1 - side][a] != cols[1 - side][b]:
                    continue
                if cols[side][a] == cols[side][b] and cols[side][a] != 0:
                    # This is only a redundant-candidate heuristic; the final
                    # row basis and ranks are computed by trusted native GF2.
                    continue
                zero = (cols[side][a] == 0) + (cols[side][b] == 0)
                favored = int(side == order) if order < 2 else 0
                available.append(((-zero, -favored, priority[(side, a, b)]), side, a, b))
            if not available:
                break
            _, side, a, b = min(available)
            append(side, (a, b))
            pin_count[side] += 1
            candidates.remove((side, a, b))
        missing = [(side, q) for side in [0, 1] for q in range(n) if cols[side][q] == 0]
        if not missing:
            break
        changed = False
        for side, q in missing:
            if cols[side][q]:
                continue
            for support in holes[side]:
                if q not in support:
                    continue
                syndrome = 0
                for a in support:
                    syndrome ^= cols[1 - side][a]
                if syndrome == 0:
                    append(side, support)
                    holes[side].remove(support)
                    changed = True
                    break
        if not changed:
            break
    missing = sum(c == 0 for side in cols for c in side)
    if missing:
        return None, {"uncovered_qubit_sector_pairs": missing}
    matrices = []
    for side in [0, 1]:
        h = dense(rows[side], n)
        _, independent = F.gf2_rref(h.T)
        matrices.append(h[independent])
    # Sparse bit-column construction already maintains commutation; verify it
    # independently here, using integer matrix multiplication only.
    assert not np.any((matrices[0] @ matrices[1].T) % 2)
    k = n - len(matrices[0]) - len(matrices[1])
    return matrices, {
        "k": k,
        "n": n,
        "pin_rows_added": pin_count,
        "remaining_holes": list(map(len, holes)),
        "ranks": list(map(len, matrices)),
        "g_at_hypothetical_d3": 9 * k / n,
    }


def supports(matrix):
    return [list(map(int, np.flatnonzero(row))) for row in matrix]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
