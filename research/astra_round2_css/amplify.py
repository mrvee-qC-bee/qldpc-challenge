"""Bounded CSS tensor-product screen using the unchanged verifier and kit.

The construction is Eqs. 14--15 of arXiv:2609.37231. Distances are only
witness-backed bounds; all candidates and newly found witnesses are retained.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "verify"), str(ROOT / "research/kit")]
import gf2
from coordination import staging_dir
from submit import make_submission, save_submission
from surrogate import distance_rand_witness, validate_logical


def dense(doc, side):
    h = np.zeros((len(doc["checks"][side]), doc["n"]), dtype=np.int8)
    for i, support in enumerate(doc["checks"][side]):
        h[i, support] = 1
    return h


def sparse_basis(h):
    order = np.argsort(h.sum(1), kind="stable")
    _, pivots = gf2.rref(h[order].T)
    return h[order[pivots]]


def product(hx, hz, gx, gz):
    n, a = hx.shape[1], gx.shape[1]
    mx, mz, ax, az = len(hx), len(hz), len(gx), len(gz)

    def z(r, c):
        return np.zeros((r, c), dtype=np.int8)

    def eye(v):
        return np.eye(v, dtype=np.int8)

    x = np.block(
        [
            [np.kron(hx, eye(a)), z(mx * a, mz * ax), np.kron(eye(mx), gz.T)],
            [np.kron(eye(n), gx), np.kron(hz.T, eye(ax)), z(n * ax, mx * az)],
        ]
    )
    zz = np.block(
        [
            [np.kron(hz, eye(a)), np.kron(eye(mz), gx.T), z(mz * a, mx * az)],
            [np.kron(eye(n), gz), z(n * az, mz * ax), np.kron(hx.T, eye(az))],
        ]
    )
    assert not ((x.astype(np.int64) @ zz.T) % 2).any()
    return x, zz


def weight(doc):
    return max(map(len, doc["checks"]["X"] + doc["checks"]["Z"]))


def dominated(board, n, k, d, w):
    return [
        p
        for p, b in board
        if b["n"] <= n
        and b["k"] >= k
        and b["distance"]["d"] >= d
        and weight(b) <= w
        and (b["n"], b["k"], b["distance"]["d"], weight(b)) != (n, k, d, w)
    ]


def parameter_duplicate(board, n, k, d, w):
    return any((b["n"], b["k"], b["distance"]["d"], weight(b)) == (n, k, d, w) for _, b in board)


def package(hx, hz, base, amp, label, out, seed, trials):
    doc = make_submission(
        hx,
        hz,
        name=f"CSS tensor product of {label}",
        construction="Central CSS tensor product, arXiv:2609.37231 Eqs. 14-15; "
        f"base {label[0]}, amplifier {label[1]}, independent sparse check rows.",
        authors=["@mrvee-qC-bee"],
        family="other",
        trials=1,
        seed=seed,
        references=["https://arxiv.org/abs/2609.37231"],
    )
    doc["provenance"].update(model="GPT-6 Astra", novelty="unknown")
    path = out / f"{label[0]}--{label[1]}--initial.json"
    assert not save_submission(doc, str(path))
    # Product logicals are valid upper-bound witnesses, explicitly checked by
    # the trusted kit. No multiplicative lower bound is assumed here.
    for side in ("X", "Z"):
        bw, aw = base["distance"][side]["witness"], amp["distance"][side]["witness"]
        wit = sorted(q * amp["n"] + j for q in bw for j in aw)
        ok, reason = validate_logical(hx, hz, side, len(wit), wit)
        assert ok, reason
        if len(wit) < doc["distance"][side]["value"]:
            doc["distance"][side].update(value=len(wit), witness=wit)
    doc["distance"]["d"] = min(doc["distance"][s]["value"] for s in ("X", "Z"))
    path = path.with_name(path.name.replace("initial", "product-witness"))
    assert not save_submission(doc, str(path))
    found = distance_rand_witness(hx, hz, trials=trials, seed=seed + 3, backend="fast", threads=1, pair_depth=20)
    assert not found.rejected, found.rejected
    if found.side in ("X", "Z"):
        # Save every accelerator witness even if it doesn't beat product bound.
        fdoc = json.loads(json.dumps(doc))
        fdoc["distance"][found.side].update(value=found.weight, witness=found.support)
        fdoc["distance"]["d"] = min(fdoc["distance"][s]["value"] for s in ("X", "Z"))
        assert not save_submission(fdoc, str(path.with_name(path.name.replace("product-witness", "ris-witness"))))
        if found.weight < doc["distance"][found.side]["value"]:
            doc = fdoc
    path = path.with_name(path.name.replace("product-witness", "screened"))
    assert not save_submission(doc, str(path))
    return doc, path, dict(found._asdict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--max-seconds", type=int, default=700)
    ap.add_argument("--run-id", default="astra-round2-css-20261001")
    args = ap.parse_args()
    out = Path(staging_dir(args.run_id))
    board = []
    for p in sorted((ROOT / "codes").glob("*.json")):
        b = json.loads(p.read_text())
        if b.get("code_type", "CSS") == "CSS":
            board.append((p.stem, b))
    amps = []
    for a in (4, 6, 8, 10):
        h = np.ones((1, a), dtype=np.int8)
        adoc = make_submission(
            h,
            h,
            name=f"Even-parity {a}-qubit amplifier",
            construction="Single all-ones X and Z checks on even n.",
            authors=["@mrvee-qC-bee"],
            trials=1,
        )
        assert adoc["distance"]["d"] == 2
        assert not save_submission(adoc, str(out / f"amplifier-{a}.json"))
        amps.append((f"even-{a}", adoc, h, h))
    for name, b in board:
        if b["n"] <= 20 and b["k"] >= 2 and b["distance"]["d"] >= 3:
            amps.append((name, b, sparse_basis(dense(b, "X")), sparse_basis(dense(b, "Z"))))
    ranked = []
    seen = set()
    for name, b in board:
        if b["n"] > 150 or not all(b["distance"].get(s, {}).get("witness") for s in ("X", "Z")):
            continue
        hx, hz = [sparse_basis(dense(b, s)) for s in ("X", "Z")]
        for aname, a, gx, gz in amps:
            n = b["n"] * a["n"] + len(hz) * len(gx) + len(hx) * len(gz)
            k = b["k"] * a["k"]
            d = min(b["distance"][s]["value"] * a["distance"][s]["value"] for s in ("X", "Z"))
            w = max(
                int(hx.sum(1).max() + gz.sum(0).max()),
                int(gx.sum(1).max() + hz.sum(0).max()),
                int(hz.sum(1).max() + gx.sum(0).max()),
                int(gz.sum(1).max() + hx.sum(0).max()),
            )
            if n > 700 and not (n <= 1000 and w <= 8 and d <= 40):
                continue
            if d < 4 or dominated(board, n, k, d, w) or parameter_duplicate(board, n, k, d, w) or (n, k, d, w) in seen:
                continue
            seen.add((n, k, d, w))
            ranked.append((k * d * d / n, name, b, hx, hz, aname, a, gx, gz, n, k, d, w))
    ranked.sort(key=lambda x: -x[0])
    summary = [
        {"base": v[1], "amplifier": v[5], "n": v[9], "k": v[10], "product_bound": v[11], "w": v[12], "score": v[0]}
        for v in ranked
    ]
    (ROOT / "research/astra_round2_css/predicted-frontier.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"predicted_survivors": len(ranked), "top": summary[:25]}), flush=True)
    start = time.time()
    for i, v in enumerate(ranked[: args.limit]):
        if time.time() - start > args.max_seconds:
            break
        _, name, b, hx, hz, aname, a, gx, gz, *_ = v
        xx, zz = product(hx, hz, gx, gz)
        t0 = time.time()
        doc, path, found = package(xx, zz, b, a, (name, aname), out, 20261001 + 100 * i, args.trials)
        r = {
            "base": name,
            "amplifier": aname,
            "n": doc["n"],
            "k": doc["k"],
            "d": doc["distance"]["d"],
            "w": weight(doc),
            "path": str(path.relative_to(ROOT)),
            "ris": found,
            "dominators": dominated(board, doc["n"], doc["k"], doc["distance"]["d"], weight(doc)),
            "seconds": time.time() - t0,
        }
        with (ROOT / "research/astra_round2_css/screen.jsonl").open("a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
