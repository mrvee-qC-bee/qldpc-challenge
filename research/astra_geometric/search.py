"""Boundary contraction search; all distance searches use the unchanged kit.

Starts from an attributed public code. Persist every packaged proposal and SAT
counterexample. Final candidates still require validate_candidate's full gate.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "verify"), str(ROOT / "research/kit"), str(ROOT / "research/local2d")]
import gf2

try:
    import gf2_fast
except ImportError:
    gf2_fast = None
import sat_certify
from boundary_engine import _cleanup
from coordination import staging_dir
from qldpc_verify import _stabilizer_block_count
from submit import make_submission, save_submission


def dense(rows, n):
    a = np.zeros((len(rows), n), dtype=np.int8)
    for r, support in enumerate(rows):
        a[r, support] = 1
    return a


def contract(hx, hz, xy, ids, side, q, r):
    a, b = hx.copy(), hz.copy()
    own = (a, b)[side]
    pivot = own[r].copy()
    for rr in np.flatnonzero(own[:, q]):
        if rr != r:
            own[rr] ^= pivot
    if side == 0:
        a = np.delete(a, r, axis=0)
    else:
        b = np.delete(b, r, axis=0)
    a, b = np.delete(a, q, axis=1), np.delete(b, q, axis=1)
    c, labels = np.delete(xy, q, axis=0), np.delete(ids, q)
    a, b, kept = _cleanup(a, b)
    return a, b, c[kept], labels[kept]


def structural(a, b, xy, k):
    rank = gf2_fast.gf2_rank if gf2_fast is not None else gf2.rank
    if a.shape[1] - rank(a) - rank(b) != k:
        return False
    for h in (a, b):
        if int(h.sum(1).max(initial=0)) > 4:
            return False
        for row in h:
            pts = xy[np.flatnonzero(row)]
            if len(pts) and ((pts[:, None] - pts[None, :]) ** 2).sum(2).max() > 2.00000001:
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="codes/945-172-3.json")
    ap.add_argument("--seed", type=int, default=6100101)
    ap.add_argument("--max-proposals", type=int, default=600)
    ap.add_argument("--max-seconds", type=float, default=1200)
    args = ap.parse_args()
    out = Path(staging_dir())
    base = json.loads((ROOT / args.base).read_text())
    hx, hz = [dense(base["checks"][s], base["n"]) for s in ("X", "Z")]
    xy = np.array(base["locality"]["coordinates"])
    ids = np.arange(base["n"])
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    tested, rejected, moves = set(), set(), []
    count = 0
    journal = out / "journal.jsonl"
    while count < args.max_proposals and time.time() - t0 < args.max_seconds:
        proposals = []
        for side, h in enumerate((hx, hz)):
            degree, weights = h.sum(0), h.sum(1)
            for q in range(h.shape[1]):
                for r in np.flatnonzero(h[:, q]):
                    if degree[q] == 1 or weights[r] == 2:
                        key = (side, int(ids[q]), tuple(map(int, ids[np.flatnonzero(h[r])])))
                        if key not in rejected:
                            proposals.append((side, q, int(r), key))
        rng.shuffle(proposals)
        progressed = False
        for side, q, r, key in proposals:
            if count >= args.max_proposals or time.time() - t0 >= args.max_seconds:
                break
            state = contract(hx, hz, xy, ids, side, q, r)
            a, b, c, labels = state
            rejected.add(key)
            if not structural(a, b, c, base["k"]):
                continue
            digest = hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest()
            if digest in tested:
                continue
            tested.add(digest)
            count += 1
            seed = args.seed + 100 * count
            ts = time.time()
            doc = make_submission(
                a,
                b,
                name=f"Boundary-contracted checkerboard candidate {count}",
                construction=(
                    f"Local pivot contractions of {args.base}; "
                    "affine checkerboard grammar by @mathysrennela, "
                    "rectangular completion by @vprusso."
                ),
                authors=["@mrvee-qC-bee"],
                family="topological",
                trials=1,
                seed=seed,
                coordinates=c.tolist(),
                layers=1,
            )
            doc["provenance"]["model"] = "GPT-6 Astra"
            doc["provenance"]["novelty"] = "unknown"
            path = out / f"proposal-{count:04d}-{doc['n']}-{doc['k']}-{doc['distance']['d']}.json"
            assert not save_submission(doc, str(path))
            record = {
                "proposal": count,
                "path": path.name,
                "move": key,
                "n": doc["n"],
                "k": doc["k"],
                "screen_d": doc["distance"]["d"],
                "seed": seed,
                "ancestor_moves": list(moves),
            }
            if doc["distance"]["d"] >= 3:
                # This unchanged exact solver refutes d<3, or proves none exists.
                satdoc = dict(doc, distance=dict(doc["distance"], d=3))
                cert = sat_certify.certify(satdoc, tlim=15)
                (out / f"proposal-{count:04d}-sat.json").write_text(json.dumps(cert, indent=2))
                record["sat"] = cert
                for s, result in cert["sides"].items():
                    if result["status"] == "SAT":
                        doc["distance"][s] = {
                            "value": len(result["witness"]),
                            "confidence": "upper_bound",
                            "witness": result["witness"],
                        }
                        doc["distance"]["d"] = min(doc["distance"][ss]["value"] for ss in ("X", "Z"))
                        assert not save_submission(doc, str(path.with_name(path.stem + "-refuted.json")))
                if cert["d_exact"]:
                    hx, hz, xy, ids = state
                    moves.append(key)
                    record["accepted"] = True
                    record["blocks"] = _stabilizer_block_count(hx, hz, hx.shape[1])
                    (out / "best.json").write_text(
                        json.dumps(
                            {
                                "path": str(path),
                                "moves": moves,
                                "retained_base_indices": list(map(int, ids)),
                                "base": args.base,
                                "g": 9 * base["k"] / doc["n"],
                                "blocks": record["blocks"],
                            },
                            indent=2,
                        )
                    )
                    progressed = True
            record["seconds"] = time.time() - ts
            with journal.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps({k: v for k, v in record.items() if k not in ("sat", "ancestor_moves")}), flush=True)
            if progressed:
                break
        if not progressed:
            break
    print(
        json.dumps(
            {
                "finished": True,
                "proposals": count,
                "accepted": len(moves),
                "n": hx.shape[1],
                "k": base["k"],
                "g": 9 * base["k"] / hx.shape[1],
                "seconds": time.time() - t0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
