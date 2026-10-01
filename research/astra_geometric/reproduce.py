"""Reproduce a recorded contraction recipe and optionally rerun exact distance.

Every operation concerns construction; distance is delegated to the unchanged
challenge SAT certifier and structural verifier.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "verify")]
from qldpc_verify import verify
from sat_certify import certify
from search import contract, dense


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reconstruct(recipe):
    base_bytes = (ROOT / recipe["base"]).read_bytes()
    assert hashlib.sha256(base_bytes).hexdigest() == recipe["base_file_sha256"]
    base = json.loads(base_bytes)
    a, b = [dense(base["checks"][s], base["n"]) for s in ("X", "Z")]
    xy = np.asarray(base["locality"]["coordinates"])
    ids = np.arange(base["n"])
    for side, original, pivot in recipe["moves"]:
        q = list(ids).index(original)
        h = (a, b)[side]
        matches = [r for r, row in enumerate(h) if list(map(int, ids[np.flatnonzero(row)])) == pivot]
        # A redundant presentation can contain identical pivot rows. All
        # copies become zero in the same elimination and cleanup removes
        # them, so selecting the first reproduces the same cleaned code.
        assert matches, f"No pivot matches {pivot} at original qubit {original}"
        a, b, xy, ids = contract(a, b, xy, ids, side, q, matches[0])
    checks = {s: [list(map(int, np.flatnonzero(row))) for row in h] for s, h in [("X", a), ("Z", b)]}
    assert digest(checks) == recipe["checks_sha256"]
    assert list(map(int, ids)) == recipe["retained_base_indices"]
    return checks, xy.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=Path, default=HERE / "recipe.json")
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--certify", action="store_true")
    args = ap.parse_args()
    recipe = json.loads(args.recipe.read_text())
    checks, xy = reconstruct(recipe)
    doc = json.loads(args.candidate.read_text())
    assert doc["checks"] == checks
    assert doc["locality"]["coordinates"] == xy
    report = verify(doc, refute=False)
    assert report["ok"], report
    print(json.dumps({"reproduced": True, "computed": report["computed"]}, indent=2))
    if args.certify:
        cert = certify(doc, tlim=30)
        print(json.dumps(cert, indent=2))
        assert cert["d_exact"], cert


if __name__ == "__main__":
    main()
