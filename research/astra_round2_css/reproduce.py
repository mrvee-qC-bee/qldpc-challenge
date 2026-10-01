"""Reconstruct the six-copy [[574,56,<=20]] code and check retained evidence."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "verify"), str(ROOT / "research/kit")]
import gf2
import qldpc_verify
from surrogate import validate_logical


def matrix(doc, side):
    out = np.zeros((len(doc["checks"][side]), doc["n"]), dtype=np.int8)
    for i, row in enumerate(doc["checks"][side]):
        out[i, row] = 1
    return out


def independent_rows(h):
    order = np.argsort(h.sum(1), kind="stable")
    _, pivots = gf2.rref(h[order].T)
    return h[order[pivots]]


def reconstruct(base):
    hx, hz = (independent_rows(matrix(base, side)) for side in ("X", "Z"))
    n, mx, mz = base["n"], len(hx), len(hz)
    assert (n, mx, mz) == (84, 35, 35)
    eye = np.eye(6, dtype=np.int8)
    ones = np.ones((1, 6), dtype=np.int8)
    x = np.block(
        [
            [np.kron(hx, eye), np.zeros((6 * mx, mz), np.int8), np.kron(np.eye(mx, dtype=np.int8), ones.T)],
            [np.kron(np.eye(n, dtype=np.int8), ones), hz.T, np.zeros((n, mx), np.int8)],
        ]
    )
    z = np.block(
        [
            [np.kron(hz, eye), np.kron(np.eye(mz, dtype=np.int8), ones.T), np.zeros((6 * mz, mx), np.int8)],
            [np.kron(np.eye(n, dtype=np.int8), ones), np.zeros((n, mz), np.int8), hx.T],
        ]
    )
    return x, z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    raw = args.candidate.read_bytes()
    candidate = json.loads(raw)
    base = json.loads((ROOT / "codes/84-14-10.json").read_text())
    hx, hz = reconstruct(base)
    assert np.array_equal(hx, matrix(candidate, "X"))
    assert np.array_equal(hz, matrix(candidate, "Z"))
    assert (candidate["n"], candidate["k"], candidate["distance"]["d"]) == (574, 56, 20)
    rep = qldpc_verify.verify(candidate, refute=False)
    assert rep["ok"], rep
    here = Path(__file__).resolve().parent
    gate = json.loads((here / "candidate-gate-574.json").read_text())
    assert gate["passed"] and gate["gates"]["novelty"]["board_advancing"]
    assert gate["candidate"]["fingerprint"] == rep["fingerprint"]
    assert gate["candidate"]["signature"] == rep["signature"]["hash"]
    validator_hash = hashlib.sha256((ROOT / "verify/validate_candidate.py").read_bytes()).hexdigest()
    assert validator_hash == gate["validator"]["source_sha256"]
    deep = json.loads((here / "deep-evidence-574.json").read_text())
    found = deep["result"]
    ok, why = validate_logical(hx, hz, found["side"], found["weight"], found["support"])
    assert ok, why
    out = {
        "parameters": [574, 56, 20],
        "checks_reconstructed_exactly": True,
        "trusted_structural_verification_passed": True,
        "retained_deep_witness_verified": True,
        "candidate_sha256": hashlib.sha256(raw).hexdigest(),
        "fingerprint": rep["fingerprint"],
        "signature": rep["signature"]["hash"],
        "validator_source_sha256": validator_hash,
        "distance_status": "upper_bound",
        "new_distance_search_run": False,
        "computed": rep["computed"],
    }
    if args.report:
        args.report.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
