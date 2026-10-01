"""Rebuild the final code or audit all compact archived cases with trusted tools.

No custom distance routine is present. Optional fresh certification calls the
unchanged SAT verifier. Any unexpected counterexample is saved through the kit.
"""

import argparse
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "verify/gf2.py").exists())
sys.path[:0] = [str(HERE), str(ROOT / "verify"), str(ROOT / "research/kit"), str(ROOT / "research/local2d")]
import gf2
import sat_certify
from boundary_engine import _cleanup
from coordination import staging_dir
from qldpc_verify import _stabilizer_block_count, verify
from rectangle_constructor import construct
from submit import save_submission
from surrogate import validate_logical


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def supports(h):
    return [list(map(int, np.flatnonzero(row))) for row in h]


def dense(rows, n):
    h = np.zeros((len(rows), n), dtype=np.int8)
    for i, row in enumerate(rows):
        h[i, row] = 1
    return h


def contract(state, move):
    a, b, xy, ids = (x.copy() for x in state)
    side, original_q, original_support = move
    q = list(ids).index(original_q)
    h = (a, b)[side]
    rows = [r for r, row in enumerate(h) if list(map(int, ids[np.flatnonzero(row)])) == original_support]
    assert rows, "Recorded pivot absent"
    r = rows[0]
    pivot = h[r].copy()
    for rr in np.flatnonzero(h[:, q]):
        if rr != r:
            h[rr] ^= pivot
    if side == 0:
        a = np.delete(a, r, 0)
    else:
        b = np.delete(b, r, 0)
    a, b = np.delete(a, q, 1), np.delete(b, q, 1)
    xy, ids = np.delete(xy, q, 0), np.delete(ids, q)
    a, b, keep = _cleanup(a, b)
    return a, b, xy[keep], ids[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", type=Path, default=ROOT / "codes/925-173-3.json")
    ap.add_argument("--certify", action="store_true", help="Fresh unchanged SAT proof that no logical below3 exists")
    ap.add_argument(
        "--audit-evidence", action="store_true", help="Rebuild every archived case and validate every witness"
    )
    args = ap.parse_args()
    archive = json.loads((HERE / "search-evidence.json").read_text())
    recipe = json.loads((HERE / "recipe.json").read_text())
    cases = archive["cases"]

    @lru_cache(maxsize=64)
    def rebuild(case_id):
        if case_id in archive["sources"]:
            source = archive["sources"][case_id]
            path = source["path"]
            path = HERE / Path(path).name if path.startswith("research/astra_round2_geometry/") else ROOT / path
            doc = json.loads(path.read_text())
            assert digest(doc["checks"]) == source["checks_sha256"]
            a, b = [dense(doc["checks"][side], doc["n"]) for side in ("X", "Z")]
            return a, b, np.asarray(doc["locality"]["coordinates"]), np.arange(doc["n"])
        entry = cases[case_id]
        rule = entry["recipe"]
        kind = rule["kind"]
        if kind == "rectangle":
            p = rule["parameters"]
            (a, b), _ = construct(**p)
            a, b = a.astype(np.int8), b.astype(np.int8)
            xy = np.asarray([(x, y) for x in range(p["W"]) for y in range(p["H"])], dtype=float)
            state = (a, b, xy, np.arange(a.shape[1]))
        else:
            a, b, xy, ids = (value.copy() for value in rebuild(rule["parent"]))
            if kind == "contract":
                if rule.get("reset_labels"):
                    ids = np.arange(a.shape[1])
                state = contract((a, b, xy, ids), rule["move"])
            elif kind == "sequence":
                state = (a, b, xy, np.arange(a.shape[1]))
                for move in rule["moves"]:
                    state = contract(state, move)
            elif kind == "repair":
                ids = np.arange(a.shape[1])
                for fix in rule["fixes"]:
                    support = [list(ids).index(q) for q in fix["support_in_origin"]]
                    ok, why = validate_logical(a, b, fix["side"], len(support), support)
                    assert ok, why
                    row = np.zeros(a.shape[1], dtype=np.int8)
                    row[support] = 1
                    if fix["side"] == "X":
                        a = np.vstack([a, row])
                    else:
                        b = np.vstack([b, row])
                    a, b, keep = _cleanup(a, b)
                    xy, ids = xy[keep], ids[keep]
                state = (a, b, xy, ids)
            elif kind == "project":
                blocks = rule["blocks"]
                assert sorted(q for block in blocks for q in block["indices"]) == list(range(a.shape[1]))
                for h, side in ((a, "X"), (b, "Z")):
                    ranks = [gf2.rank(h[:, block["indices"]]) for block in blocks]
                    assert ranks == [block["rank_" + side] for block in blocks]
                    assert sum(ranks) == gf2.rank(h)
                assert all(block["k"] == block["n"] - block["rank_X"] - block["rank_Z"] for block in blocks)
                assert all(block["k"] == 0 for block in blocks[1:])
                assert _stabilizer_block_count(a, b, a.shape[1])[1] == [block["n"] for block in blocks]
                keep = rule["keep"]
                a, b, xy, ids = a[:, keep], b[:, keep], xy[keep], ids[keep]
                state = (a[a.any(1)], b[b.any(1)], xy, ids)
            else:
                raise AssertionError(f"Unknown recipe kind: {kind}")
        a, b, xy, _ = state
        assert a.shape[1] == entry["n"]
        assert digest({"X": supports(a), "Z": supports(b)}) == entry["checks_sha256"], case_id
        assert digest(xy.tolist()) == entry["coordinates_sha256"], case_id
        return state

    count = 0
    chosen = list(cases) if args.audit_evidence else [recipe["final_case"]]
    for case_id in chosen:
        entry = cases[case_id]
        a, b, _, _ = rebuild(case_id)
        assert a.shape[1] - gf2.rank(a) - gf2.rank(b) == entry["k"]
        assert not np.any((a @ b.T) % 2)
        for variant in entry["variants"]:
            for side in ("X", "Z"):
                witness = variant["distance"][side]
                ok, why = validate_logical(a, b, side, witness["value"], witness["witness"])
                assert ok, (case_id, side, why)
                count += 1
        for side, result in entry.get("sat", {}).get("sides", {}).items():
            if result["status"] == "SAT":
                assert any(variant["distance"][side]["witness"] == result["witness"] for variant in entry["variants"])
    a, b, xy, _ = rebuild(recipe["final_case"])
    candidate = json.loads(args.candidate.read_text())
    assert candidate["checks"] == {"X": supports(a), "Z": supports(b)}
    assert candidate["locality"]["coordinates"] == xy.tolist()
    assert digest(candidate["checks"]) == recipe["checks_sha256"]
    assert digest(xy.tolist()) == recipe["coordinates_sha256"]
    report = verify(candidate)
    assert report["ok"], report
    result = dict(
        n=candidate["n"],
        k=candidate["k"],
        d_upper=candidate["distance"]["d"],
        arrays_reproduced=True,
        archive_cases_audited=len(chosen),
        stored_witness_slots_validated=count,
        fingerprint=report["fingerprint"],
        structural_ok=report["ok"],
    )
    if args.certify:
        cert = sat_certify.certify(candidate, tlim=30)
        result["fresh_sat"] = cert
        for side, side_result in cert["sides"].items():
            if side_result["status"] == "SAT":
                candidate["distance"][side] = dict(
                    value=len(side_result["witness"]), confidence="upper_bound", witness=side_result["witness"]
                )
                candidate["distance"]["d"] = min(candidate["distance"][s]["value"] for s in ("X", "Z"))
                path = Path(staging_dir()) / f"unexpected-925-refutation-{side}.json"
                assert not save_submission(candidate, str(path))
                result["unexpected_counterexample_saved"] = str(path)
        assert cert["d_exact"], result
    print(json.dumps(result, indent=2))
    print(f"fingerprint={report['fingerprint']}")


if __name__ == "__main__":
    main()
