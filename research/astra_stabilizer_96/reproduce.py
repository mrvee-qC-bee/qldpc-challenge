"""Rebuild the 96-qubit stabilizer fold and verify its retained witnesses.

Run from the repository root with no arguments. An optional candidate argument
selects the JSON to compare, and --output saves the reconstructed submission.
No distance search or board scan is performed.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "verify"), str(ROOT / "research/kit")]
import heuristic_distance as hd
from qldpc_verify import verify
from submit import save_submission


def digest(value):
    """Hash a JSON value independently of whitespace and key ordering."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def checks():
    """Return S=(A|BP) on Z_24 x Z_4, indexed by (i,j) -> 4*i+j."""
    a_terms = [(9, 3), (4, 0), (6, 0), (19, 2)]
    b_terms = [(22, 1), (8, 1), (7, 0), (5, 1)]
    generators = []
    for i in range(24):
        for j in range(4):
            # A shifts positively; P inverts the column of B.
            xs = sorted(4 * ((i + a) % 24) + ((j + b) % 4) for a, b in a_terms)
            zs = sorted(4 * ((-i - a) % 24) + ((-j - b) % 4) for a, b in b_terms)
            generators.append({"X": xs, "Z": zs})
    return {"S": generators}


def verify_parent(doc):
    """Recover the published CSS parent and compare both matrices exactly."""
    a_matrix, folded_b = hd.stabilizer_matrices(doc)
    inversion = [4 * ((-i) % 24) + ((-j) % 4) for i in range(24) for j in range(4)]
    b_matrix = folded_b[:, inversion]
    hx = np.concatenate([a_matrix, b_matrix], axis=1)
    hz = np.concatenate([b_matrix.T, a_matrix.T], axis=1)
    parent = json.loads((ROOT / "codes/192-20-16.json").read_text())
    for side, matrix in (("X", hx), ("Z", hz)):
        if not np.array_equal(matrix, hd._matrix(parent["checks"][side], parent["n"])):
            raise ValueError("The recovered CSS parent differs from its published board entry.")


def verify_archived_witnesses(doc, evidence):
    """Check every retained audit witness with the trusted Pauli predicate."""
    a_matrix, b_matrix = hd.stabilizer_matrices(doc)
    audits = list(evidence["searches"])
    audits.extend(
        {"lightest_found": section["weight"], "witness": section["witness"]}
        for section in evidence["section_audit"]["sections"].values()
    )
    for audit in audits:
        witness = audit["witness"]
        vector = np.zeros(2 * doc["n"], dtype=np.int8)
        vector[witness["X"]] = 1
        vector[np.asarray(witness["Z"], dtype=int) + doc["n"]] = 1
        weight = int(hd.pauli_weight_rows(vector[None, :], doc["n"])[0])
        if weight != audit["lightest_found"] or not hd.valid_pauli_logical(vector, a_matrix, b_matrix):
            raise ValueError("An archived search witness failed trusted verification.")
    return len(audits)


def main():
    """Reconstruct checks, verify witnesses, and report the trusted fingerprint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="?", type=Path, default=ROOT / "codes/96-10-12.json")
    parser.add_argument("--output", type=Path, help="Optionally save the reconstructed submission.")
    args = parser.parse_args()
    original = json.loads(args.candidate.read_text())
    if (original.get("code_type"), original.get("n"), original.get("k"), original["distance"]["d"]) != (
        "stabilizer",
        96,
        10,
        12,
    ):
        raise ValueError("Expected the [[96,10,<=12]] stabilizer candidate.")
    rebuilt = copy.deepcopy(original)
    rebuilt["checks"] = checks()
    if rebuilt["checks"] != original["checks"]:
        raise ValueError("Regenerated checks do not exactly match the candidate.")
    if rebuilt["distance"] != original["distance"]:
        raise ValueError("The supplied distance witness was changed.")

    evidence = json.loads((HERE / "evidence.json").read_text())
    if digest(rebuilt["checks"]) != evidence["checks_sha256"]:
        raise ValueError("Checks differ from the historical audit record.")
    if digest(rebuilt["distance"]) != evidence["distance_sha256"]:
        raise ValueError("Distance claim or witness differs from the historical audit record.")
    report = verify(rebuilt, refute=False)
    if not report["ok"]:
        failures = [row for row in report["checks"] if not row["ok"]]
        raise ValueError(json.dumps({"verification_failed": failures}, indent=2))
    if report["fingerprint"] != evidence["historical_gate"]["fingerprint"]:
        raise ValueError("Fingerprint differs from the historical gate record.")
    verify_parent(rebuilt)
    witness_count = verify_archived_witnesses(rebuilt, evidence)
    if args.output:
        errors = save_submission(rebuilt, str(args.output))
        if errors:
            raise ValueError(json.dumps({"schema_errors": errors}, indent=2))
    print(
        json.dumps(
            {
                "checks_match_exactly": True,
                "supplied_witness_retained": True,
                "published_parent_matrices_match_exactly": True,
                "trusted_structural_verification_passed": True,
                "archived_witnesses_verified": witness_count,
                "random_search_or_board_scan_performed": False,
                "fingerprint": report["fingerprint"],
                "computed": {
                    key: report["computed"][key]
                    for key in ("rank_S", "k", "max_check_weight", "weight_class", "locality_class")
                },
            },
            indent=2,
        )
    )
    print(f"fingerprint={report['fingerprint']}")


if __name__ == "__main__":
    main()
