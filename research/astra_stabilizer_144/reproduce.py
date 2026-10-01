"""Rebuild the [[144,10,<=16]] inversion fold and check every archived witness.

An optional positional candidate selects the submission JSON to reproduce.
The default is codes/144-10-16-b.json, leaving the earlier CSS entry intact.
No random search is run.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / "verify/qldpc_verify.py").exists())
sys.path[:0] = [str(ROOT / "verify"), str(ROOT / "research/kit")]
import heuristic_distance as hd
from qldpc_verify import verify
from submit import save_submission


def digest(value):
    """Hash a JSON value independently of formatting."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_checks():
    """Build the explicit opposite-translation generators on Z_12 x Z_12."""
    a_terms = [(5, 3), (4, 8), (2, 4), (7, 6)]
    b_terms = [(10, 1), (8, 1), (11, 4), (5, 5)]
    return {
        "S": [
            {
                "X": sorted(12 * ((i + a) % 12) + ((j + b) % 12) for a, b in a_terms),
                "Z": sorted(12 * ((-i - a) % 12) + ((-j - b) % 12) for a, b in b_terms),
            }
            for i in range(12)
            for j in range(12)
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", nargs="?", type=Path, default=ROOT / "codes/144-10-16-b.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = json.loads(args.candidate.read_text())
    rebuilt = copy.deepcopy(source)
    rebuilt["checks"] = build_checks()
    if rebuilt["checks"] != source["checks"]:
        raise ValueError("The explicit formula differs from the submitted checks.")
    evidence = json.loads((HERE / "evidence.json").read_text())
    if digest(rebuilt["checks"]) != evidence["matrix_sha256"]:
        raise ValueError("The generator matrix differs from the audit record.")
    if digest(rebuilt["distance"]) != evidence["distance_sha256"]:
        raise ValueError("The submitted witness differs from the audit record.")
    report = verify(rebuilt, refute=False)
    if not report["ok"] or report["fingerprint"] != evidence["fingerprint"]:
        raise ValueError("Trusted structural verification or fingerprint comparison failed.")
    a_matrix, b_matrix = hd.stabilizer_matrices(rebuilt)
    identity = np.eye(144, dtype=np.int8)
    hx = np.vstack(
        [
            np.concatenate([a_matrix, np.zeros_like(a_matrix), b_matrix], axis=1),
            np.concatenate([identity] * 3, axis=1),
        ]
    )
    hz = np.concatenate([b_matrix, a_matrix ^ b_matrix, a_matrix], axis=1)
    embedded_checked = 0
    for record in evidence["witnesses"]:
        claim = record["distance"]
        witness = claim["P"]["witness"]
        vector = np.zeros(288, dtype=np.int8)
        vector[witness["X"]] = 1
        vector[np.array(witness["Z"], dtype=int) + 144] = 1
        if not hd.valid_pauli_logical(vector, a_matrix, b_matrix):
            raise ValueError("An archived witness is not a nontrivial logical.")
        if int(hd.pauli_weight_rows(vector[None, :], 144)[0]) != claim["d"]:
            raise ValueError("An archived witness has an incorrect weight.")
        for origin in record["origins"]:
            if "embedding_weight" not in origin:
                continue
            lifted = np.zeros(432, dtype=np.int8)
            lifted[origin["support"]] = 1
            side = origin["side"]
            if lifted.sum() != origin["embedding_weight"] or side not in ("X", "Z"):
                raise ValueError("An embedded witness has an incorrect side or weight.")
            if not hd._valid_logical(lifted, hz if side == "X" else hx, hx if side == "X" else hz):
                raise ValueError("An embedded witness is not a nontrivial CSS logical.")
            u, v, w = lifted[:144], lifted[144:288], lifted[288:]
            mapped = np.concatenate([u ^ v, v ^ w] if side == "X" else [w, u])
            if not np.array_equal(mapped, vector):
                raise ValueError("The embedded witness does not map to the archived Pauli logical.")
            embedded_checked += 1
    # The checked even-CSS comparison supports the mathematical parity argument.
    comparison = json.loads((ROOT / "codes/144-10-16.json").read_text())
    if comparison.get("code_type") != "CSS" or not verify(comparison, refute=False)["ok"]:
        raise ValueError("The comparison entry is not a verified CSS code.")
    if any(len(row) % 2 for side in ("X", "Z") for row in comparison["checks"][side]):
        raise ValueError("The comparison entry no longer has only even generators.")
    odd_row = rebuilt["checks"]["S"][14]
    if len(set(odd_row["X"]) | set(odd_row["Z"])) != 7:
        raise ValueError("The supplied odd-weight stabilizer invariant changed.")
    if args.output:
        errors = save_submission(rebuilt, str(args.output))
        if errors:
            raise ValueError(errors)
    print(
        json.dumps(
            {
                "checks_match_exactly": True,
                "trusted_structural_verification_passed": True,
                "archived_witnesses_verified": len(evidence["witnesses"]),
                "embedded_witnesses_and_maps_verified": embedded_checked,
                "lowest_archived_weight": min(record["distance"]["d"] for record in evidence["witnesses"]),
                "odd_stabilizer_weight": 7,
                "comparison_css_generators_all_even": True,
                "rank_S": report["computed"]["rank_S"],
                "k": report["computed"]["k"],
                "fingerprint": report["fingerprint"],
            },
            indent=2,
        )
    )
    print(f"fingerprint={report['fingerprint']}")


if __name__ == "__main__":
    main()
