"""Reconstruct the t=19 member of the published cyclic toric family.

Reference: Kovalev, Dumer, Pryadko, Phys. Rev. A 84, 062319 (2011),
Example 11. Construction is deterministic; verification uses the unchanged
challenge verifier. This script performs no distance search or certification.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "verify"))

import qldpc_verify


def digest(value):
    """Hash a JSON value without depending on indentation or key order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build():
    """Return the generators and explicit logical for t=19."""
    t = 19
    d = 2 * t + 1
    n = t * t + (t + 1) * (t + 1)
    checks = {"S": [{"X": sorted(((i + t) % n, (i + t + 1) % n)), "Z": sorted((i, (i + d) % n))} for i in range(n)]}
    witness = {"X": [t], "Z": list(range(d))}

    # This invertible affine relabeling takes every generator to a cyclic
    # shift of the paper's Z X I^(2t-1) X Z seed.
    assert math.gcd(2 * t, n) == 1
    for i, row in enumerate(checks["S"]):
        mapped = {side: sorted((2 * t * q + 2 * t + 2) % n for q in row[side]) for side in ("X", "Z")}
        shift = 2 * t * i % n
        expected = {
            "X": sorted(((shift + 1) % n, (shift + d) % n)),
            "Z": sorted((shift, (shift + d + 1) % n)),
        }
        assert mapped == expected
    return checks, witness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=ROOT / "codes/761-1-39.json")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    candidate_bytes = args.candidate.read_bytes()
    doc = json.loads(candidate_bytes)
    checks, witness = build()
    assert doc["checks"] == checks
    assert doc["distance"]["P"]["witness"] == witness
    assert (doc["n"], doc["k"], doc["distance"]["d"]) == (761, 1, 39)
    assert doc["distance"]["P"]["confidence"] == "upper_bound"
    assert doc["provenance"]["novelty"] == "known_parameters"
    assert "@mrvee-qC-bee" in doc["provenance"]["authors"]
    verified = qldpc_verify.verify(doc, refute=False)
    assert verified["ok"], verified
    gate_path = Path(__file__).with_name("candidate-gate.json")
    gate_bytes = gate_path.read_bytes()
    gate = json.loads(gate_bytes)
    assert gate["passed"]
    assert verified["fingerprint"] == gate["candidate"]["fingerprint"]
    assert verified["signature"]["hash"] == gate["candidate"]["signature"]
    validator_hash = hashlib.sha256((ROOT / "verify/validate_candidate.py").read_bytes()).hexdigest()
    assert validator_hash == gate["validator"]["source_sha256"]
    fresh_bytes = Path(__file__).with_name("prepublication-verification.json").read_bytes()
    fresh = json.loads(fresh_bytes)
    assert fresh["ok"]
    assert fresh["fingerprint"] == verified["fingerprint"]
    assert fresh["signature"]["hash"] == verified["signature"]["hash"]
    fresh_refutation = next(row for row in fresh["checks"] if row["check"] == "distance_not_refuted")
    assert fresh_refutation["ok"]
    summary = {
        "parameters": [doc["n"], doc["k"], doc["distance"]["d"]],
        "checks_reconstructed_exactly": True,
        "witness_reconstructed_exactly": True,
        "affine_relabeling_matches_published_generators": True,
        "trusted_structure_and_witness_verification_passed": True,
        "random_refutation_rerun": False,
        "exact_distance_certificate_claimed": False,
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "checks_sha256": digest(checks),
        "witness_sha256": digest(witness),
        "fingerprint": verified["fingerprint"],
        "signature": verified["signature"]["hash"],
        "prior_full_gate_sha256": hashlib.sha256(gate_bytes).hexdigest(),
        "prepublication_verification_sha256": hashlib.sha256(fresh_bytes).hexdigest(),
        "prepublication_refutation_passed": True,
        "prepublication_refutation_detail": fresh_refutation["detail"],
        "refutation_budget_note": (
            "The trusted default refutation uses an 8000-trial ceiling and a 10-second time cap. "
            "Its receipt reports the configured budget, not an actual completed-trial count."
        ),
        "validator_source_sha256": validator_hash,
        "receipt_bridge": (
            "The retained gate records the same code fingerprint, signature, n, k and d. "
            "This run freshly verifies the current JSON and its witness. The prior receipt "
            "does not record a whole-document digest, so original metadata-byte identity is not asserted."
        ),
        "computed": verified["computed"],
    }
    if args.report:
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
