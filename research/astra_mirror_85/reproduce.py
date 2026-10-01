"""Reconstruct the published mirror code and check every archived witness.

Run from any directory with the challenge's documented dependencies installed.
This uses the trusted verifier for structure and Pauli logical predicates.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "verify"))
import heuristic_distance as hd
import qldpc_verify


def matrices(recipe):
    moduli, subset_a, subset_b = recipe
    group = list(itertools.product(*(range(m) for m in moduli)))
    index = {g: i for i, g in enumerate(group)}
    x = np.zeros((len(group), len(group)), dtype=np.int8)
    z = np.zeros_like(x)
    for i, g in enumerate(group):
        for a in subset_a:
            z[i, index[tuple((ai + gi) % m for ai, gi, m in zip(a, g, moduli))]] = 1
        for b in subset_b:
            x[i, index[tuple((bi - gi) % m for bi, gi, m in zip(b, g, moduli))]] = 1
    return x, z


def check_witness(x, z, record):
    n = x.shape[1]
    v = np.zeros(2 * n, dtype=np.int8)
    witness = record["witness"]
    v[witness["X"]] = 1
    v[np.asarray(witness["Z"], dtype=int) + n] = 1
    assert len(set(witness["X"]) | set(witness["Z"])) == record["weight"]
    assert hd.valid_pauli_logical(v, x, z)


def main():
    doc = json.loads((ROOT / "codes/85-8-9.json").read_text())
    evidence = json.loads((Path(__file__).parent / "evidence.json").read_text())
    x, z = matrices(evidence["search"]["85-8-w6"]["recipe"])
    dx, dz = hd.stabilizer_matrices(doc)
    assert np.array_equal(x, dx) and np.array_equal(z, dz)
    report = qldpc_verify.verify(doc, refute=False)
    assert report["ok"], report
    checked = 0
    for record in evidence["search"].values():
        a, b = matrices(record["recipe"])
        assert not ((a @ b.T + b @ a.T) % 2).any()
        for witness in record["witnesses"]:
            check_witness(a, b, witness)
            checked += 1
    for rung in evidence["deep_searches"]:
        side = rung["result"]["sides"]["P"]
        check_witness(x, z, {"weight": side["lightest_found"], "witness": side["witness"]})
        checked += 1
    check_witness(x, z, evidence["tripled_search"])
    checked += 1
    print(
        json.dumps(
            {
                "exact_matrix_reconstruction": True,
                "trusted_structure_and_submission_witness": report["ok"],
                "archived_witnesses_checked": checked,
                "fingerprint": report["fingerprint"],
                "official_exact_distance_certificate": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
