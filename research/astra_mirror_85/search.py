"""Reproduce selected published mirror codes using trusted distance engines.

Source: Khesin and Lu, arXiv:2603.05496v1, Section 5 table.
Generators are Z on A+g, X on B-g in the lexicographically indexed group.
Every returned witness, including tightened bounds, is saved immediately.
"""

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "verify"), str(ROOT / "research/kit"), str(ROOT / "cli")]
import gf2
import heuristic_distance as hd
import qldpc
from coordination import staging_dir, unique_path
from submit import save_submission

RECIPES = {
    "85-8-w6": ([5, 17], [(0, 0), (0, 1), (1, 9)], [(0, 0), (0, 4), (1, 2)]),
    "91-4-w6": ([7, 13], [(0, 0), (0, 1)], [(0, 0), (1, 0), (2, 3), (5, 4)]),
    "60-4-w6": ([2, 2, 3, 5], [(0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 2, 1)], [(0, 1, 0, 0), (1, 0, 1, 1), (1, 1, 2, 2)]),
    "99-4-w7": ([3, 3, 11], [(0, 0, 0), (0, 1, 1), (0, 2, 3)], [(0, 0, 0), (0, 1, 5), (1, 0, 4), (2, 1, 8)]),
    "99-6-w7": ([9, 11], [(0, 0), (3, 1), (6, 3)], [(0, 0), (0, 2), (1, 1), (1, 4)]),
    "93-5-w7": ([3, 31], [(0, 0), (0, 1), (1, 12)], [(0, 0), (0, 4), (0, 23), (1, 26)]),
    "75-4-w7": ([3, 25], [(0, 0), (0, 1), (1, 8)], [(0, 0), (0, 2), (0, 16), (2, 11)]),
}
OUT = Path(__file__).resolve().parent


def matrices(moduli, subset_a, subset_b):
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


def persist(doc, tag):
    dest = unique_path(
        str(
            Path(staging_dir("astra-round2-mirror-20261001"))
            / f"{doc['n']}-{doc['k']}-{doc['distance']['d']}-{tag}.json"
        ),
        doc,
    )
    errors = save_submission(doc, dest)
    if errors:
        raise RuntimeError(errors)
    return str(Path(dest).resolve())


def run(tag, trials, fast_trials, seed, axes):
    if fast_trials and hd._fast is None:
        raise RuntimeError("Build the repository's gf2_fast accelerator before repeating this search.")
    recipe = RECIPES[tag]
    x, z = matrices(*recipe)
    args = SimpleNamespace(
        trials=trials,
        fast_trials=fast_trials,
        seed=seed,
        authors=["@mrvee-qC-bee", "Andrey Boris Khesin", "Jonathan Z. Lu"],
        construction=(
            "Published abelian mirror code, arXiv:2603.05496v1 Section 5. "
            f"G has cyclic factors {recipe[0]}, A={recipe[1]}, B={recipe[2]}. "
            "Lexicographic qubits g; each generator has Z on A+g and X on B-g."
        ),
        date="2026-10-01",
        model="GPT-6 Astra",
        notes="Literature reproduction. Numerical distance is a witnessed upper bound; no new-construction claim.",
        name=None,
        family="other",
        _coords=None,
    )
    doc = qldpc.build_stabilizer_submission(x, z, args)
    doc["provenance"]["novelty"] = "known_parameters"
    doc["provenance"]["references"] = ["https://arxiv.org/abs/2603.05496"]
    path = persist(doc, tag + "-initial")
    results = {
        "recipe": recipe,
        "initial_candidate": path,
        "seed": seed,
        "trials": trials,
        "fast_trials": fast_trials,
        "axes": {},
    }
    (OUT / f"{tag}-evidence.json").write_text(json.dumps(results, indent=2) + "\n")
    n = doc["n"]
    if axes:
        for label, h, op in [("Y", x ^ z, x), ("X", z, x), ("Z", x, z)]:
            row = (gf2.kernel_basis(h.T) @ op) % 2
            weight, v = hd.ris_min_logical(row, h, trials=axes, seed=seed + 100, pair_depth=20)
            if v is None:
                continue
            full = np.concatenate(
                [v if label != "Z" else np.zeros(n, dtype=np.int8), v if label != "X" else np.zeros(n, dtype=np.int8)]
            )
            assert hd.valid_pauli_logical(full, x, z)
            witness = hd.pauli_witness(full, n)
            axisdoc = copy.deepcopy(doc)
            axisdoc["distance"] = {
                "d": int(weight),
                "P": {"value": int(weight), "confidence": "upper_bound", "witness": witness},
            }
            axisdoc["name"] = f"[[{n},{doc['k']},{weight}]]"
            axispath = persist(axisdoc, tag + "-axis-" + label)
            results["axes"][label] = {
                "weight": int(weight),
                "witness": witness,
                "candidate": axispath,
                "trials": axes,
                "seed": seed + 100,
                "pair_depth": 20,
            }
            if weight < doc["distance"]["d"]:
                doc, path = axisdoc, axispath
            (OUT / f"{tag}-evidence.json").write_text(json.dumps(results, indent=2) + "\n")
            print(tag, label, weight, flush=True)
    results["best_candidate"] = path
    results["score"] = doc["k"] * doc["distance"]["d"] ** 2 / n
    (OUT / f"{tag}-evidence.json").write_text(json.dumps(results, indent=2) + "\n")
    print("FINAL", tag, path, results["score"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tags", nargs="*", default=list(RECIPES))
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--fast-trials", type=int, default=20000)
    p.add_argument("--axes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=26100111)
    a = p.parse_args()
    for tag in a.tags:
        run(tag, a.trials, a.fast_trials, a.seed, a.axes)
