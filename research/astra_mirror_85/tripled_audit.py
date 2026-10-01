"""Trusted native RIS on the exact three-bit Pauli-to-CSS reduction.

For S=(A|B), H_X=[A,0,B;I,I,I] and H_Z=[B,A+B,A]. The X quotient
maps by (u,v,w)->(u+v,v+w), whose minimum lift weight is Pauli weight;
the Z quotient maps by (u,u+w,w)->(w,u), with doubled Pauli weight.
Both embedded and source witnesses are checked by trusted predicates.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "verify"), str(ROOT / "research/kit")]
import heuristic_distance as hd
from coordination import staging_dir, unique_path
from submit import save_submission

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("--trials", type=int, default=400000)
p.add_argument("--seed", type=int, default=9510017)
args = p.parse_args()
if hd._fast is None:
    raise RuntimeError("Build the repository's gf2_fast accelerator before repeating this search.")
doc = json.loads(args.source.read_text())
a, b = hd.stabilizer_matrices(doc)
n = doc["n"]
identity = np.eye(n, dtype=np.int8)
hx = np.vstack([np.concatenate([a, np.zeros_like(a), b], axis=1), np.concatenate([identity] * 3, axis=1)])
hz = np.concatenate([b, a ^ b, a], axis=1)
assert not ((hx @ hz.T) % 2).any()
weight, side, support = hd._fast.distance_rand_witness(hx, hz, args.trials, args.seed, 8, 8)
assert side in ("X", "Z")
v = np.zeros(3 * n, dtype=np.int8)
v[list(support)] = 1
assert hd._valid_logical(v, hz if side == "X" else hx, hx if side == "X" else hz)
u, mid, w = v[:n], v[n : 2 * n], v[2 * n :]
pauli = np.concatenate([u ^ mid, mid ^ w] if side == "X" else [w, u])
assert hd.valid_pauli_logical(pauli, a, b)
wit = hd.pauli_witness(pauli, n)
d = len(set(wit["X"]) | set(wit["Z"]))
result = {
    "trials": args.trials,
    "seed": args.seed,
    "pair_depth": 8,
    "threads": 8,
    "embedded_side": side,
    "embedded_weight": int(weight),
    "embedded_support": list(map(int, support)),
    "weight": d,
    "witness": wit,
    "embedded_and_source_witness_valid": True,
}
doc["distance"] = {"d": d, "P": {"value": d, "confidence": "upper_bound", "witness": wit}}
doc["name"] = f"[[{n},{doc['k']},{d}]]"
dest = unique_path(
    str(
        (Path(staging_dir("astra-mirror-tripled-reproduction")) / args.source.name).with_name(
            f"{n}-{doc['k']}-{d}-tripled-{args.seed}.json"
        )
    ),
    doc,
)
errors = save_submission(doc, dest)
if errors:
    raise RuntimeError(errors)
(Path(__file__).parent / f"tripled-{n}-{args.seed}.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
