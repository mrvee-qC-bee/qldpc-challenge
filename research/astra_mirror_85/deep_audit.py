"""Independent trusted Pauli searches with immediate witness retention."""

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "verify"), str(ROOT / "research/kit")]
import heuristic_distance as hd
from coordination import unique_path
from submit import save_submission

parser = argparse.ArgumentParser()
parser.add_argument("candidate", type=Path)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--trials", type=int, default=20000)
parser.add_argument("--fast-trials", type=int, default=400000)
args = parser.parse_args()
if args.fast_trials and hd._fast is None:
    raise RuntimeError("Build the repository's gf2_fast accelerator before repeating this search.")
doc = json.loads(args.candidate.read_text())
doc["provenance"]["novelty"] = "known_parameters"
result = hd.estimate(doc, trials=args.trials, seed=args.seed, fast_trials=args.fast_trials, max_seconds=None)
result["candidate_source"] = str(args.candidate.resolve())
side = result["sides"]["P"]
if side["lightest_found"] is not None:
    revised = copy.deepcopy(doc)
    weight = side["lightest_found"]
    revised["distance"] = {"d": weight, "P": {"value": weight, "confidence": "upper_bound", "witness": side["witness"]}}
    revised["name"] = f"[[{doc['n']},{doc['k']},{weight}]]"
    dest = unique_path(str(args.candidate.with_name(f"{doc['n']}-{doc['k']}-{weight}-deep-{args.seed}.json")), revised)
    errors = save_submission(revised, dest)
    if errors:
        raise RuntimeError(errors)
    result["saved_candidate"] = dest
out = Path(__file__).resolve().parent / f"deep-{doc['n']}-{doc['k']}-seed{args.seed}.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
