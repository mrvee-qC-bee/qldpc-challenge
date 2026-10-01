"""Fresh-seed trusted RIS confirmation; preserve every returned witness."""

import argparse
import json
from pathlib import Path

from amplify import ROOT, dense
from submit import save_submission
from surrogate import distance_rand_witness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--trials", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=26010007)
    args = ap.parse_args()
    path = Path(args.candidate).resolve()
    doc = json.loads(path.read_text())
    x, z = dense(doc, "X"), dense(doc, "Z")
    result = distance_rand_witness(x, z, trials=args.trials, seed=args.seed, backend="fast", threads=1, pair_depth=20)
    assert not result.rejected, result.rejected
    if result.side in ("X", "Z"):
        witness_doc = json.loads(json.dumps(doc))
        witness_doc["distance"][result.side].update(value=result.weight, witness=result.support)
        witness_doc["distance"]["d"] = min(witness_doc["distance"][s]["value"] for s in ("X", "Z"))
        target = path.with_name(path.stem + f"-deep-{args.seed}.json")
        assert not save_submission(witness_doc, str(target))
    report = {
        "candidate": str(path.relative_to(ROOT)),
        "trials": args.trials,
        "seed": args.seed,
        "pair_depth": 20,
        "result": result._asdict(),
        "refuted": result.weight < doc["distance"]["d"],
    }
    (ROOT / "research/astra_round2_css" / f"deep-{doc['n']}-{args.seed}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
