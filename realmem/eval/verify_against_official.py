"""Re-score our retrieval output with the vendored official scorer and diff.

This is the independent check on the whole harness: if our numbers and
``compute_auto_metrics_for_realmem.py``'s numbers agree on the same files, then
whatever else is wrong, the metric is not.

Run it after any real evaluation:

    python -m eval.verify_against_official --all-personas

NumPy 2.0 removed ``np.asfarray``, which the vendored scorer still calls, so a
shim is injected at import time. The vendored file itself is left untouched —
it is the reference definition, and editing it would destroy the only
independent oracle available. If you run the official script directly and see
``AttributeError: module 'numpy' has no attribute 'asfarray'``, that is this
same incompatibility, not a bug in your data.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

from eval import metrics as ours
from eval import schema
from eval.config import (DATASET_DIR, RETRIEVAL_RESULT_DIR, DEFAULT_ARMS,
                         list_personas)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_auto_metrics_for_realmem as official  # noqa: E402

TOL = 1e-9


def official_scores(retrieval_results: dict, data: dict) -> dict:
    """The vendored main() loop, run in-process so we score identical inputs."""
    query_to_gold, corpus = official.get_ground_truth(data)
    corpus = sorted(list(corpus))
    idx = {cid: i for i, cid in enumerate(corpus)}

    acc = {f"{m}@{k}": [] for k in (5, 10, 20)
           for m in ("recall_all", "recall_any", "ndcg_any")}
    n = 0
    for query_text, obj in retrieval_results.items():
        key = query_text if query_text in query_to_gold else query_text.strip()
        if key not in query_to_gold:
            continue
        rankings = []
        for item in obj.get("ranked_items", []):
            if isinstance(item, dict) and item.get("res_type") == "chunk":
                rid = item.get("chunk_id")
                if rid in idx:
                    rankings.append(idx[rid])
        for k in (5, 10, 20):
            r_any, r_all, nd = official.evaluate_retrieval(
                rankings, query_to_gold[key], corpus, k=k)
            acc[f"recall_any@{k}"].append(r_any)
            acc[f"recall_all@{k}"].append(r_all)
            acc[f"ndcg_any@{k}"].append(nd)
        n += 1

    out = {k: round(float(np.mean(v)), 4) for k, v in acc.items() if v}
    out["n_evaluated"] = n
    return out


def main():
    p = argparse.ArgumentParser(description="Diff our metrics against the official scorer")
    p.add_argument("--personas", default=None)
    p.add_argument("--all-personas", action="store_true")
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    p.add_argument("--retrieval-result-dir", default=RETRIEVAL_RESULT_DIR)
    p.add_argument("--dataset-dir", default=DATASET_DIR)
    args = p.parse_args()

    if args.all_personas:
        personas = list_personas(args.dataset_dir)
    elif args.personas:
        personas = [x.strip() for x in args.personas.split(",") if x.strip()]
    else:
        p.error("Specify --personas or --all-personas")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    print(f"{'persona':16s} {'arm':20s} {'metric':16s} {'ours':>9s} {'official':>9s} {'diff':>9s}")
    print("-" * 84)

    checked = mismatches = missing = 0
    for persona in personas:
        ds = os.path.join(args.dataset_dir, f"{persona}_dialogues_256k.json")
        if not os.path.exists(ds):
            continue
        with open(ds, "r", encoding="utf-8") as f:
            data = json.load(f)
        gold = schema.extract_retrieval_gold(data)
        corpus = schema.corpus_ids(data)

        for arm in arms:
            path = os.path.join(args.retrieval_result_dir, persona,
                                f"{arm}_retrieval_results.json")
            if not os.path.exists(path):
                missing += 1
                continue
            with open(path, "r", encoding="utf-8") as f:
                results = json.load(f)

            off = official_scores(results, data)
            us = ours.compute_auto_metrics(results, gold, corpus)

            for k in (5, 10, 20):
                for o_key, u_key in ((f"recall_any@{k}", f"recall_any@{k}"),
                                     (f"recall_all@{k}", f"recall_all@{k}"),
                                     (f"ndcg_any@{k}", f"ndcg@{k}")):
                    o, u = off.get(o_key, 0.0), us.get(u_key, 0.0)
                    checked += 1
                    if abs(o - u) > TOL:
                        mismatches += 1
                        print(f"{persona:16s} {arm:20s} {u_key:16s} {u:9.4f} {o:9.4f} "
                              f"{u - o:+9.4f}  <-- MISMATCH")
            if off["n_evaluated"] != us["n_evaluated"]:
                mismatches += 1
                print(f"{persona:16s} {arm:20s} {'n_evaluated':16s} "
                      f"{us['n_evaluated']:9d} {off['n_evaluated']:9d}  <-- MISMATCH")
            else:
                print(f"{persona:16s} {arm:20s} {'all 9 metrics':16s} "
                      f"{'match':>9s} {'match':>9s} {'0':>9s}   n={us['n_evaluated']}")

    print("-" * 84)
    print(f"compared {checked} metric values across {len(personas)} personas")
    if missing:
        print(f"{missing} arm/persona combinations had no results file (skipped)")
    if mismatches:
        print(f"FAILED: {mismatches} mismatches — our numbers are NOT the official numbers")
        return 1
    if checked == 0:
        print("nothing was compared; run eval/run_eval.py first")
        return 1
    print("PASS: every value matches the vendored official scorer exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
