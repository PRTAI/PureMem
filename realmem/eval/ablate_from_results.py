"""Recover intermediate-stage rankings from an existing three-stage run.

A finished ``three_stage_*`` result file records, per ranked item, the position
it held *before* Stage 3 (``stage2_rank``) and how much of its score came from
Stage 1 tags (``stage1_bonus``). That is enough to reconstruct what the pipeline
would have returned had it stopped earlier — no API calls, no re-run.

So the attribution question ("how much of the gain is Stage 1 vs Stage 3?")
is answerable for free from artefacts you already have:

    simple_embedding      cosine only                     (separate arm)
    stage2_reconstructed  Stage 1 + Stage 2, no judge     (recovered here)
    three_stage_rerank    Stage 1 + Stage 2 + Stage 3     (as run)

WARNING — this reconstruction is an APPROXIMATION, not a substitute for running
the arm. Prefer a real ``stage2_only`` run whenever you have one.

Why it cannot be exact: ``retrieve()`` returns ``ranked[:RETRIEVE_K]`` (20) out
of a Stage-2 pool of ``POOL_SIZE`` (40). Stage 3 can promote a candidate ranked
21-40 by Stage 2 into that top 20, displacing one Stage 2 had ranked higher. So
the surviving 20 are "the items Stage 3 chose", and re-sorting them by
stage2_rank yields those items in Stage-2 order — NOT Stage 2's actual top 20.

An earlier version of this docstring claimed the recovery was complete for
gate_mode='rerank' because that mode drops nothing. That reasoning missed the
RETRIEVE_K truncation and was wrong. Measured against a real stage2_only run:
per-persona metric differences up to 0.018, with ~5% of queries affected.

The bias is systematic, not random: recall@k for k >= RETRIEVE_K is inflated to
exactly the full arm's value, because both are scored over the same 20 items.
That makes the reconstruction useless for the one question it looks best at —
"does Stage 3 add new candidates?" — since it forces the answer to zero by
construction.

Use ``--validate-against stage2_only`` to quantify the gap on your own data.

    python -m eval.ablate_from_results --personas Lin_Wanyu
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import DATASET_DIR, RETRIEVAL_RESULT_DIR, list_personas
from eval.metrics import compute_auto_metrics, per_query_scores, DEFAULT_KS
from eval.stats import summarize_comparison
from eval import schema


def reconstruct_stage2(results: dict) -> dict:
    """Re-sort each query's items by the rank Stage 2 gave them."""
    out = {}
    for q, rec in results.items():
        items = [i for i in rec.get("ranked_items", []) if i.get("stage2_rank") is not None]
        if len(items) != len(rec.get("ranked_items", [])):
            # Missing the field entirely -> produced before it was recorded.
            return {}
        ordered = sorted(items, key=lambda i: i["stage2_rank"])
        out[q] = {**rec, "ranked_items": [{**i, "rank": n}
                                          for n, i in enumerate(ordered, 1)]}
    return out


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate(personas, args, rows):
    """Check the reconstruction against an arm that was actually executed.

    The reconstruction reorders a finished run by its recorded stage2_rank. If
    that is faithful, it must reproduce a real Stage-1+2 run exactly — same
    ranking, same metrics. Any gap means the attribution in the results doc is
    measuring something other than what it claims.
    """
    import numpy as np
    print("\n" + "=" * 96)
    print(f"VALIDATION: reconstruction vs really-executed '{args.validate_against}'")
    print("=" * 96)

    metrics = [f"{m}@{k}" for m in ("recall_any", "recall_all", "ndcg") for k in DEFAULT_KS]
    worst = 0.0
    order_mismatch = missing = 0
    print(f"{'persona':16s}{'max |metric diff|':>20}{'ranking identical':>20}{'n':>7}")
    print("-" * 96)

    for persona in personas:
        pdir = os.path.join(args.retrieval_result_dir, persona)
        real_path = os.path.join(pdir, f"{args.validate_against}_retrieval_results.json")
        full_path = os.path.join(pdir, f"{args.arm}_retrieval_results.json")
        ds = os.path.join(args.dataset_dir, f"{persona}_dialogues_256k.json")
        if not (os.path.exists(real_path) and os.path.exists(full_path)):
            missing += 1
            print(f"{persona:16s}{'(missing input)':>20}")
            continue

        with open(ds, "r", encoding="utf-8") as f:
            data = json.load(f)
        gold, corpus = schema.extract_retrieval_gold(data), schema.corpus_ids(data)

        real = load(real_path)
        recon = reconstruct_stage2(load(full_path))
        if not recon:
            missing += 1
            continue

        a = compute_auto_metrics(real, gold, corpus)
        b = compute_auto_metrics(recon, gold, corpus)
        diff = max(abs(a[m] - b[m]) for m in metrics)
        worst = max(worst, diff)

        same = 0
        shared = set(real) & set(recon)
        for q in shared:
            ra = [i["chunk_id"] for i in real[q]["ranked_items"]]
            rb = [i["chunk_id"] for i in recon[q]["ranked_items"]]
            if ra == rb:
                same += 1
            else:
                order_mismatch += 1
        print(f"{persona:16s}{diff:>20.6f}{f'{same}/{len(shared)}':>20}{len(shared):>7}")

    print("-" * 96)
    print(f"worst metric difference across all personas: {worst:.6f}")
    if missing:
        print(f"{missing} persona(s) could not be checked")
    if worst < 1e-9 and order_mismatch == 0:
        print("EXACT: the offline reconstruction reproduces a real Stage-1+2 run.")
        return 0
    if worst < 1e-9:
        print(f"Metrics identical, but {order_mismatch} queries differ in ranking "
              f"order — ties broken differently. Attribution numbers stand.")
        return 0
    print(f"MISMATCH: reconstruction differs from the executed arm by up to "
          f"{worst:.6f}.\nThe stage attribution in the results doc is NOT measuring "
          f"a genuine Stage-1+2 ranking; investigate before citing it.")
    return 1


def main():
    p = argparse.ArgumentParser(description="Free Stage-1/Stage-3 attribution")
    p.add_argument("--personas", default=None)
    p.add_argument("--all-personas", action="store_true")
    p.add_argument("--arm", default="three_stage_rerank",
                   help="Which three-stage arm to reconstruct from")
    p.add_argument("--baseline", default="simple_embedding")
    p.add_argument("--retrieval-result-dir", default=RETRIEVAL_RESULT_DIR)
    p.add_argument("--dataset-dir", default=DATASET_DIR)
    p.add_argument("--validate-against", default=None, metavar="ARM",
                   help="Compare the reconstruction with a really-executed arm "
                        "(e.g. stage2_only). They must match exactly.")
    args = p.parse_args()

    personas = (list_personas(args.dataset_dir) if args.all_personas
                else [x.strip() for x in (args.personas or "").split(",") if x.strip()])
    if not personas:
        p.error("Specify --personas or --all-personas")

    if "gated" in args.arm:
        print("Refusing to reconstruct from a gated arm: candidates that failed "
              "the vote were dropped, so the Stage-2 ordering cannot be recovered "
              "in full. Use --arm three_stage_rerank.")
        return 1

    metrics = [f"{m}@{k}" for m in ("recall_any", "recall_all", "ndcg") for k in DEFAULT_KS]
    rows, scores = {}, {}
    promoted_total = promoted_queries = items_total = 0

    for persona in personas:
        ds = os.path.join(args.dataset_dir, f"{persona}_dialogues_256k.json")
        pdir = os.path.join(args.retrieval_result_dir, persona)
        full_path = os.path.join(pdir, f"{args.arm}_retrieval_results.json")
        base_path = os.path.join(pdir, f"{args.baseline}_retrieval_results.json")
        if not (os.path.exists(ds) and os.path.exists(full_path) and os.path.exists(base_path)):
            print(f"skip {persona}: missing inputs")
            continue

        with open(ds, "r", encoding="utf-8") as f:
            data = json.load(f)
        gold = schema.extract_retrieval_gold(data)
        corpus = schema.corpus_ids(data)

        full = load(full_path)

        # Quantify how far the reconstruction can be off: any item whose
        # stage2_rank exceeds the returned depth was promoted across the
        # truncation boundary, displacing one Stage 2 ranked higher.
        for rec in full.values():
            items = rec.get("ranked_items", [])
            depth = len(items)
            items_total += depth
            n_promoted = sum(1 for i in items
                             if (i.get("stage2_rank") or 0) > depth)
            promoted_total += n_promoted
            if n_promoted:
                promoted_queries += 1

        stage2 = reconstruct_stage2(full)
        if not stage2:
            print(f"skip {persona}: results predate the stage2_rank field; re-run "
                  f"to enable this attribution")
            continue

        variants = {
            args.baseline: load(base_path),
            "stage2_reconstructed": stage2,
            args.arm: full,
        }
        for name, res in variants.items():
            rows.setdefault(name, {})[persona] = compute_auto_metrics(res, gold, corpus)
            for metric, (m, k) in {"recall_any@5": ("recall_any", 5),
                                   "recall_any@10": ("recall_any", 10),
                                   "recall_all@10": ("recall_all", 10),
                                   "ndcg@10": ("ndcg", 10)}.items():
                scores.setdefault(name, {}).setdefault(metric, {}).update(
                    {f"{persona}\x00{q}": v for q, v in
                     per_query_scores(res, gold, corpus, k=k, metric=m).items()})

    if not rows:
        print("nothing to report")
        return 1

    if args.validate_against:
        rc = _validate(personas, args, rows)
        if rc:
            return rc

    order = [args.baseline, "stage2_reconstructed", args.arm]
    import numpy as np

    print("\n" + "=" * 104)
    print("STAGE ATTRIBUTION  (reconstructed offline from stage2_rank; no API calls)")
    print("=" * 104)
    if promoted_total:
        print(f"  APPROXIMATION: {promoted_total} of {items_total} returned items "
              f"({promoted_total/items_total:.2%}) were promoted by Stage 3 from "
              f"beyond the\n  returned depth, affecting {promoted_queries} queries. "
              f"For those, the reconstruction is not\n  Stage 2's real top-k. "
              f"recall@k at k >= depth is inflated to the full arm's value by\n"
              f"  construction — do not read 'Stage 3 adds no new candidates' off "
              f"this table.\n  Run a real stage2_only arm and use "
              f"--validate-against to quantify.\n")
    print(f"{'variant':<24}" + "".join(f"{m.replace('recall_', ''):>9}" for m in metrics))
    print("-" * 104)
    for name in order:
        if name not in rows:
            continue
        vals = [np.mean([rows[name][p][m] for p in rows[name]]) for m in metrics]
        print(f"{name:<24}" + "".join(f"{v:>9.4f}" for v in vals))

    print("\n" + "-" * 104)
    print("attribution (delta on each step)")
    print("-" * 104)
    print(f"{'step':<34}" + "".join(f"{m.replace('recall_', ''):>9}" for m in metrics))
    for a, b, label in ((order[1], order[0], "Stage 1 (tag soft weighting)"),
                        (order[2], order[1], "Stage 3 (LLM rerank)")):
        if a not in rows or b not in rows:
            continue
        d = [np.mean([rows[a][p][m] for p in rows[a]])
             - np.mean([rows[b][p][m] for p in rows[b]]) for m in metrics]
        print(f"{label:<34}" + "".join(f"{v:>+9.4f}" for v in d))

    print("\n" + "=" * 104)
    print("PAIRED TESTS on each step")
    print("=" * 104)
    print(f"{'step':<34}{'metric':<15}{'delta':>9}{'p / 95% CI':>26}{'discordant':>12}")
    print("-" * 104)
    for a, b, label in ((order[1], order[0], "Stage 1 vs cosine"),
                        (order[2], order[1], "Stage 3 vs Stage 1+2"),
                        (order[2], order[0], "full vs cosine")):
        if a not in scores or b not in scores:
            continue
        for metric in ("recall_any@5", "recall_any@10", "recall_all@10", "ndcg@10"):
            r = summarize_comparison(a, b, scores[a][metric], scores[b][metric], metric)
            if not r.get("n"):
                continue
            detail = (f"p={r['p_value']:.4g}" if r["test"] == "mcnemar_exact"
                      else f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
            disc = str(r["discordant"]) if r["test"] == "mcnemar_exact" else "-"
            print(f"{label:<34}{metric:<15}{r['delta']:>+9.4f}{detail:>26}{disc:>12}")
            if "warning" in r:
                print(f"{'':<34}  ! {r['warning']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
