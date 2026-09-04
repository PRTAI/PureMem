"""Compare retrieval arms: auto metrics, paired tests, injection diagnostics.

LLM-judge metrics are NOT computed here — they need generated answers, and this
module only sees retrieval output. The previous version pretended otherwise: it
hardcoded ``generated_answer=""`` and then guarded the QA prompt on that string
being non-empty, so ``--llm-judge`` could never produce a QA score. Generation
and judging now live in run_qa_eval.py, which owns both halves; this module
reads back the summary that writes.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import (
    RETRIEVAL_RESULT_DIR, DATASET_DIR, MAX_METRIC_K,
    persona_dataset_path, list_personas,
)
from eval.metrics import compute_auto_metrics, per_query_scores, DEFAULT_KS
from eval.stats import summarize_comparison
from eval import schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AUTO_METRICS = [f"{m}@{k}" for m in ("recall_any", "recall_all", "ndcg") for k in DEFAULT_KS]


def _load_arm(persona_dir: str, arm: str) -> dict:
    for name in (f"{arm}_retrieval_results.json", f"DRYRUN-{arm}_retrieval_results.json"):
        path = os.path.join(persona_dir, name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _depth_report(results: dict) -> dict:
    lengths = [len(r.get("ranked_items", [])) for r in results.values()]
    if not lengths:
        return {}
    return {
        "mean_depth": round(float(np.mean(lengths)), 2),
        "empty_rate": round(sum(1 for n in lengths if n == 0) / len(lengths), 4),
        "shallow_rate": round(sum(1 for n in lengths if 0 < n < MAX_METRIC_K) / len(lengths), 4),
    }


def analyze(personas: List[str], arm_names: List[str],
            retrieval_result_dir: str, dataset_dir: str,
            baseline: str = "simple_embedding") -> dict:

    per_persona: Dict[str, dict] = {}
    paired_scores: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)

    for persona in personas:
        path = os.path.join(dataset_dir, f"{persona}_dialogues_256k.json")
        if not os.path.exists(path):
            logger.warning("No dialogue file for %s", persona)
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        gold = schema.extract_retrieval_gold(data)
        corpus = schema.corpus_ids(data)
        persona_dir = os.path.join(retrieval_result_dir, persona)
        persona_out = {}

        for arm in arm_names:
            results = _load_arm(persona_dir, arm)
            if results is None:
                continue
            summary = compute_auto_metrics(results, gold, corpus)
            summary.update(_depth_report(results))
            persona_out[arm] = {"auto": summary}
            paired_scores[arm][persona] = {
                metric: per_query_scores(results, gold, corpus, k=k, metric=m)
                for metric, (m, k) in {
                    "recall_any@5": ("recall_any", 5),
                    "recall_any@10": ("recall_any", 10),
                    "recall_all@10": ("recall_all", 10),
                    "ndcg@10": ("ndcg", 10),
                }.items()
            }
            logger.info("  %-20s %s", arm,
                        {k: summary[k] for k in ("recall_any@5", "recall_any@10",
                                                 "ndcg@10", "n_evaluated")})

        if persona_out:
            per_persona[persona] = persona_out

    if not per_persona:
        logger.error("No retrieval results found. Run eval/run_eval.py first.")
        return {}

    present = [a for a in arm_names if any(a in v for v in per_persona.values())]
    aggregate = _aggregate(per_persona, present)
    _print_auto_table(aggregate, present, per_persona)
    comparisons = _paired_comparisons(paired_scores, present, baseline)

    out_path = os.path.join(retrieval_result_dir, "analysis_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"per_persona": per_persona, "aggregate": aggregate,
                   "paired_comparisons": comparisons,
                   "personas": personas, "arms": present}, f, indent=2, ensure_ascii=False)
    logger.info("Analysis written to %s", out_path)
    return aggregate


def _aggregate(per_persona: dict, arm_names: List[str]) -> dict:
    """Macro-average across personas, carrying the spread and the n."""
    aggregate = {}
    for arm in arm_names:
        vals = defaultdict(list)
        n_total = 0
        for _persona, arms in per_persona.items():
            auto = arms.get(arm, {}).get("auto")
            if not auto:
                continue
            n_total += auto.get("n_evaluated", 0)
            for m in AUTO_METRICS + ["mean_depth", "empty_rate", "shallow_rate"]:
                if m in auto:
                    vals[m].append(auto[m])
        aggregate[arm] = {
            m: {"mean": round(float(np.mean(v)), 4),
                "std": round(float(np.std(v)), 4),
                "n_personas": len(v)}
            for m, v in vals.items()
        }
        aggregate[arm]["n_queries_total"] = n_total
    return aggregate


def _print_auto_table(aggregate: dict, arm_names: List[str], per_persona: dict):
    """Macro-average over personas.

    The previous implementation's comprehension read ``vals[i]`` inside a loop
    over personas without using the loop variable, so every column reported the
    first persona's value rather than a mean.
    """
    cols = [(f"recall_any@{k}", f"any@{k}") for k in DEFAULT_KS] + \
           [(f"recall_all@{k}", f"all@{k}") for k in DEFAULT_KS] + \
           [(f"ndcg@{k}", f"ndcg@{k}") for k in DEFAULT_KS]
    width = 21 + 7 + 9 * len(cols)

    print("\n" + "=" * width)
    print("RETRIEVAL QUALITY  (macro-average over personas)")
    print("  any@k = at least one gold session in top k    all@k = every gold session in top k")
    print("=" * width)
    header = f"{'arm':<21}{'n':>7}"
    for _key, label in cols:
        header += f"{label:>9}"
    print(header)
    print("-" * width)
    for arm in arm_names:
        agg = aggregate.get(arm, {})
        row = f"{arm:<21}{agg.get('n_queries_total', 0):>7}"
        for key, _label in cols:
            row += f"{agg.get(key, {}).get('mean', 0.0):>9.4f}"
        print(row)

    # Spread across personas, so a headline mean driven by one easy persona is
    # visible rather than implied.
    print("-" * width)
    print(f"{'std across personas':<21}{'':>7}", end="")
    for key, _label in cols:
        stds = [aggregate.get(a, {}).get(key, {}).get("std", 0.0) for a in arm_names]
        print(f"{max(stds) if stds else 0.0:>9.4f}", end="")
    print("  (worst arm)")

    print(f"\n{'arm':<21}{'depth':>9}{'empty':>9}{'shallow':>9}   "
          f"(shallow = returned <{MAX_METRIC_K} items, so @{MAX_METRIC_K} is capped)")
    print("-" * 70)
    for arm in arm_names:
        agg = aggregate.get(arm, {})
        print(f"{arm:<21}"
              f"{agg.get('mean_depth', {}).get('mean', 0.0):>9.2f}"
              f"{agg.get('empty_rate', {}).get('mean', 0.0):>9.3f}"
              f"{agg.get('shallow_rate', {}).get('mean', 0.0):>9.3f}")

    print(f"\n{'arm':<21}{'persona':<17}{'recall_any@5':>14}{'recall_any@10':>15}{'ndcg@10':>10}")
    print("-" * 78)
    for arm in arm_names:
        for persona, arms in sorted(per_persona.items()):
            auto = arms.get(arm, {}).get("auto")
            if auto:
                print(f"{arm:<21}{persona:<17}{auto['recall_any@5']:>14.4f}"
                      f"{auto['recall_any@10']:>15.4f}{auto['ndcg@10']:>10.4f}")


def _paired_comparisons(paired_scores: dict, arm_names: List[str], baseline: str) -> list:
    """Every arm against the baseline, pooling queries across personas.

    Queries are pooled with a persona-qualified key so that identical question
    text in two personas is not silently merged into one pair.
    """
    if baseline not in paired_scores:
        logger.warning("Baseline arm %r absent; skipping paired tests", baseline)
        return []

    def pooled(arm: str, metric: str) -> Dict[str, float]:
        out = {}
        for persona, metrics in paired_scores[arm].items():
            for q, v in metrics.get(metric, {}).items():
                out[f"{persona}\x00{q}"] = v
        return out

    print("\n" + "=" * 92)
    print(f"PAIRED COMPARISON vs {baseline}   (McNemar exact for recall, "
          f"bootstrap CI for NDCG)")
    print("=" * 92)
    print(f"{'arm':<21}{'metric':<15}{'delta':>9}{'p / 95% CI':>26}{'discordant':>12}")
    print("-" * 92)

    comparisons = []
    for arm in arm_names:
        if arm == baseline:
            continue
        for metric in ("recall_any@5", "recall_any@10", "recall_all@10", "ndcg@10"):
            res = summarize_comparison(arm, baseline, pooled(arm, metric),
                                       pooled(baseline, metric), metric)
            comparisons.append(res)
            if res.get("n", 0) == 0:
                continue
            if res["test"] == "mcnemar_exact":
                detail = f"p={res['p_value']:.4g}"
                disc = str(res["discordant"])
            else:
                detail = f"[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]"
                disc = "-"
            print(f"{arm:<21}{metric:<15}{res['delta']:>+9.4f}{detail:>26}{disc:>12}")
            if "warning" in res:
                print(f"{'':<21}  ! {res['warning']}")
    return comparisons


def main():
    p = argparse.ArgumentParser(description="Analyze RealMemBench retrieval arms")
    p.add_argument("--personas", default=None, help="Comma-separated")
    p.add_argument("--all-personas", action="store_true")
    p.add_argument("--arms", default=None, help="Comma-separated; default: autodetect")
    p.add_argument("--retrieval-result-dir", default=RETRIEVAL_RESULT_DIR)
    p.add_argument("--dataset-dir", default=DATASET_DIR)
    p.add_argument("--baseline", default="simple_embedding")
    args = p.parse_args()

    if args.all_personas:
        personas = list_personas(args.dataset_dir)
    elif args.personas:
        personas = [x.strip() for x in args.personas.split(",") if x.strip()]
    else:
        p.error("Specify --personas or --all-personas")

    if args.arms:
        arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    else:
        found = set()
        for persona in personas:
            d = os.path.join(args.retrieval_result_dir, persona)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith("_retrieval_results.json"):
                        name = f[: -len("_retrieval_results.json")]
                        found.add(name[len("DRYRUN-"):] if name.startswith("DRYRUN-") else name)
        # Stable, meaningful order rather than alphabetical.
        preferred = ["no_memory", "concrete_only", "simple_embedding", "stage2_only",
                     "three_stage_rerank", "three_stage_gated"]
        arm_names = [a for a in preferred if a in found] + sorted(found - set(preferred))
        if not arm_names:
            logger.error("No retrieval results under %s", args.retrieval_result_dir)
            return

    logger.info("Personas: %s", personas)
    logger.info("Arms: %s", arm_names)
    analyze(personas, arm_names, args.retrieval_result_dir, args.dataset_dir, args.baseline)


if __name__ == "__main__":
    main()

