"""Paired significance tests on the LLM-judge scores.

run_qa_eval.py reports per-arm means. A mean difference is not a result on its
own: arms are judged on identical queries, so the comparison is paired and the
information is in the per-query disagreements. This reads back the
``detailed_results`` each arm wrote and tests them properly.

Three views of the 0-3 QA scale, because the mean alone hides the mechanism:

  mean          bootstrap CI on the paired difference
  used_memory   share scoring >= 2 ("used at least part of the user's memory")
  conflicted    share scoring 0 ("contradicts the user's memory")

That last one is the metric run_qa_eval calls ``qa_hallucination_rate``,
following the vendored script's naming. The name is misleading and worth
restating whenever it is quoted: the official rubric defines score 0 as
*conflicting with* user memory, not as fabrication. A no-memory arm scores few
0s simply because generic answers cannot contradict anything — it parks on
score 1 ("does not conflict but is generic") instead. Read `conflicted`
alongside the score-1 share or it will tell you the opposite of the truth.

    python -m eval.analyze_qa --personas Lin_Wanyu
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import RETRIEVAL_RESULT_DIR, DEFAULT_ARMS, list_personas
from eval.stats import paired_binary_test, paired_bootstrap_ci


def load_arm(persona_dir: str, arm: str, problems: list) -> dict:
    """Load one arm's per-query judge records, recording *why* if it cannot.

    'No files found' was previously reported for three quite different
    situations — absent file, unreadable file, and a file whose
    detailed_results was empty. They need different fixes, so they get
    different messages.
    """
    path = os.path.join(persona_dir, f"{arm}_llm_metrics.json")
    if not os.path.exists(path):
        problems.append(f"missing: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        problems.append(f"unreadable ({type(exc).__name__}): {path}")
        return {}

    detailed = payload.get("detailed_results")
    if detailed is None:
        problems.append(
            f"no 'detailed_results' key: {path} "
            f"(keys present: {sorted(payload)}) — written by an older run_qa_eval")
        return {}
    if not detailed:
        summary = payload.get("summary", {})
        problems.append(
            f"empty 'detailed_results': {path} "
            f"(summary says n_qa_scored={summary.get('n_qa_scored')})")
        return {}
    return detailed


def main():
    p = argparse.ArgumentParser(description="Paired tests on LLM-judge scores")
    p.add_argument("--personas", default=None)
    p.add_argument("--all-personas", action="store_true")
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    p.add_argument("--baseline", default="simple_embedding")
    p.add_argument("--retrieval-result-dir", default=RETRIEVAL_RESULT_DIR)
    args = p.parse_args()

    personas = (list_personas() if args.all_personas
                else [x.strip() for x in (args.personas or "").split(",") if x.strip()])
    if not personas:
        p.error("Specify --personas or --all-personas")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    # arm -> {persona\0question: record}
    pooled = {a: {} for a in arms}
    problems = []
    for persona in personas:
        pdir = os.path.join(args.retrieval_result_dir, persona)
        for arm in arms:
            for q, rec in load_arm(pdir, arm, problems).items():
                pooled[arm][f"{persona}\x00{q}"] = rec

    present = [a for a in arms if pooled[a]]
    if not present:
        print("Could not load any judge records.\n")
        print(f"  looked under : {args.retrieval_result_dir}")
        print(f"  personas     : {personas}")
        print(f"  arms         : {arms}\n")
        for msg in problems:
            print(f"  {msg}")
        for persona in personas:
            pdir = os.path.join(args.retrieval_result_dir, persona)
            print(f"\n  files actually in {pdir}:")
            if not os.path.isdir(pdir):
                print("    (directory does not exist)")
                continue
            names = sorted(os.listdir(pdir))
            if not names:
                print("    (empty)")
            for n in names:
                size = os.path.getsize(os.path.join(pdir, n))
                print(f"    {size:>12,}  {n}")
        print("\nIf you see *_generation_results.json but no *_llm_metrics.json, "
              "the judge\nphase did not finish. Re-run:  python -m eval.run_qa_eval "
              "--personas " + ",".join(personas))
        return 1

    if problems:
        print("Some arms could not be loaded:")
        for msg in problems:
            print(f"  {msg}")
        print()

    def series(arm, field, transform=None):
        out = {}
        for k, rec in pooled[arm].items():
            if field in rec and isinstance(rec[field], (int, float)):
                out[k] = transform(rec[field]) if transform else float(rec[field])
        return out

    print("\n" + "=" * 100)
    print("LLM JUDGE — per-arm summary")
    print("=" * 100)
    print(f"{'arm':<22}{'n_qa':>7}{'qa_mean':>10}{'used>=2':>10}{'conflict':>10}"
          f"{'perfect':>10}{'n_mem':>7}{'mem_recall':>12}{'helpful':>9}")
    print("-" * 100)
    for arm in present:
        qa = series(arm, "qa_score")
        rec = series(arm, "mem_recall")
        helpful = series(arm, "mem_helpful")
        dist = Counter(int(v) for v in qa.values())
        n = len(qa) or 1
        print(f"{arm:<22}{len(qa):>7}{np.mean(list(qa.values())) if qa else 0:>10.4f}"
              f"{sum(1 for v in qa.values() if v >= 2)/n:>10.4f}"
              f"{dist.get(0,0)/n:>10.4f}{dist.get(3,0)/n:>10.4f}"
              f"{len(rec):>7}"
              f"{(np.mean(list(rec.values())) if rec else float('nan')):>12.4f}"
              f"{(np.mean(list(helpful.values())) if helpful else float('nan')):>9.4f}")

    print(f"\n{'arm':<22}" + "".join(f"{'score '+str(i):>10}" for i in range(4)))
    print("-" * 62)
    for arm in present:
        dist = Counter(int(v) for v in series(arm, "qa_score").values())
        print(f"{arm:<22}" + "".join(f"{dist.get(i,0):>10}" for i in range(4)))
    print("\nscore 1 = 'does not conflict but is generic'. A no-memory arm parks "
          "there,\nwhich is why its 'conflict' rate looks flattering.")

    if args.baseline not in pooled or not pooled[args.baseline]:
        print(f"\nBaseline {args.baseline!r} absent; skipping paired tests.")
        return 0

    print("\n" + "=" * 100)
    print(f"PAIRED vs {args.baseline}")
    print("=" * 100)
    print(f"{'arm':<22}{'metric':<14}{'delta':>9}{'p / 95% CI':>26}{'discordant':>12}")
    print("-" * 100)

    for arm in present:
        if arm == args.baseline:
            continue
        # Continuous scales -> bootstrap CI on the paired mean difference.
        for field, label in (("qa_score", "qa_mean"), ("mem_recall", "mem_recall"),
                             ("mem_helpful", "mem_helpful")):
            a, b = series(arm, field), series(args.baseline, field)
            if not (set(a) & set(b)):
                continue
            r = paired_bootstrap_ci(a, b)
            detail = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
            print(f"{arm:<22}{label:<14}{r['delta']:>+9.4f}{detail:>26}{'-':>12}")

        # Binary readings of the same scale -> McNemar.
        for thr, label in ((lambda v: 1.0 if v >= 2 else 0.0, "used>=2"),
                           (lambda v: 1.0 if v == 0 else 0.0, "conflict")):
            a, b = series(arm, "qa_score", thr), series(args.baseline, "qa_score", thr)
            if not (set(a) & set(b)):
                continue
            r = paired_binary_test(a, b)
            detail = f"p={r['p_value']:.4g}"
            print(f"{arm:<22}{label:<14}{r['delta']:>+9.4f}{detail:>26}"
                  f"{r['discordant']:>12}")
            if "warning" in r:
                print(f"{'':<22}  ! {r['warning']}")

    print("\nNote: mem_recall is computed only where evidence was retrieved AND "
          "gold\nmemory is annotated, so n_mem differs between arms. A missing "
          "value is a\nskipped measurement, not a zero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
