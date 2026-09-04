"""Prove our metrics are the official metrics. No network, no spend.

The claim this harness rests on is that its Recall/NDCG are comparable with
published RealMemBench numbers. That is only true if our implementation agrees
with the vendored scorer, so this file does not re-derive the formulas — it
executes ``compute_auto_metrics_for_realmem.py`` on real data and diffs.

Three levels, each a prerequisite for the next:

  1. ground truth + corpus definitions agree
  2. the per-query scoring functions agree on adversarial rankings
  3. end-to-end aggregates agree on synthetic retrieval results

Two compatibility notes:

* ``np.asfarray`` was removed in NumPy 2.0, so the vendored module cannot run
  unpatched on a modern install. We inject the shim it needs rather than edit
  the file — the vendored scripts stay read-only, and on the user's server this
  same breakage will hit ``compute_auto_metrics_for_realmem.py`` directly.
* Our extractor restricts to ``is_query`` turns while the official one walks
  every User turn. Measured over all 10 personas the official set is a strict
  superset (43 extra queries out of 1458) and the two agree on the gold set for
  every shared query. Since the official script takes its denominator from the
  retrieval file, this is not a difference in what gets scored — asserted below.

Run:  python -m eval.selftest_metrics
"""

import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NumPy 2.x removed asfarray; the vendored scorer still calls it.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

from eval import metrics as ours
from eval import schema
from eval.config import DATASET_DIR, list_personas

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_auto_metrics_for_realmem as official  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def load(persona):
    with open(os.path.join(DATASET_DIR, f"{persona}_dialogues_256k.json"),
             "r", encoding="utf-8") as f:
        return json.load(f)


# ── 1. Definitions ──

def test_ground_truth_agrees():
    total_ours = total_official = total_extra = 0
    disagreements = []
    corpus_mismatch = []

    for persona in list_personas():
        data = load(persona)
        off_gold, off_corpus = official.get_ground_truth(data)
        our_gold = schema.extract_retrieval_gold(data)
        our_corpus = schema.corpus_ids(data)

        total_ours += len(our_gold)
        total_official += len(off_gold)
        total_extra += len(set(off_gold) - set(our_gold))

        if sorted(off_corpus) != our_corpus:
            corpus_mismatch.append(persona)
        for q in set(off_gold) & set(our_gold):
            if set(off_gold[q]) != set(our_gold[q]):
                disagreements.append((persona, q[:40]))
        extra_ours = set(our_gold) - set(off_gold)
        if extra_ours:
            disagreements.append((persona, f"{len(extra_ours)} queries we have and official does not"))

    check("corpus id sets are identical", not corpus_mismatch, str(corpus_mismatch))
    check("gold sets agree on every shared query", not disagreements,
          str(disagreements[:3]))
    check("our query set is a subset of the official one",
          total_ours <= total_official,
          f"ours={total_ours} official={total_official}")
    print(f"        ours={total_ours} official={total_official} "
          f"(official-only={total_extra}, all non-is_query turns)")


# ── 2. Scoring functions ──

def test_scoring_functions_agree():
    rng = random.Random(42)
    corpus = [f"S{i:03d}" for i in range(60)]
    mismatches = []

    for trial in range(300):
        n_gold = rng.choice([1, 1, 2, 3, 5, 12])
        gold = set(rng.sample(corpus, n_gold))
        depth = rng.choice([0, 1, 3, 5, 15, 20, 25])
        rankings = [corpus.index(c) for c in rng.sample(corpus, min(depth, len(corpus)))]
        # Occasionally inject an out-of-range index, as a hallucinated id would.
        if trial % 7 == 0 and rankings:
            rankings[0] = len(corpus) + 5

        for k in (5, 10, 20):
            o_any, o_all, o_ndcg = official.evaluate_retrieval(rankings, list(gold), corpus, k=k)
            u_any, u_all, u_ndcg = ours.evaluate_retrieval(rankings, gold, corpus, k)
            if (o_any, o_all) != (u_any, u_all) or abs(o_ndcg - u_ndcg) > 1e-12:
                mismatches.append((trial, k, (o_any, o_all, o_ndcg), (u_any, u_all, u_ndcg)))

    check("evaluate_retrieval matches on 300 randomized cases x 3 k values",
          not mismatches, str(mismatches[:2]))

    dcg_bad = []
    for _ in range(200):
        rel = [rng.choice([0, 1]) for _ in range(rng.randint(0, 30))]
        k = rng.choice([5, 10, 20])
        if abs(official.dcg(rel, k) - ours.dcg(rel, k)) > 1e-12:
            dcg_bad.append((rel, k))
    check("dcg matches on 200 randomized cases", not dcg_bad, str(dcg_bad[:2]))


# ── 3. End to end ──

def _synthetic_results(data, mode, rng):
    """Retrieval results of varying quality, in the official on-disk format."""
    gold = schema.extract_retrieval_gold(data)
    corpus = schema.corpus_ids(data)
    out = {}
    for question, golds in gold.items():
        if mode == "empty":
            items = []
        elif mode == "perfect":
            rest = [c for c in corpus if c not in golds]
            picked = list(golds) + rng.sample(rest, min(20 - len(golds), len(rest)))
            items = picked[:20]
        elif mode == "random":
            items = rng.sample(corpus, min(20, len(corpus)))
        elif mode == "half":
            keep = list(golds)[: max(1, len(golds) // 2)]
            rest = [c for c in corpus if c not in golds]
            items = rng.sample(rest, min(10, len(rest))) + keep
        elif mode == "hallucinated":
            items = [f"NOT_A_SESSION_{i}" for i in range(5)] + list(golds)
        out[question] = {
            "question": question,
            "ranked_items": [
                {"res_type": "chunk", "chunk_id": cid, "content": "x", "rank": i + 1}
                for i, cid in enumerate(items)
            ],
        }
    return out


def _official_aggregate(retrieval_results, data):
    """Replicates compute_auto_metrics_for_realmem.main()'s loop exactly."""
    query_to_gold, all_corpus_ids = official.get_ground_truth(data)
    all_corpus_ids = sorted(list(all_corpus_ids))
    corpus_id_to_idx = {cid: i for i, cid in enumerate(all_corpus_ids)}

    acc = {f"{m}@{k}": [] for k in (5, 10, 20)
           for m in ("recall_all", "recall_any", "ndcg_any")}
    n = 0
    for query_text, result_obj in retrieval_results.items():
        gt_key = query_text if query_text in query_to_gold else query_text.strip()
        if gt_key not in query_to_gold:
            continue
        correct_docs = query_to_gold[gt_key]
        rankings = []
        for item in result_obj.get("ranked_items", []):
            if isinstance(item, dict) and item.get("res_type") == "chunk":
                rid = item.get("chunk_id")
                if rid in corpus_id_to_idx:
                    rankings.append(corpus_id_to_idx[rid])
        for k in (5, 10, 20):
            r_any, r_all, nd = official.evaluate_retrieval(rankings, correct_docs,
                                                           all_corpus_ids, k=k)
            acc[f"recall_all@{k}"].append(r_all)
            acc[f"recall_any@{k}"].append(r_any)
            acc[f"ndcg_any@{k}"].append(nd)
        n += 1
    summary = {key: round(float(np.mean(v)), 4) for key, v in acc.items() if v}
    summary["n_evaluated"] = n
    return summary


def test_end_to_end_aggregates_agree():
    rng = random.Random(7)
    all_bad = []
    for persona in list_personas()[:4]:
        data = load(persona)
        gold = schema.extract_retrieval_gold(data)
        corpus = schema.corpus_ids(data)

        for mode in ("perfect", "half", "random", "empty", "hallucinated"):
            results = _synthetic_results(data, mode, rng)
            off = _official_aggregate(results, data)
            us = ours.compute_auto_metrics(results, gold, corpus)

            for k in (5, 10, 20):
                pairs = [(f"recall_all@{k}", f"recall_all@{k}"),
                         (f"recall_any@{k}", f"recall_any@{k}"),
                         (f"ndcg_any@{k}", f"ndcg@{k}")]
                for o_key, u_key in pairs:
                    if abs(off.get(o_key, 0.0) - us.get(u_key, 0.0)) > 1e-9:
                        all_bad.append((persona, mode, o_key, off.get(o_key), us.get(u_key)))
            if off["n_evaluated"] != us["n_evaluated"]:
                all_bad.append((persona, mode, "n_evaluated",
                                off["n_evaluated"], us["n_evaluated"]))

    check("aggregates match the official scorer across 4 personas x 5 quality modes",
          not all_bad, str(all_bad[:3]))


def test_depth_ceiling_is_visible():
    """A 15-long list cannot score recall@20 above recall@15 — the failure mode
    the previous POOL_SIZE=15 config produced silently."""
    data = load(list_personas()[0])
    gold = schema.extract_retrieval_gold(data)
    corpus = schema.corpus_ids(data)
    rng = random.Random(3)

    deep = _synthetic_results(data, "random", rng)
    shallow = {q: {"question": q, "ranked_items": v["ranked_items"][:15]}
               for q, v in deep.items()}

    d = ours.compute_auto_metrics(deep, gold, corpus)
    s = ours.compute_auto_metrics(shallow, gold, corpus)
    check("truncating to 15 cannot improve recall@20",
          s["recall_any@20"] <= d["recall_any@20"] + 1e-12)
    check("truncating to 15 leaves recall@10 untouched",
          abs(s["recall_any@10"] - d["recall_any@10"]) < 1e-12)


def test_official_prompts_load():
    from eval.official_prompts import (QA_eval_prompt, Mem_eval_prompt,
                                       construct_evidence_text, describe)
    check("QA prompt carries the full rubric",
          "sounds reasonable" in QA_eval_prompt and "Score 3" in QA_eval_prompt,
          f"{len(QA_eval_prompt)} chars")
    check("Mem prompt carries the three-step recall definition",
          "step1" in Mem_eval_prompt and "step3" in Mem_eval_prompt)
    check("evidence formatting matches the official layout",
          construct_evidence_text(
              [{"res_type": "chunk", "content": "a"},
               {"res_type": "chunk", "content": "b"}], "chunk", 5)
          == "---- idx 1 ----\na\n\n---- idx 2 ----\nb")
    print(f"        {describe()}")


def test_dataset_facts_still_hold():
    """The design rests on measured properties of the corpus. If a dataset
    refresh changes them, the Stage-1 rationale needs revisiting."""
    forward = own = gold_without_abstract = total = 0
    for persona in list_personas():
        data = load(persona)
        dl = data["dialogues"]
        idx = {s["session_uuid"]: i for i, s in enumerate(dl) if s.get("session_uuid")}
        for s_idx, session, t_idx, _turn, _q in schema.iter_queries(data):
            turns = session["dialogue_turns"]
            nxt = turns[t_idx + 1] if t_idx + 1 < len(turns) else {}
            for u in set(nxt.get("memory_session_uuids", []) or []):
                if u not in idx:
                    continue
                total += 1
                j = idx[u]
                if j > s_idx:
                    forward += 1
                elif j == s_idx:
                    own += 1
                if not dl[j].get("extracted_memory"):
                    gold_without_abstract += 1

    check("gold never points at a future session", forward == 0, f"{forward}/{total}")
    check("gold almost never points at the query's own session", own <= 5, f"{own}/{total}")
    check("gold never points at a session with no extracted_memory",
          gold_without_abstract == 0, f"{gold_without_abstract}/{total}")
    print(f"        {total} gold references checked")


def main():
    print("selftest_metrics")
    print("-" * 60)
    for fn in [
        test_ground_truth_agrees,
        test_scoring_functions_agree,
        test_end_to_end_aggregates_agree,
        test_depth_ceiling_is_visible,
        test_official_prompts_load,
        test_dataset_facts_still_hold,
    ]:
        print(f"\n{fn.__name__}:")
        try:
            fn()
        except Exception:
            import traceback
            traceback.print_exc()
            FAILURES.append(fn.__name__)

    print("\n" + "-" * 60)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("our metrics are the official metrics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
