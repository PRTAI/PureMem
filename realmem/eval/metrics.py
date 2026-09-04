"""Recall@k / NDCG@k, semantics-identical to the vendored official scorer.

This is a re-implementation of ``compute_auto_metrics_for_realmem.py``'s
``dcg`` / ``ndcg`` / ``evaluate_retrieval``, not an improvement on them. Where
the official version looks odd it is reproduced anyway, because the point is
comparability with published numbers, and ``selftest_metrics.py`` asserts the
two agree exactly on real data.

Two deliberate deviations, both mechanical:

* ``np.asfarray`` was removed in NumPy 2.0, so the official module raises
  AttributeError on a modern install. We use ``np.asarray(..., dtype=float)``,
  which is what ``asfarray`` did. The selftest injects a shim so the official
  code can still be executed for the comparison.
* Retrieved ids absent from the corpus are skipped rather than occupying a rank
  slot — this matches the official loop, which only appends indices it can
  resolve.

Reproduced quirks worth knowing when reading results:

* ``recall_all`` is ``all(gold ⊆ retrieved)``, which is vacuously true for an
  empty gold set. Gold is never empty here (the extractor drops those queries),
  so it does not bite, but the definition is the official one.
* The ideal DCG is computed over the whole corpus relevance vector, which
  amounts to "min(|gold|, k) ones" — the standard ideal ranking.
* A query whose gold set is larger than k cannot reach ``recall_all@k``. Gold
  sets run up to 12 sessions, so ``recall_all@5`` is unreachable for some
  queries by construction.
"""

from typing import Dict, Iterable, List, Sequence, Set

import numpy as np

DEFAULT_KS = (5, 10, 20)


def dcg(relevances: Sequence[float], k: int) -> float:
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    return float(rel[0] + np.sum(rel[1:] / np.log2(np.arange(2, rel.size + 1))))


def ndcg_at(rankings: Sequence[int], correct_docs: Set[str],
            corpus_ids: Sequence[str], k: int) -> float:
    relevances = [1 if doc_id in correct_docs else 0 for doc_id in corpus_ids]
    ranked = [relevances[idx] if idx < len(relevances) else 0 for idx in rankings[:k]]
    ideal = dcg(sorted(relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg(ranked, k) / ideal


def evaluate_retrieval(rankings: Sequence[int], correct_docs: Set[str],
                       corpus_ids: Sequence[str], k: int):
    """-> (recall_any, recall_all, ndcg)"""
    recalled = {corpus_ids[idx] for idx in rankings[:k] if idx < len(corpus_ids)}
    recall_any = float(any(d in recalled for d in correct_docs))
    recall_all = float(all(d in recalled for d in correct_docs))
    return recall_any, recall_all, ndcg_at(rankings, correct_docs, corpus_ids, k)


def rankings_for(ranked_items: Iterable[dict], corpus_id_to_idx: Dict[str, int],
                 res_type: str = "chunk") -> List[int]:
    """Ranked item dicts -> corpus indices, official filtering rules."""
    out = []
    for item in ranked_items:
        if isinstance(item, dict) and item.get("res_type") == res_type:
            rid = item.get("chunk_id")
            if rid in corpus_id_to_idx:
                out.append(corpus_id_to_idx[rid])
    return out


def compute_auto_metrics(retrieval_results: dict, query_to_gold: Dict[str, List[str]],
                         corpus_ids: Sequence[str], ks: Sequence[int] = DEFAULT_KS) -> dict:
    """Aggregate over one arm's results for one persona.

    The denominator is the set of queries present in ``retrieval_results`` that
    also have gold — matching the official script, which iterates the retrieval
    file and skips what it cannot match. So an arm returning an empty list still
    counts (and scores 0); only a query absent from the file is dropped.
    """
    corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    acc = {f"{m}@{k}": [] for k in ks for m in ("recall_all", "recall_any", "ndcg")}

    n_matched = n_unmatched = 0
    for query_text, result in retrieval_results.items():
        gold_key = query_text if query_text in query_to_gold else query_text.strip()
        if gold_key not in query_to_gold:
            n_unmatched += 1
            continue
        gold = set(query_to_gold[gold_key])
        rankings = rankings_for(result.get("ranked_items", []), corpus_id_to_idx)

        for k in ks:
            r_any, r_all, nd = evaluate_retrieval(rankings, gold, corpus_ids, k)
            acc[f"recall_any@{k}"].append(r_any)
            acc[f"recall_all@{k}"].append(r_all)
            acc[f"ndcg@{k}"].append(nd)
        n_matched += 1

    summary = {key: (round(float(np.mean(v)), 4) if v else 0.0) for key, v in acc.items()}
    summary["n_evaluated"] = n_matched
    summary["n_unmatched"] = n_unmatched
    return summary


def per_query_scores(retrieval_results: dict, query_to_gold: Dict[str, List[str]],
                     corpus_ids: Sequence[str], k: int, metric: str = "recall_any"
                     ) -> Dict[str, float]:
    """Per-query scores keyed by question — the input to paired significance
    tests, which need the pairing preserved rather than a mean."""
    corpus_id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    out = {}
    for query_text, result in retrieval_results.items():
        gold_key = query_text if query_text in query_to_gold else query_text.strip()
        if gold_key not in query_to_gold:
            continue
        gold = set(query_to_gold[gold_key])
        rankings = rankings_for(result.get("ranked_items", []), corpus_id_to_idx)
        r_any, r_all, nd = evaluate_retrieval(rankings, gold, corpus_ids, k)
        out[query_text] = {"recall_any": r_any, "recall_all": r_all, "ndcg": nd}[metric]
    return out
