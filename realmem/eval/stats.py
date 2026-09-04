"""Paired significance tests for arm comparisons.

Arms are evaluated on identical query sets, so the comparison is paired and the
information lives in the disagreements. Reporting two means and eyeballing the
gap throws that structure away.

Pure Python + NumPy on purpose: scipy is not in requirements.txt and the offline
selftests must run without it.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts.

    b = A right / B wrong, c = A wrong / B right. Concordant pairs carry no
    information and are excluded by construction.
    """
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * _binom_cdf(min(b, c), n, 0.5))


def paired_binary_test(a: Dict[str, float], b: Dict[str, float]) -> dict:
    """Compare two arms on a binary per-query metric (recall_any, recall_all)."""
    shared = sorted(set(a) & set(b))
    n = len(shared)
    if n == 0:
        return {"n": 0, "note": "no shared queries"}

    both = a_only = b_only = neither = 0
    for q in shared:
        av, bv = a[q] > 0.5, b[q] > 0.5
        if av and bv:
            both += 1
        elif av:
            a_only += 1
        elif bv:
            b_only += 1
        else:
            neither += 1

    discordant = a_only + b_only
    mean_a = sum(a[q] for q in shared) / n
    mean_b = sum(b[q] for q in shared) / n

    out = {
        "n": n,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "delta": round(mean_a - mean_b, 4),
        "both": both, "a_only": a_only, "b_only": b_only, "neither": neither,
        "discordant": discordant,
        "p_value": round(mcnemar_exact(a_only, b_only), 6),
    }
    # Below roughly ten discordant pairs the test cannot separate anything,
    # whatever the p-value reads.
    if discordant < 10:
        out["warning"] = (f"only {discordant} discordant pairs — the test is "
                          f"near-uninformative at this sample size")
    return out


def paired_bootstrap_ci(a: Dict[str, float], b: Dict[str, float],
                        n_boot: int = 10000, alpha: float = 0.05,
                        seed: int = 42) -> dict:
    """Percentile bootstrap CI for the paired mean difference (a - b).

    Suits continuous metrics like NDCG where McNemar does not apply. Seeded so
    reruns are reproducible.
    """
    shared = sorted(set(a) & set(b))
    n = len(shared)
    if n == 0:
        return {"n": 0, "note": "no shared queries"}

    diffs = np.array([a[q] - b[q] for q in shared], dtype=float)
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "n": n,
        "delta": round(float(diffs.mean()), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "significant": bool(lo > 0 or hi < 0),
    }


def summarize_comparison(name_a: str, name_b: str,
                         scores_a: Dict[str, Dict[str, float]],
                         scores_b: Dict[str, Dict[str, float]],
                         metric: str) -> dict:
    """Pick the right test for the metric type."""
    if metric.startswith("recall"):
        res = paired_binary_test(scores_a, scores_b)
        res["test"] = "mcnemar_exact"
    else:
        res = paired_bootstrap_ci(scores_a, scores_b)
        res["test"] = "paired_bootstrap"
    res.update({"arm_a": name_a, "arm_b": name_b, "metric": metric})
    return res
