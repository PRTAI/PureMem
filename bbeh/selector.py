"""
bbeh/selector.py — which train items become memory.

This module is where claim 1 lives, so it is worth being precise about what is
being claimed and what would falsify it.

**Claim 1.** A subset chosen from the student's *zone of proximal development*
(ZPD) works as well as using the entire train set as memory.

That statement is only interesting alongside a size-matched control. If a
random subset of the same size also matched ``full``, the result would say
"memory size doesn't matter", not "ZPD selection works". So the arms are:

===================  ========================================================
``full``             every train item with a verified teacher CoT
``zpd``              pass_rate inside the band — the hypothesis
``random_matched``   uniform random, size-matched to ``zpd``  ← the control
``stratified_matched`` per-task proportional, size-matched     ← control
``easy_only``        pass_rate == 1.0, size-matched (already mastered)
``hard_only``        pass_rate == 0.0, size-matched (out of reach)
===================  ========================================================

Claim 1 holds iff  ``acc(zpd) ≈ acc(full)``  **and**  ``acc(zpd) > acc(random_matched)``.
If ``zpd ≈ full ≈ random_matched`` the honest conclusion is that selection is
irrelevant at this size — which is a real finding, just not the hypothesised one.

**Difficulty is measured against the STUDENT, not the teacher.** ZPD is a
property of the learner. We use the teacher only to author the CoT; the band
comes from ``probe.py`` sampling the student k times per item.

**Size matching.** Harder items tend to have longer CoTs, so N ZPD items can
carry more memory chunks than N random items — a confound that would flatter
the ZPD arm. ``match_on='chunks'`` equalises total chunk count instead of item
count. Both counts are always recorded in the returned info dict so the
analysis can state which was held constant.
"""

import logging
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from bbeh import config

METHODS = (
    'full',
    'zpd',
    'random_matched',
    'stratified_matched',
    'easy_only',
    'hard_only',
)

# Methods that need a difficulty probe to be meaningful.
NEEDS_DIFFICULTY = ('zpd', 'easy_only', 'hard_only')

# Methods that are size-matched against another arm (usually ``zpd``).
NEEDS_SIZE = ('random_matched', 'stratified_matched', 'easy_only', 'hard_only')


# ═════════════════════════════════════════════════════════════════════
#  Difficulty band
# ═════════════════════════════════════════════════════════════════════

def in_zpd(pass_rate: float, low: float = config.ZPD_LOW,
           high: float = config.ZPD_HIGH, strict: bool = False) -> bool:
    """Is this item in the zone of proximal development?

    ``strict`` uses the pure Vygotskian reading — the student solves it
    *sometimes*, i.e. ``0 < p < 1`` — which is band-independent. The default
    banded form additionally trims the extremes for small k, where p=0.2 is a
    noisy estimate of "nearly never".
    """
    if pass_rate is None:
        return False
    if strict:
        return 0.0 < pass_rate < 1.0
    return low <= pass_rate <= high


def difficulty_histogram(pool: Sequence[dict],
                         difficulty: Dict[str, dict]) -> Counter:
    """``{pass_rate: count}`` over the pool — the shape of the ZPD band."""
    hist = Counter()
    for it in pool:
        rec = difficulty.get(it['id'])
        hist[None if rec is None else round(float(rec.get('pass_rate', 0.0)), 3)] += 1
    return hist


# ═════════════════════════════════════════════════════════════════════
#  Selection
# ═════════════════════════════════════════════════════════════════════

def select_subset(method: str,
                  pool: Sequence[dict],
                  difficulty: Optional[Dict[str, dict]] = None,
                  size: Optional[int] = None,
                  seed: int = config.SPLIT_SEED,
                  zpd_low: float = config.ZPD_LOW,
                  zpd_high: float = config.ZPD_HIGH,
                  zpd_strict: bool = False,
                  balance_tasks: bool = False,
                  match_on: str = 'items') -> Tuple[List[dict], dict]:
    """Select the memory subset.

    Args:
        method: one of :data:`METHODS`.
        pool: candidate train items. Each **must** already carry ``n_steps``
            (how many memory chunks it will contribute) — normally attached by
            ``teacher.py`` when the CoT was verified. Items without a verified
            CoT must not be in the pool: there is nothing to remember about them.
        difficulty: ``{item_id: {'pass_rate': float, ...}}`` from ``probe.py``.
            Required for ``zpd`` / ``easy_only`` / ``hard_only``.
        size: target size for the matched arms. Interpreted as an item count
            when ``match_on='items'``, else as a total chunk count.
        match_on: ``'items'`` or ``'chunks'``.

    Returns:
        ``(selected_items, info)``. ``info`` records what was actually achieved
        — including a ``shortfall`` flag when a pool could not fill the target,
        which the analysis must surface rather than silently compare unequal arms.
    """
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}, got {method!r}')
    if match_on not in ('items', 'chunks'):
        raise ValueError(f"match_on must be 'items' or 'chunks', got {match_on!r}")

    pool = list(pool)
    for it in pool:
        if 'n_steps' not in it:
            raise ValueError(
                f"pool item {it.get('id')!r} has no 'n_steps'; the pool must be "
                'built from verified teacher CoTs (see teacher.py)'
            )

    difficulty = difficulty or {}
    if method in NEEDS_DIFFICULTY and not difficulty:
        raise ValueError(
            f"method {method!r} needs a difficulty probe. Run:\n"
            f'  python -m bbeh.probe --model {config.STUDENT_MODEL}'
        )
    if method in NEEDS_SIZE and size is None:
        raise ValueError(
            f'method {method!r} is a size-matched control and needs --size '
            '(or --match-version pointing at the zpd arm)'
        )

    rng = random.Random(f'{seed}|{method}|{size}|{match_on}')

    # ─── 1. Restrict the pool to this method's eligible candidates ───
    if method == 'full':
        candidates = pool
    elif method == 'zpd':
        candidates = [
            it for it in pool
            if in_zpd(_pass_rate(difficulty, it), zpd_low, zpd_high, zpd_strict)
        ]
    elif method == 'easy_only':
        candidates = [it for it in pool if _pass_rate(difficulty, it) == 1.0]
    elif method == 'hard_only':
        candidates = [it for it in pool if _pass_rate(difficulty, it) == 0.0]
    else:  # random_matched / stratified_matched draw from the whole pool
        candidates = pool

    # ─── 2. Trim to the target size ──────────────────────────────────
    if method == 'full':
        selected = list(candidates)
    elif method == 'stratified_matched':
        selected = _take_task_proportional(candidates, size, match_on, rng, pool)
    elif method == 'zpd' and balance_tasks and size is not None:
        selected = _take_task_proportional(candidates, size, match_on, rng, candidates)
    elif method == 'zpd' and size is None:
        # The hypothesis arm defines the size; take the whole band.
        selected = list(candidates)
    else:
        selected = _take_random(candidates, size, match_on, rng)

    # ─── 3. Report ───────────────────────────────────────────────────
    n_items = len(selected)
    n_chunks = sum(int(it.get('n_steps', 0)) for it in selected)
    target = size
    achieved = n_chunks if match_on == 'chunks' else n_items
    shortfall = bool(target is not None and achieved < target)

    info = {
        'method': method,
        'match_on': match_on,
        'target_size': target,
        'n_candidates_eligible': len(candidates),
        'n_items': n_items,
        'n_chunks': n_chunks,
        'shortfall': shortfall,
        'seed': seed,
        'balance_tasks': bool(balance_tasks),
        'per_task': dict(sorted(Counter(it['task'] for it in selected).items())),
        'pass_rate_hist': {
            str(k): v for k, v in sorted(
                difficulty_histogram(selected, difficulty).items(),
                key=lambda kv: (kv[0] is None, kv[0])
            )
        },
    }
    if method in ('zpd',):
        info['zpd_band'] = {'low': zpd_low, 'high': zpd_high, 'strict': zpd_strict}

    if shortfall:
        logging.warning(
            '%s could only supply %d %s of the %d requested (eligible pool: %d). '
            'The arms are NOT size-matched — analysis must say so.',
            method, achieved, match_on, target, len(candidates)
        )

    logging.info('selector %-19s -> %4d items / %5d chunks (eligible %d)',
                 method, n_items, n_chunks, len(candidates))
    return selected, info


def _pass_rate(difficulty: Dict[str, dict], item: dict) -> Optional[float]:
    rec = difficulty.get(item['id'])
    if rec is None:
        return None
    try:
        return float(rec.get('pass_rate'))
    except (TypeError, ValueError):
        return None


def _take_random(candidates: Sequence[dict], size: Optional[int],
                 match_on: str, rng: random.Random) -> List[dict]:
    """Uniform random draw, stopping at an item count or a chunk budget."""
    pool = list(candidates)
    rng.shuffle(pool)
    if size is None:
        return pool
    if match_on == 'items':
        return pool[:size]
    out, total = [], 0
    for it in pool:
        if total >= size:
            break
        out.append(it)
        total += int(it.get('n_steps', 0))
    return out


def _take_task_proportional(candidates: Sequence[dict], size: Optional[int],
                            match_on: str, rng: random.Random,
                            reference: Sequence[dict]) -> List[dict]:
    """Draw ``size`` keeping the per-task distribution of ``reference``.

    Used for ``stratified_matched`` (reference = whole pool, i.e. preserve the
    train distribution) and for ``--balance-tasks`` ZPD (reference = the band
    itself, i.e. spread the quota evenly rather than letting two tasks dominate).
    Largest-remainder allocation, then a greedy fill for leftovers.
    """
    if size is None:
        return list(candidates)

    by_task: Dict[str, List[dict]] = defaultdict(list)
    for it in candidates:
        by_task[it['task']].append(it)
    for items in by_task.values():
        rng.shuffle(items)

    ref_counts = Counter(it['task'] for it in reference)
    ref_total = sum(ref_counts.values()) or 1

    # Chunk budgets are converted to an item target via the mean chunk size, then
    # corrected by the greedy fill below.
    if match_on == 'chunks':
        mean_steps = (sum(int(it.get('n_steps', 0)) for it in candidates)
                      / max(1, len(candidates))) or 1.0
        item_target = max(1, round(size / mean_steps))
    else:
        item_target = size

    quotas, remainders = {}, {}
    for task, items in by_task.items():
        exact = item_target * ref_counts.get(task, 0) / ref_total
        quotas[task] = min(len(items), int(exact))
        remainders[task] = exact - int(exact)

    # Largest-remainder pass.
    assigned = sum(quotas.values())
    for task in sorted(remainders, key=lambda t: (-remainders[t], t)):
        if assigned >= item_target:
            break
        if quotas[task] < len(by_task[task]):
            quotas[task] += 1
            assigned += 1

    selected = []
    for task in sorted(by_task):
        selected.extend(by_task[task][:quotas[task]])

    # Greedy fill: rounding and per-task exhaustion both leave us short.
    if match_on == 'chunks':
        total = sum(int(it.get('n_steps', 0)) for it in selected)
        if total < size:
            chosen = {id(x) for x in selected}
            leftovers = [it for it in candidates if id(it) not in chosen]
            rng.shuffle(leftovers)
            for it in leftovers:
                if total >= size:
                    break
                selected.append(it)
                total += int(it.get('n_steps', 0))
    elif len(selected) < item_target:
        chosen = {id(x) for x in selected}
        leftovers = [it for it in candidates if id(it) not in chosen]
        rng.shuffle(leftovers)
        selected.extend(leftovers[:item_target - len(selected)])

    rng.shuffle(selected)
    return selected


# ═════════════════════════════════════════════════════════════════════
#  Diagnostics
# ═════════════════════════════════════════════════════════════════════

def print_pool_report(pool: Sequence[dict], difficulty: Dict[str, dict],
                      zpd_low: float = config.ZPD_LOW,
                      zpd_high: float = config.ZPD_HIGH,
                      zpd_strict: bool = False) -> None:
    """Print the difficulty landscape — read this BEFORE building any arm.

    The thing to look for: how many items land in the band at all. If the
    student is at floor on BBEH (very plausible — it is designed to be brutal),
    almost everything sits at pass_rate 0 and the ZPD arm will be tiny. That is
    a finding about the benchmark/model pairing, and it must be known before
    spending teacher tokens.
    """
    n = len(pool)
    hist = difficulty_histogram(pool, difficulty)
    n_band = sum(1 for it in pool
                 if in_zpd(_pass_rate(difficulty, it), zpd_low, zpd_high, zpd_strict))
    n_probed = sum(1 for it in pool if _pass_rate(difficulty, it) is not None)

    print(f'pool: {n} items with a verified CoT; {n_probed} of them probed')
    print(f'ZPD band [{zpd_low}, {zpd_high}]'
          f'{" strict" if zpd_strict else ""}: {n_band} items '
          f'({n_band / n:.1%} of pool)' if n else 'pool empty')
    print('\npass_rate histogram:')
    for rate in sorted(hist, key=lambda r: (r is None, r)):
        label = 'unprobed' if rate is None else f'{rate:.2f}'
        bar = '#' * min(60, hist[rate] * 60 // max(1, n))
        print(f'  {label:>8s}  {hist[rate]:5d}  {bar}')

    print('\nper-task ZPD yield:')
    by_task_total = Counter(it['task'] for it in pool)
    by_task_band = Counter(
        it['task'] for it in pool
        if in_zpd(_pass_rate(difficulty, it), zpd_low, zpd_high, zpd_strict)
    )
    for task in sorted(by_task_total):
        tot, band = by_task_total[task], by_task_band.get(task, 0)
        print(f'  {task:34s} {band:3d}/{tot:3d}  ({band / tot:5.1%})')
