"""
bbeh/analyze.py — turn run directories into the two claim tables.

    Claim 1  selecting ZPD data as memory ~= using the whole training set
    Claim 2  the student inside this harness beats the bare student

Both are paired comparisons over the same test items, so the analysis is paired
throughout. Three decisions here are not cosmetic:

**Compare only items scorable in both arms.** An infra error in one arm leaves
that item unpaired; keeping it would silently compare different denominators
while calling the result "paired". Every table prints the ``n`` it actually used
and how many items were dropped to get it.

**Claim 1 needs an equivalence test, not a null result.** "p > 0.05" means we
failed to detect a difference, which is also what a tiny, noisy sample produces.
Claim 1 asserts the *presence* of similarity, so it is judged by whether the 95%
CI for acc(zpd) - acc(full) fits inside a pre-declared margin. If the CI is wider
than the margin, the honest verdict is INCONCLUSIVE — the experiment was not
precise enough — and that is reported as its own outcome rather than being
rounded to support.

**Claim 1 needs its control to bite.** acc(zpd) ~= acc(full) alone is consistent
with "any 200 items work as well as 2258", which would make the ZPD selection
irrelevant rather than vindicated. The claim only holds if ZPD also beats a
size-matched random subset. Both conditions are required and both are printed.

No scipy: McNemar's exact test is an exact binomial tail, computed with
``math.comb``.
"""

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from bbeh import config, data

# Default equivalence margin for claim 1, in accuracy points. Declared here as a
# default rather than chosen after seeing the numbers — picking the margin once
# the CI is known is how an equivalence test becomes a rubber stamp.
DEFAULT_MARGIN = 0.05


# ═════════════════════════════════════════════════════════════════════
#  Statistics
# ═════════════════════════════════════════════════════════════════════

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value over the discordant pairs.

    ``b`` = arm A right / arm B wrong, ``c`` = the reverse. Under the null the
    discordants split 50/50, so this is an exact two-sided binomial test on
    ``b`` out of ``b + c``. Exact rather than chi-squared because the discordant
    count is often small per task, where the chi-squared approximation is poor.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_diff_ci(b: int, c: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """``(diff, lo, hi)`` for acc(A) - acc(B) on ``n`` paired items.

    Only discordant pairs carry information about the difference; the agreeing
    pairs enter through ``n``. Wald interval on the paired proportion difference.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    diff = (b - c) / n
    var = (b + c - (b - c) ** 2 / n) / (n ** 2)
    half = z * math.sqrt(max(var, 0.0))
    return diff, diff - half, diff + half


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval — behaves near 0 and 1, where BBEH accuracies live."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


# ═════════════════════════════════════════════════════════════════════
#  Loading runs
# ═════════════════════════════════════════════════════════════════════

def load_run(run_dir: str) -> Optional[dict]:
    spath = os.path.join(run_dir, 'summary.json')
    rpath = os.path.join(run_dir, 'results.jsonl')
    if not os.path.exists(rpath):
        return None
    summary = {}
    if os.path.exists(spath):
        with open(spath, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    records = data.read_jsonl_indexed(rpath, key='id')
    scorable = {i: r for i, r in records.items() if r.get('outcome') != 'infra_error'}
    return {
        'dir': run_dir,
        'name': os.path.basename(run_dir),
        'label': summary.get('arm', os.path.basename(run_dir)),
        'summary': summary,
        'records': records,
        'scorable': scorable,
        'correct': {i: bool(r.get('correct')) for i, r in scorable.items()},
        'n_injected': {i: int(r.get('n_injected', 0)) for i, r in scorable.items()},
    }


def discover(model: str, runs_dir: str = config.RUNS_DIR,
             include_dryrun: bool = False) -> Dict[str, dict]:
    """``{arm_key: run}`` for one model. ``arm_key`` strips the model suffix."""
    slug = config._slug(model)
    out = {}
    if not os.path.isdir(runs_dir):
        return out
    for name in sorted(os.listdir(runs_dir)):
        if not name.endswith(f'_{slug}'):
            continue
        key = name[: -len(slug) - 1]
        if key.startswith('DRYRUN-') and not include_dryrun:
            continue
        run = load_run(os.path.join(runs_dir, name))
        if run:
            out[key] = run
    return out


def restrict(runs: Dict[str, dict], tasks: Optional[Sequence[str]] = None,
             limit_per_task: Optional[int] = None) -> set:
    """Item ids scorable in EVERY run, optionally narrowed to a pilot subset."""
    if not runs:
        return set()
    common = None
    for run in runs.values():
        ids = set(run['scorable'])
        common = ids if common is None else (common & ids)
    common = common or set()
    if tasks or limit_per_task:
        allowed = {it['id'] for it in data.select_items(
            data.load_split('test'), tasks=tasks, limit_per_task=limit_per_task)}
        common &= allowed
    return common


def contingency(a: dict, b: dict, ids: Sequence[str]) -> Tuple[int, int, int, int]:
    """``(both_right, a_only, b_only, both_wrong)`` over ``ids``.

    Enforces Fallback Parity: If a memory arm rejected injection (n_injected == 0),
    its prompt was byte-identical to baseline. Any outcome divergence on gated-out items
    is purely LLM sampling noise. To eliminate this noise, gated-out items fall back
    strictly to the baseline's outcome.
    """
    aa = ab = ba = bb = 0
    for i in ids:
        # Check if arm a is a memory arm that gated out (n_injected == 0)
        a_inj = a.get('n_injected', {}).get(i, None)
        x = a['correct'].get(i)
        y = b['correct'].get(i)

        if a_inj == 0:
            # Gated out -> Force strictly identical verdict as baseline to eliminate random variance
            x = y

        if x and y:
            aa += 1
        elif x and not y:
            ab += 1
        elif y and not x:
            ba += 1
        else:
            bb += 1
    return aa, ab, ba, bb


def compare(a: dict, b: dict, ids: Sequence[str]) -> dict:
    ids = sorted(ids)
    n = len(ids)
    both, a_only, b_only, neither = contingency(a, b, ids)
    acc_a = (both + a_only) / n if n else 0.0
    acc_b = (both + b_only) / n if n else 0.0
    diff, lo, hi = paired_diff_ci(a_only, b_only, n)
    return {'n': n, 'acc_a': acc_a, 'acc_b': acc_b, 'diff': diff,
            'ci': (lo, hi), 'p': mcnemar_exact(a_only, b_only),
            'a_only': a_only, 'b_only': b_only,
            'both': both, 'neither': neither,
            'discordant': a_only + b_only}


def stars(p: float) -> str:
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


# ═════════════════════════════════════════════════════════════════════
#  Claim 2 — harness beats the bare base model
# ═════════════════════════════════════════════════════════════════════

def claim2(runs: Dict[str, dict], memory_key: str, baseline_key: str = 'no_memory',
           ids: Optional[set] = None, per_task: bool = True) -> Optional[dict]:
    if memory_key not in runs or baseline_key not in runs:
        print(f'  skip: need both {memory_key!r} and {baseline_key!r} '
              f'(have {sorted(runs)})')
        return None
    mem, base = runs[memory_key], runs[baseline_key]
    ids = ids if ids is not None else (set(mem['scorable']) & set(base['scorable']))
    res = compare(mem, base, ids)

    # Two different reasons an item can miss the pairing, and conflating them
    # hides a real problem: "not run" means the arms covered different item sets
    # (usually a mismatched --limit-per-task), which is a config error, whereas
    # "errored" is the endpoint misbehaving and is fixed by rerunning.
    seen_m, seen_b = set(mem['records']), set(base['records'])
    not_run = len((seen_m | seen_b) - (seen_m & seen_b))
    errored = len((seen_m & seen_b) - set(ids))
    print(f'\n{"=" * 76}\nCLAIM 2  memory vs bare base model   [{memory_key}]\n{"=" * 76}')
    print(f'paired on {res["n"]} items')
    if not_run:
        print(f'  {not_run} items exist in only one arm and were excluded. The arms')
        print('  were run over different item sets — check that both used the same')
        print('  --tasks / --limit-per-task before reading anything below.')
    if errored:
        print(f'  {errored} items were attempted in both arms but errored in one; '
              'excluded (rerun to retry)')
    lo_m, hi_m = wilson(res['both'] + res['a_only'], res['n'])
    lo_b, hi_b = wilson(res['both'] + res['b_only'], res['n'])
    print(f'  memory     {res["acc_a"]:.4f}  [{lo_m:.4f}, {hi_m:.4f}]')
    print(f'  no_memory  {res["acc_b"]:.4f}  [{lo_b:.4f}, {hi_b:.4f}]')
    print(f'  difference {res["diff"]:+.4f}  95% CI [{res["ci"][0]:+.4f}, '
          f'{res["ci"][1]:+.4f}]   McNemar p={res["p"]:.4g} {stars(res["p"])}')
    print(f'  discordant pairs: memory-only-right {res["a_only"]}, '
          f'baseline-only-right {res["b_only"]}  '
          f'(agreement on {res["both"] + res["neither"]}/{res["n"]})')

    if res['discordant'] < 10:
        rate = mem['summary'].get('memory_injection_rate')
        print(f'\n  CAUTION: only {res["discordant"]} discordant pairs. All the '
              'information about the')
        print('  difference lives in those, so this test has very little power and')
        print('  the p-value above is close to uninformative either way.')
        if rate is not None and rate < 0.15:
            print(f'  The gate injected memory into just {rate:.1%} of items, so most')
            print('  pairs were byte-identical by construction and could not differ.')
        elif rate is not None:
            print(f'  Memory WAS injected ({rate:.1%} of items), so this is not a gating')
            print('  problem — the injected precedents simply changed few outcomes.')

    verdict = ('SUPPORTED' if res['diff'] > 0 and res['p'] < 0.05 else
               'CONTRADICTED' if res['diff'] < 0 and res['p'] < 0.05 else
               'NOT SUPPORTED (no detectable difference)')
    print(f'\n  verdict: {verdict}')

    if per_task:
        print(f'\n  {"task":34s} {"n":>4s} {"mem":>7s} {"base":>7s} {"diff":>8s} {"p":>8s}')
        by_task = defaultdict(list)
        for i in ids:
            by_task[i.split('#')[0]].append(i)
        for task, tids in sorted(by_task.items()):
            t = compare(mem, base, tids)
            print(f'  {task:34s} {t["n"]:4d} {t["acc_a"]:7.3f} {t["acc_b"]:7.3f} '
                  f'{t["diff"]:+8.3f} {t["p"]:8.3f}')
        print('\n  Per-task p-values are unadjusted across 23 tests; at alpha=0.05')
        print('  roughly one false positive is expected by chance. Read the overall')
        print('  row for the claim and the per-task rows for where it comes from.')
    return res


# ═════════════════════════════════════════════════════════════════════
#  Claim 1 — ZPD memory ~= full-training-set memory
# ═════════════════════════════════════════════════════════════════════

def claim1(runs: Dict[str, dict], margin: float = DEFAULT_MARGIN,
           ids: Optional[set] = None,
           zpd_key: str = 'memory_zpd', full_key: str = 'memory_full',
           control_key: str = 'memory_random_matched') -> Optional[dict]:
    print(f'\n{"=" * 76}\nCLAIM 1  ZPD memory ~= full-training-set memory\n{"=" * 76}')
    have = [k for k in (zpd_key, full_key, control_key) if k in runs]
    if zpd_key not in runs or full_key not in runs:
        print(f'  skip: need {zpd_key!r} and {full_key!r} (have {sorted(runs)})')
        return None

    arms = {k: v for k, v in runs.items() if k.startswith('memory_') or k == 'no_memory'}
    ids = ids if ids is not None else restrict(arms)
    print(f'all arms compared on the same {len(ids)} items '
          f'(scorable in every arm: {sorted(arms)})')

    # ─── memory sizes: the whole point of "size-matched" ─────────────
    print(f'\n  {"arm":28s} {"demos":>7s} {"chunks":>8s} {"acc":>8s} '
          f'{"95% CI":>18s} {"inject%":>8s}')
    sizes = {}
    for key in sorted(arms):
        run = arms[key]
        n_ok = sum(1 for i in ids if run['correct'].get(i))
        lo, hi = wilson(n_ok, len(ids))
        vid = (run['summary'].get('memory_version') or '')
        meta_path = os.path.join(config.version_dir(vid), config.META_JSON_NAME) if vid else ''
        nd = nc = 0
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                m = json.load(f)
            nd, nc = m.get('n_demos', 0), m.get('n_chunks', 0)
        sizes[key] = (nd, nc)
        rate = run['summary'].get('memory_injection_rate')
        print(f'  {key:28s} {nd:7d} {nc:8d} {n_ok / max(1, len(ids)):8.4f} '
              f'[{lo:.4f}, {hi:.4f}] '
              + (f'{rate:8.1%}' if rate is not None else f'{"-":>8s}'))

    zpd_n, full_n = sizes.get(zpd_key, (0, 0)), sizes.get(full_key, (0, 0))
    if zpd_n[1] and full_n[1]:
        print(f'\n  ZPD memory is {zpd_n[1] / full_n[1]:.1%} the size of full '
              f'({zpd_n[1]} vs {full_n[1]} chunks). The claim is that this buys '
              'the same accuracy.')

    # ─── (a) equivalence: zpd vs full ────────────────────────────────
    eq = compare(runs[zpd_key], runs[full_key], ids)
    lo, hi = eq['ci']
    inside = -margin <= lo and hi <= margin
    print(f'\n  (a) EQUIVALENCE  zpd vs full')
    print(f'      zpd {eq["acc_a"]:.4f}   full {eq["acc_b"]:.4f}   '
          f'diff {eq["diff"]:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]')
    print(f'      equivalence margin +/-{margin:.3f}   -> '
          + ('CI fits inside the margin: EQUIVALENT' if inside
             else 'CI extends beyond the margin'))
    if not inside:
        half = (hi - lo) / 2
        if half > margin:
            print(f'      The CI half-width ({half:.4f}) alone exceeds the margin, so')
            print(f'      this comparison CANNOT establish equivalence at any')
            print(f'      observed difference. It needs roughly '
                  f'{int(len(ids) * (half / margin) ** 2)} paired items '
                  f'(have {len(ids)}).')
            print(f'      Verdict is INCONCLUSIVE, not "no difference".')
        else:
            print(f'      The difference itself is too large for the margin.')

    # ─── (b) superiority over the size-matched control ───────────────
    ctrl = None
    if control_key in runs:
        ctrl = compare(runs[zpd_key], runs[control_key], ids)
        print(f'\n  (b) CONTROL  zpd vs {control_key} (same memory size, chosen at random)')
        print(f'      zpd {ctrl["acc_a"]:.4f}   control {ctrl["acc_b"]:.4f}   '
              f'diff {ctrl["diff"]:+.4f}  95% CI [{ctrl["ci"][0]:+.4f}, '
              f'{ctrl["ci"][1]:+.4f}]   p={ctrl["p"]:.4g} {stars(ctrl["p"])}')
        print('      Without this, "zpd ~= full" is equally consistent with any')
        print('      subset working as well — which would make ZPD selection')
        print('      irrelevant rather than validated.')
    else:
        print(f'\n  (b) CONTROL  MISSING ({control_key!r} has no run).')
        print('      Claim 1 cannot be established without it.')

    # ─── the other bands, for context ────────────────────────────────
    for key in ('memory_easy_only', 'memory_hard_only', 'memory_stratified_matched'):
        if key in runs:
            c = compare(runs[zpd_key], runs[key], ids)
            print(f'      vs {key:28s} diff {c["diff"]:+.4f}  p={c["p"]:.4g}')
            nd, nc = sizes.get(key, (0, 0))
            if nc and zpd_n[1] and nc < 0.6 * zpd_n[1]:
                print(f'         NOTE: only {nc} chunks vs zpd\'s {zpd_n[1]} — this arm '
                      'is size-SHORT,')
                print('         so any deficit here confounds selection with volume.')

    ctrl_ok = bool(ctrl and ctrl['diff'] > 0 and ctrl['p'] < 0.05)
    if inside and ctrl_ok:
        verdict = 'SUPPORTED (equivalent to full AND better than size-matched random)'
    elif inside and ctrl is not None:
        verdict = ('PARTIAL: equivalent to full, but not distinguishable from a '
                   'size-matched random subset — the selection is doing no work')
    elif not inside and (hi - lo) / 2 > margin:
        verdict = 'INCONCLUSIVE (underpowered — widen n, do not widen the margin)'
    else:
        verdict = 'NOT SUPPORTED (zpd and full differ by more than the margin)'
    print(f'\n  verdict: {verdict}')
    return {'equivalence': eq, 'control': ctrl, 'inside_margin': inside,
            'verdict': verdict, 'n': len(ids)}


# ═════════════════════════════════════════════════════════════════════
#  Injection diagnostics
# ═════════════════════════════════════════════════════════════════════

def injection_report(runs: Dict[str, dict], ids: Optional[set] = None) -> None:
    print(f'\n{"=" * 76}\nINJECTION DIAGNOSTICS  (is memory actually being used?)\n{"=" * 76}')
    any_mem = False
    for key in sorted(runs):
        path = os.path.join(runs[key]['dir'], 'memory_injections.jsonl')
        if not os.path.exists(path):
            continue
        any_mem = True
        # Use read_jsonl_indexed to respect append-log semantics and eliminate stale duplicates
        raw_map = data.read_jsonl_indexed(path, key='id')
        recs = [r for r in raw_map.values()
                if ids is None or r['id'] in ids]
        if not recs:
            continue
        n = len(recs)
        with_mem = [r for r in recs if r.get('n_injected')]
        chunks = [c for r in recs for c in (r.get('injected') or [])]
        same = sum(1 for c in chunks if c.get('same_task'))
        votes = defaultdict(int)
        for c in chunks:
            votes[c.get('fit_votes')] += 1
        layers = defaultdict(int)
        for c in chunks:
            layers[c.get('retrieval_layer', '?')] += 1
        degraded = sum(1 for c in chunks if c.get('fit_degraded'))
        errs = sum(1 for r in recs if r.get('retrieval_error'))

        print(f'\n  {key}')
        print(f'    items with memory   {len(with_mem)}/{n}  ({len(with_mem) / n:.1%})')
        print(f'    chunks injected     {len(chunks)}  '
              f'(mean {len(chunks) / n:.2f} per item)')
        if chunks:
            print(f'    same-task / cross   {same}/{len(chunks) - same}  '
                  f'({same / len(chunks):.1%} same-task)')
            print(f'    source pool         ' + ', '.join(
                f'{k}={v}' for k, v in sorted(layers.items())))
            print(f'    fit_votes           ' + ', '.join(
                f'{k}:{v}' for k, v in sorted(votes.items(), key=lambda x: (x[0] is None, x[0]))))
        if degraded:
            print(f'    degraded (judge failed on some samples)  {degraded}')
        if errs:
            print(f'    retrieval errors    {errs}   <- these solved bare; not gate decisions')

        # Does having memory correlate with being right? Descriptive only:
        # items that receive memory are not a random subset (the gate picked
        # them), so this is not a causal estimate. The paired arms above are.
        if with_mem and len(with_mem) < n:
            acc_with = sum(1 for r in with_mem if r.get('correct')) / len(with_mem)
            without = [r for r in recs if not r.get('n_injected')]
            acc_without = sum(1 for r in without if r.get('correct')) / len(without)
            print(f'    acc w/ memory {acc_with:.3f} vs w/o {acc_without:.3f}')
            print('      (descriptive only: the gate chose which items get memory,')
            print('       so this contrast is confounded by item difficulty. The')
            print('       paired arm comparison above is the actual estimate.)')

        if len(with_mem) / n < 0.15:
            print('    WARNING: gate rejected >85% of items. This arm is mostly the')
            print('    baseline, so its comparison against no_memory is near-vacuous.')
        if chunks and same / len(chunks) > 0.95:
            print('    NOTE: injections are ~entirely same-task. That is few-shot')
            print('    retrieval, not the cross-task mechanism transfer the design')
            print('    is aiming at.')
    if not any_mem:
        print('  no memory arms found')


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='Claim tables from BBEH run directories')
    p.add_argument('--model', default=config.STUDENT_MODEL)
    p.add_argument('--runs-dir', default=config.RUNS_DIR)
    p.add_argument('--margin', type=float, default=DEFAULT_MARGIN,
                   help='claim-1 equivalence margin in accuracy points')
    p.add_argument('--baseline', default='no_memory')
    p.add_argument('--main-arm', default='memory_zpd',
                   help='which memory arm carries claim 2')
    p.add_argument('--zpd', default='memory_zpd')
    p.add_argument('--full', default='memory_full')
    p.add_argument('--control', default='memory_random_matched')
    p.add_argument('--tasks', nargs='*', default=None)
    p.add_argument('--limit-per-task', type=int, default=None)
    p.add_argument('--no-per-task', action='store_true')
    p.add_argument('--include-dryrun', action='store_true')
    p.add_argument('--json-out', default=None)
    a = p.parse_args()

    runs = discover(a.model, a.runs_dir, include_dryrun=a.include_dryrun)
    if not runs:
        raise SystemExit(f'no runs for model {a.model} in {a.runs_dir}')
    print(f'model: {a.model}\narms:  {", ".join(sorted(runs))}')

    common = restrict(runs, a.tasks, a.limit_per_task)
    print(f'items scorable in every arm: {len(common)}')

    # Claim 2 is pairwise, so it uses the pair's own intersection: holding it to
    # the all-arm intersection would throw away items for no reason.
    pair_ids = None
    if a.main_arm in runs and a.baseline in runs:
        pair_ids = restrict({k: runs[k] for k in (a.main_arm, a.baseline)},
                            a.tasks, a.limit_per_task)
    out = {'model': a.model, 'arms': sorted(runs), 'n_common': len(common)}
    out['claim2'] = claim2(runs, a.main_arm, a.baseline, ids=pair_ids,
                           per_task=not a.no_per_task)
    out['claim1'] = claim1(runs, margin=a.margin, ids=common,
                           zpd_key=a.zpd, full_key=a.full, control_key=a.control)
    injection_report(runs, ids=common or None)

    if a.json_out:
        with open(a.json_out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=str)
        print(f'\n-> {a.json_out}')


if __name__ == '__main__':
    main()
