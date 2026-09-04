"""
bbeh/selftest_analyze.py — offline checks for the claim statistics.

Every other selftest in this harness guards data plumbing. This one guards the
arithmetic that turns runs into claims, which is the more dangerous surface: a
broken pipeline usually crashes, whereas a broken confidence interval prints a
clean table with a wrong verdict on it and nothing looks amiss.

Four properties, in rough order of how badly a bug would hurt:

  1. **McNemar's p is exact**, checked against hand-computable values. If it were
     merely approximate, the per-task tables (often <10 discordant pairs) would
     be systematically anti-conservative.
  2. **The 95% CI actually covers 95% of the time**, verified by simulation
     rather than by inspection. The whole claim-1 equivalence verdict is a
     statement about this interval, so if its coverage is really 80% then
     "equivalent" means much less than it appears to.
  3. **Underpowered data returns INCONCLUSIVE, never SUPPORTED.** This is the
     failure mode the equivalence framing exists to prevent — a small, noisy
     experiment "showing" that ZPD matches the full set.
  4. **Accuracies are recomputed on the paired subset**, not read from
     summary.json. Arms that errored on different items have different
     denominators there, and quoting those side by side is an unpaired
     comparison wearing a paired label.
"""

import json
import os
import random
import shutil
import sys
import tempfile

from bbeh import analyze as A

_FAILS = []


def check(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f'   {detail}' if detail else ''))
    if not cond:
        _FAILS.append(name)


# ═════════════════════════════════════════════════════════════════════

def test_mcnemar():
    print('\n=== McNemar exact test ===')
    # b=c: perfectly balanced discordance, no evidence either way.
    check('balanced discordance -> p = 1', abs(A.mcnemar_exact(5, 5) - 1.0) < 1e-12)
    check('no discordant pairs -> p = 1', A.mcnemar_exact(0, 0) == 1.0)
    # All 5 discordants favour one arm: two-sided p = 2 * (1/2)^5.
    check('5-0 split -> p = 2/32', abs(A.mcnemar_exact(5, 0) - 0.0625) < 1e-12,
          f'{A.mcnemar_exact(5, 0):.6f}')
    # 10-0: 2 * (1/2)^10.
    check('10-0 split -> p = 2/1024',
          abs(A.mcnemar_exact(10, 0) - 2 / 1024) < 1e-12,
          f'{A.mcnemar_exact(10, 0):.6f}')
    # 8-2 out of 10: 2 * P(X<=2) = 2 * (1+10+45)/1024
    check('8-2 split matches the binomial tail',
          abs(A.mcnemar_exact(8, 2) - 2 * 56 / 1024) < 1e-12,
          f'{A.mcnemar_exact(8, 2):.6f}')
    check('symmetric in its arguments',
          all(abs(A.mcnemar_exact(b, c) - A.mcnemar_exact(c, b)) < 1e-15
              for b, c in ((3, 7), (0, 4), (11, 2))))
    check('p never exceeds 1 even at the smallest n',
          all(A.mcnemar_exact(b, c) <= 1.0
              for b in range(4) for c in range(4)))
    # Direction is the caller's job; the test itself is two-sided and so must
    # not care which arm won.
    check('a large one-sided imbalance is significant',
          A.mcnemar_exact(20, 4) < 0.01, f'{A.mcnemar_exact(20, 4):.5f}')


def test_ci_coverage():
    print('\n=== CI coverage (simulation, not inspection) ===')
    rng = random.Random(20260814)
    # Paired binary data: each item independently lands in one of four cells.
    # True difference = p_a_only - p_b_only.
    cells = [('both', 0.30), ('a_only', 0.15), ('b_only', 0.08), ('neither', 0.47)]
    true_diff = 0.15 - 0.08
    n, trials = 300, 2000
    covered = 0
    widths = []
    for _ in range(trials):
        counts = {k: 0 for k, _ in cells}
        for _ in range(n):
            u, acc = rng.random(), 0.0
            for k, p in cells:
                acc += p
                if u <= acc:
                    counts[k] += 1
                    break
        d, lo, hi = A.paired_diff_ci(counts['a_only'], counts['b_only'], n)
        covered += lo <= true_diff <= hi
        widths.append(hi - lo)
    rate = covered / trials
    check('95% CI covers the true difference ~95% of the time',
          0.93 <= rate <= 0.97, f'{rate:.3f} over {trials} sims (n={n})')
    check('interval width is plausible, not degenerate',
          0.02 < sum(widths) / len(widths) < 0.20,
          f'mean width {sum(widths) / len(widths):.4f}')

    d, lo, hi = A.paired_diff_ci(12, 4, 100)
    check('diff equals (b-c)/n', abs(d - 0.08) < 1e-12)
    check('CI is centred on the difference', abs((lo + hi) / 2 - d) < 1e-12)
    check('zero discordance gives a zero-width interval at 0',
          A.paired_diff_ci(0, 0, 50) == (0.0, 0.0, 0.0))

    lo, hi = A.wilson(0, 40)
    check('Wilson stays inside [0,1] at the boundary',
          lo == 0.0 and 0 < hi < 0.15, f'[{lo:.4f}, {hi:.4f}]')
    lo, hi = A.wilson(40, 40)
    check('Wilson handles p=1 without exploding', hi == 1.0 and 0.85 < lo < 1.0)


# ═════════════════════════════════════════════════════════════════════
#  Fabricated runs, so the verdict logic can be exercised end to end
# ═════════════════════════════════════════════════════════════════════

def _write_run(root, arm_key, model, correct_map, version=None, meta=None,
               injections=None):
    d = os.path.join(root, f'{arm_key}_{model}')
    os.makedirs(d, exist_ok=True)
    recs = []
    for i, (item_id, val) in enumerate(sorted(correct_map.items())):
        rec = {'id': item_id, 'task': item_id.split('#')[0],
               'outcome': 'infra_error' if val is None else 'ok',
               'correct': bool(val), 'response': '' if val is None else 'x',
               'n_injected': 1 if injections else 0}
        recs.append(rec)
    with open(os.path.join(d, 'results.jsonl'), 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    with open(os.path.join(d, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'arm': arm_key, 'model': model, 'memory_version': version,
                   'memory_injection_rate': 1.0 if injections else 0.0}, f)
    if version and meta:
        vdir = os.path.join(root, '_versions', version)
        os.makedirs(vdir, exist_ok=True)
        with open(os.path.join(vdir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f)
    return d


class _Prefixed:
    """stdout wrapper that stamps a tag onto every non-blank line."""

    def __init__(self, stream, tag):
        self._s, self._tag, self._bol = stream, tag, True

    def write(self, text):
        for part in text.splitlines(keepends=True):
            if self._bol and part.strip():
                self._s.write(self._tag)
            self._bol = part.endswith('\n')
            self._s.write(part)

    def flush(self):
        self._s.flush()


def tagged(fn, *args, **kwargs):
    """Call ``fn``, labelling everything it prints as fabricated.

    ``analyze.claim1`` and ``claim2`` print a complete results table as a side
    effect, so exercising the verdict logic dumps four real-looking tables —
    accuracies, confidence intervals, stars, a verdict line — built from
    invented data. Lifted out of context into a chat message or a log excerpt
    they are indistinguishable from a finished experiment, and someone will
    read a claim off them. Stamping at source means any excerpt, however short,
    carries the warning with it; a banner above the block would not survive
    being copied from the middle.
    """
    real = sys.stdout
    sys.stdout = _Prefixed(real, 'FABRICATED| ')
    try:
        return fn(*args, **kwargs)
    finally:
        sys.stdout = real


def _synth(n, acc, seed, flip=0):
    """n items with roughly ``acc`` correct; ``flip`` items forced wrong."""
    rng = random.Random(seed)
    ids = [f'task_a#{i:04d}' for i in range(n)]
    out = {i: rng.random() < acc for i in ids}
    for i in ids[:flip]:
        out[i] = False
    return out


def test_verdicts():
    print('\n=== claim verdicts on fabricated runs ===')
    print('  NOTE: the tables below are computed from INVENTED runs. Every')
    print('  accuracy, CI and verdict in them is a test fixture, not a result.')
    print('  Lines from them are tagged "FABRICATED|".')
    root = tempfile.mkdtemp(prefix='bbeh_analyze_')
    model = 'testmodel'
    orig_vd = A.config.version_dir
    A.config.version_dir = lambda v: os.path.join(root, '_versions', v)
    try:
        # --- Case 1: genuinely equivalent, large n -> SUPPORTED --------
        n = 1200
        rng = random.Random(7)
        ids = [f'task_a#{i:04d}' for i in range(n)]
        zpd = {i: rng.random() < 0.50 for i in ids}
        # full agrees with zpd except for a symmetric handful of flips: same
        # accuracy in expectation, which is exactly what "equivalent" means.
        full = dict(zpd)
        for i in ids[:40]:
            full[i] = not full[i]
        for i in ids[40:80]:
            full[i] = zpd[i]
        ctrl = {i: rng.random() < 0.42 for i in ids}     # size-matched, worse
        base = {i: rng.random() < 0.40 for i in ids}
        _write_run(root, 'memory_zpd', model, zpd, 'zpd',
                   {'n_demos': 200, 'n_chunks': 949}, injections=True)
        _write_run(root, 'memory_full', model, full, 'full',
                   {'n_demos': 2258, 'n_chunks': 9800}, injections=True)
        _write_run(root, 'memory_random_matched', model, ctrl, 'rand',
                   {'n_demos': 200, 'n_chunks': 951}, injections=True)
        _write_run(root, 'no_memory', model, base)

        runs = A.discover(model, root)
        check('discover finds all four arms', len(runs) == 4, sorted(runs))
        common = A.restrict(runs)
        check('restrict returns the shared item set', len(common) == n)

        res1 = tagged(A.claim1, runs, margin=0.05, ids=common)
        check('equivalent + control beaten -> SUPPORTED',
              res1 and res1['verdict'].startswith('SUPPORTED'), res1['verdict'])

        # --- Case 2: same data, 40 items -> INCONCLUSIVE ---------------
        small = set(sorted(common)[:40])
        res2 = tagged(A.claim1, runs, margin=0.05, ids=small)
        check('same effect at n=40 -> INCONCLUSIVE, never SUPPORTED',
              res2 and res2['verdict'].startswith('INCONCLUSIVE'), res2['verdict'])

        # --- Case 3: zpd genuinely worse -> NOT SUPPORTED --------------
        worse = dict(zpd)
        for i in ids[:250]:
            worse[i] = False
        shutil.rmtree(os.path.join(root, f'memory_zpd_{model}'))
        _write_run(root, 'memory_zpd', model, worse, 'zpd',
                   {'n_demos': 200, 'n_chunks': 949}, injections=True)
        runs3 = A.discover(model, root)
        res3 = tagged(A.claim1, runs3, margin=0.05, ids=common)
        check('a real 20-point deficit -> NOT SUPPORTED',
              res3 and res3['verdict'].startswith('NOT SUPPORTED'), res3['verdict'])

        # --- Case 4: control not beaten -> PARTIAL ---------------------
        shutil.rmtree(os.path.join(root, f'memory_zpd_{model}'))
        shutil.rmtree(os.path.join(root, f'memory_random_matched_{model}'))
        _write_run(root, 'memory_zpd', model, zpd, 'zpd',
                   {'n_demos': 200, 'n_chunks': 949}, injections=True)
        _write_run(root, 'memory_random_matched', model, dict(zpd), 'rand',
                   {'n_demos': 200, 'n_chunks': 951}, injections=True)
        runs4 = A.discover(model, root)
        res4 = tagged(A.claim1, runs4, margin=0.05, ids=common)
        check('equivalent to full but tied with random -> PARTIAL',
              res4 and res4['verdict'].startswith('PARTIAL'), res4['verdict'])

        # --- Case 5: paired subset, not summary.json -------------------
        # zpd errors on 300 items that no_memory answered. summary-level
        # accuracies would be computed over different denominators.
        holed = {i: (None if k < 300 else v)
                 for k, (i, v) in enumerate(sorted(zpd.items()))}
        shutil.rmtree(os.path.join(root, f'memory_zpd_{model}'))
        _write_run(root, 'memory_zpd', model, holed, 'zpd',
                   {'n_demos': 200, 'n_chunks': 949}, injections=True)
        runs5 = A.discover(model, root)
        pair = A.restrict({k: runs5[k] for k in ('memory_zpd', 'no_memory')})
        check('errored items are dropped from the pairing',
              len(pair) == n - 300, f'{len(pair)} paired of {n}')
        r = A.compare(runs5['memory_zpd'], runs5['no_memory'], pair)
        expect = sum(1 for i in pair if zpd[i]) / len(pair)
        check('accuracy is recomputed on the paired subset',
              abs(r['acc_a'] - expect) < 1e-12, f'{r["acc_a"]:.4f}')
        check('both arms use the same denominator',
              r['both'] + r['a_only'] + r['b_only'] + r['neither'] == len(pair))

        res6 = tagged(A.claim2, runs5, 'memory_zpd', 'no_memory',
                      ids=pair, per_task=False)
        check('claim 2 reports the paired n, not the run size',
              res6['n'] == n - 300)
    finally:
        A.config.version_dir = orig_vd
        shutil.rmtree(root, ignore_errors=True)


def test_margin_is_not_a_dial():
    print('\n=== the equivalence margin must bind ===')
    # Identical arms, but only 20 items: the CI is far too wide to conclude
    # anything, and no amount of "the difference is 0.0" should change that.
    d, lo, hi = A.paired_diff_ci(1, 1, 20)
    half = (hi - lo) / 2
    check('a tiny sample cannot fit inside a 0.05 margin',
          half > 0.05, f'half-width {half:.4f} vs margin 0.050')
    # And the required-n hint should point somewhere sane.
    need = int(20 * (half / 0.05) ** 2)
    check('required-n hint scales as 1/margin^2 and exceeds the current n',
          need > 20, f'suggests ~{need} items')


def main() -> int:
    print(f'{"=" * 68}\nanalysis selftest (offline)\n{"=" * 68}')
    test_mcnemar()
    test_ci_coverage()
    test_margin_is_not_a_dial()
    test_verdicts()
    print(f'\n{"=" * 68}')
    if _FAILS:
        print(f'{len(_FAILS)} FAILURES: {_FAILS}')
        return 1
    print('all analysis checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
