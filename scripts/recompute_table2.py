"""Recompute the counterfactual placebo table (manuscript Table 2) from artifacts.

Supersedes ``eval_cf_fast.py``, which hardcodes absolute Linux paths and reports
only the four accuracies. This version:

  * uses repo-relative paths, so it runs wherever the repo is checked out;
  * excludes infra errors from the paired comparison instead of scoring them 0,
    matching the policy in ``run.py`` / ``reasoner.py``;
  * runs the statistics the manuscript's appendix promises but never reports:
    exact McNemar per gate subset, a two-sided gate-label permutation test on
    the selectivity contrast, and stratified paired bootstrap 95% CIs;
  * computes the table under BOTH definitions of the gate label, because they
    disagree on 129 of 690 queries:
      manifest  — ``gate_decision`` in counterfactual_manifest.json (what the
                  manuscript's N=192/498 split reflects)
      run       — whether the real memory_zpd run actually injected anything
                  (n_injected > 0), i.e. the gate the deployed system applied

Both are reported because "we freeze the gate decision" is only well defined if
the two agree, and they do not. Which one belongs in the paper is a claim about
what was frozen, not something this script can settle.

    python -m bbeh.work.recompute_table2
    python bbeh/work/recompute_table2.py --permutations 20000 --bootstrap 10000
"""

import argparse
import io
import json
import math
import os
import random
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANIFEST = os.path.join(BASE, 'work', 'frozen_experiment_manifest',
                        'counterfactual_manifest.json')
REAL_RUN = os.path.join(BASE, 'runs',
                        'memory_zpd_counterfactual_real_top1_gemini-3.5-flash')
PLACEBO_RUN = os.path.join(BASE, 'runs',
                           'memory_zpd_counterfactual_placebo_gemini-3.5-flash')
ZPD_RUN = os.path.join(BASE, 'runs', 'memory_zpd_gemini-3.5-flash')


# ═════════════════════════════════════════════════════════════════════
#  IO
# ═════════════════════════════════════════════════════════════════════

def read_jsonl(path, key='id'):
    """Index a jsonl by ``key``, last record wins (matches append-then-resume)."""
    out = {}
    with io.open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec[key]] = rec
    return out


# ═════════════════════════════════════════════════════════════════════
#  Statistics (no scipy; the harness does not depend on it)
# ═════════════════════════════════════════════════════════════════════

def mcnemar_exact(b, c):
    """Two-sided exact McNemar p over the discordant pairs.

    b = real correct / placebo wrong, c = the reverse. Same implementation as
    ``bbeh/analyze.py``: an exact binomial tail, not a chi-squared approximation,
    because discordant counts here are small.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def permutation_p(d_accept, d_reject, n_perm, seed=12345):
    """Permutation test on S = mean(d_accept) - mean(d_reject).

    Shuffles the gate LABEL across the pooled paired effects while preserving
    the observed subset sizes, which is the null the manuscript states: Accept
    and Reject have equal mean paired effects.

    Returns ``(p_two_sided, p_one_sided, observed_S)``. Both tails are reported
    because they differ by roughly 2x here and the appendix specifies two-sided;
    printing only one invites quoting whichever is smaller.
    """
    pooled = list(d_accept) + list(d_reject)
    n_a = len(d_accept)
    obs = _mean(d_accept) - _mean(d_reject)
    rng = random.Random(seed)
    two = one = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        s = _mean(pooled[:n_a]) - _mean(pooled[n_a:])
        if abs(s) >= abs(obs) - 1e-12:
            two += 1
        if s >= obs - 1e-12:
            one += 1
    # Add-one correction: a permutation p is never exactly 0.
    return ((two + 1) / float(n_perm + 1),
            (one + 1) / float(n_perm + 1), obs)


def bootstrap_ci(d_accept, d_reject, n_boot, seed=999, alpha=0.05):
    """Stratified paired bootstrap CIs for Δ_accept, Δ_reject and S.

    Resamples queries WITHIN each gate subset, so the subset sizes stay at their
    observed values and the two Δ's stay paired with the S they produce.
    """
    rng = random.Random(seed)
    n_a, n_r = len(d_accept), len(d_reject)
    da, dr, ds = [], [], []
    for _ in range(n_boot):
        ba = [d_accept[rng.randrange(n_a)] for _ in range(n_a)] if n_a else []
        br = [d_reject[rng.randrange(n_r)] for _ in range(n_r)] if n_r else []
        ma, mr = _mean(ba), _mean(br)
        da.append(ma)
        dr.append(mr)
        ds.append(ma - mr)
    return _pct(da, alpha), _pct(dr, alpha), _pct(ds, alpha)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _pct(xs, alpha):
    xs = sorted(xs)
    if not xs:
        return (float('nan'), float('nan'))
    lo = xs[max(0, int(math.floor((alpha / 2) * len(xs))))]
    hi = xs[min(len(xs) - 1, int(math.ceil((1 - alpha / 2) * len(xs))) - 1)]
    return (lo, hi)


# ═════════════════════════════════════════════════════════════════════
#  Table
# ═════════════════════════════════════════════════════════════════════

def build_table(pairs, label_of, name, n_perm, n_boot):
    """pairs: {qid: (real_rec, placebo_rec)}; label_of: qid -> 'accept'|'reject'."""
    subsets = {'accept': [], 'reject': []}
    for qid, (r, p) in pairs.items():
        g = label_of(qid)
        if g in subsets:
            subsets[g].append((qid, r, p))

    print('\n' + '=' * 74)
    print('Gate label source: %s' % name)
    print('=' * 74)
    print('%-8s %5s %10s %10s %10s   %5s %5s %10s'
          % ('subset', 'N', 'Real', 'Placebo', 'Delta', 'b', 'c', 'McNemar p'))
    print('-' * 74)

    d = {}
    stats = {}
    for g in ('accept', 'reject'):
        rows = subsets[g]
        n = len(rows)
        if not n:
            print('%-8s %5d   (empty)' % (g, 0))
            d[g] = []
            continue
        nr = sum(1 for _q, r, _p in rows if r.get('correct'))
        np_ = sum(1 for _q, _r, p in rows if p.get('correct'))
        b = sum(1 for _q, r, p in rows if r.get('correct') and not p.get('correct'))
        c = sum(1 for _q, r, p in rows if p.get('correct') and not r.get('correct'))
        pv = mcnemar_exact(b, c)
        d[g] = [int(bool(r.get('correct'))) - int(bool(p.get('correct')))
                for _q, r, p in rows]
        stats[g] = dict(n=n, real=nr / n, placebo=np_ / n,
                        delta=(nr - np_) / n, b=b, c=c, p=pv)
        print('%-8s %5d %9.2f%% %9.2f%% %+9.2f pp   %5d %5d %10.4f'
              % (g, n, 100 * nr / n, 100 * np_ / n, 100 * (nr - np_) / n, b, c, pv))

    if d.get('accept') and d.get('reject'):
        p_two, p_one, obs = permutation_p(d['accept'], d['reject'], n_perm)
        ci_a, ci_r, ci_s = bootstrap_ci(d['accept'], d['reject'], n_boot)
        print('-' * 74)
        print('Selectivity  S = Delta_accept - Delta_reject = %+.2f pp' % (100 * obs))
        print('  permutation test (%d perms, sizes preserved): '
              'two-sided p = %.4f, one-sided p = %.4f' % (n_perm, p_two, p_one))
        p_perm = p_two
        print('  bootstrap 95%% CI (%d resamples):' % n_boot)
        print('    Delta_accept  [%+.2f, %+.2f] pp' % (100 * ci_a[0], 100 * ci_a[1]))
        print('    Delta_reject  [%+.2f, %+.2f] pp' % (100 * ci_r[0], 100 * ci_r[1]))
        print('    S             [%+.2f, %+.2f] pp' % (100 * ci_s[0], 100 * ci_s[1]))
        stats['selectivity'] = dict(S=obs, p_perm_two_sided=p_two,
                                    p_perm_one_sided=p_one,
                                    ci_accept=ci_a, ci_reject=ci_r, ci_S=ci_s)

        # What the gate policy actually buys, end to end.
        n_all = sum(len(v) for v in subsets.values())
        acc_real = sum(1 for _g, rows in subsets.items()
                       for _q, r, _p in rows if r.get('correct')) / n_all
        acc_plac = sum(1 for _g, rows in subsets.items()
                       for _q, _r, p in rows if p.get('correct')) / n_all
        acc_gate = (sum(1 for _q, r, _p in subsets['accept'] if r.get('correct'))
                    + sum(1 for _q, _r, p in subsets['reject'] if p.get('correct'))
                    ) / n_all
        print('  policy accuracy over all %d paired queries:' % n_all)
        print('    always inject real top-1 : %.2f%%' % (100 * acc_real))
        print('    always inject placebo    : %.2f%%' % (100 * acc_plac))
        print('    gate policy              : %.2f%%' % (100 * acc_gate))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--permutations', type=int, default=20000,
                    help='matches the appendix (20,000)')
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--drop-truncated', action='store_true',
                    help='exclude pairs where either arm hit the token ceiling. '
                         'A truncated response scores 0 because the answer line '
                         'never appeared, which is a format failure independent '
                         'of what was in the memory slot, so it is noise added '
                         'equally to both arms. Changes the result materially '
                         '(see the header of the printed table) and must be '
                         'reported as a stated filter, not applied silently.')
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    man = {r['query_id']: r for r in json.load(io.open(MANIFEST, encoding='utf-8'))}
    real = read_jsonl(os.path.join(REAL_RUN, 'results.jsonl'))
    plac = read_jsonl(os.path.join(PLACEBO_RUN, 'results.jsonl'))
    inj = read_jsonl(os.path.join(ZPD_RUN, 'memory_injections.jsonl'))

    print('=' * 74)
    print('ARTIFACTS')
    print('=' * 74)
    print('manifest rows        %d' % len(man))
    print('real top-1 records   %d   outcomes %s'
          % (len(real), dict(Counter(r.get('outcome') for r in real.values()))))
    print('placebo records      %d   outcomes %s'
          % (len(plac), dict(Counter(r.get('outcome') for r in plac.values()))))

    ids = [q for q in man if q in real and q in plac]
    dropped = [q for q in ids
               if real[q].get('outcome') == 'infra_error'
               or plac[q].get('outcome') == 'infra_error']
    pairs = {q: (real[q], plac[q]) for q in ids if q not in dropped}
    print('paired queries       %d   (dropped %d for an infra error in either arm)'
          % (len(pairs), len(dropped)))

    if args.drop_truncated:
        before = len(pairs)
        pairs = {q: (r, p) for q, (r, p) in pairs.items()
                 if r.get('outcome') != 'truncated' and p.get('outcome') != 'truncated'}
        print('TRUNCATION FILTER ON   %d -> %d pairs (%d dropped because either arm '
              'hit the token ceiling)' % (before, len(pairs), before - len(pairs)))

    # How far apart the two gate definitions are.
    agree = sum(1 for q in pairs
                if (man[q]['gate_decision'] == 'accept')
                == ((inj.get(q, {}).get('n_injected') or 0) > 0))
    print('gate-label agreement %d/%d  (manifest label vs what memory_zpd injected)'
          % (agree, len(pairs)))

    out = {}
    out['manifest'] = build_table(
        pairs, lambda q: man[q]['gate_decision'],
        'counterfactual_manifest.json  gate_decision', args.permutations, args.bootstrap)
    out['run'] = build_table(
        pairs, lambda q: 'accept' if (inj.get(q, {}).get('n_injected') or 0) > 0 else 'reject',
        'memory_zpd run  n_injected > 0', args.permutations, args.bootstrap)

    if args.json_out:
        with io.open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=list)
        print('\nwrote %s' % args.json_out)


if __name__ == '__main__':
    main()
