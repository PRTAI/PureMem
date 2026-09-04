"""
bbeh/selftest_retrieval.py — offline checks for Stage 1/2/3 wiring.

Everything here runs against a DRYRUN memory version with a fake judge and
costs nothing. What it is actually protecting:

  * the tau + vote gate really gates, and an empty return is reachable;
  * the gate is monotone in tau (a stricter threshold can never admit more);
  * Stage 1's tag bonus is a nudge, not a filter — cross-task candidates must
    still be able to win, or the cross-task transfer claim is untestable by
    construction;
  * retrieval never returns a chunk sourced from the query item itself;
  * judge infra failures degrade toward no-memory rather than toward a
    confident 0.0 verdict.
"""

import logging
import sys

import numpy as np

from bbeh import config, data, reranker as rr
from bbeh.retriever import MemoryRetriever, QueryEmbedder


class FakeJudge:
    """Judge stub. ``mode`` decides the verdict; no network, fully determinate."""

    def __init__(self, mode='all_pass', n=config.RERANK_SAMPLES_N):
        self.mode, self.n = mode, n
        self.calls = 0

    def score(self, query_text, query_task, candidates):
        self.calls += 1
        out = []
        for i, _ in enumerate(candidates):
            if self.mode == 'all_pass':
                s = [1.0] * self.n
            elif self.mode == 'all_fail':
                s = [0.0] * self.n
            elif self.mode == 'first_only':
                s = [1.0] * self.n if i == 0 else [0.0] * self.n
            elif self.mode == 'borderline':
                # 2 of 5 clear tau: below the vote threshold, so it must gate.
                s = [1.0, 1.0, 0.0, 0.0, 0.0]
            elif self.mode == 'infra_dead':
                s = []                       # every sample failed
            else:
                raise ValueError(self.mode)
            out.append({'samples': s, 'mean': sum(s) / len(s) if s else 0.0,
                        'std': 0.0, 'n': len(s), 'n_requested': self.n,
                        'degraded': len(s) < self.n})
        return out


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
    vid = 'DRYRUN-zpd'
    r = MemoryRetriever(vid)
    test = data.select_items(data.load_split('test'), limit_per_task=3)
    qemb = QueryEmbedder(dry_run=True).embed_items(test)
    probe = test[:12]
    failures = []

    def check(name, cond, detail=''):
        print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f'   {detail}' if detail else ''))
        if not cond:
            failures.append(name)

    print(f'\n{"=" * 68}\nretrieval selftest on {vid}\n{"=" * 68}')
    print(f'{len(r.records)} chunks / {len(r.demos)} demos, {len(probe)} probe queries\n')

    # ─── Stage 2: recall produces candidates at all ──────────────────
    pools = [r._gather_candidate_pool(qemb[it['id']], pool_size=5) for it in probe]
    check('stage 2 returns a non-empty pool for every query',
          all(len(p) > 0 for p in pools),
          f'min={min(len(p) for p in pools)} max={max(len(p) for p in pools)}')
    layers = {c['retrieval_layer'] for p in pools for c in p}
    check('both pools contribute (abstract AND concrete)',
          layers == {'abstract', 'concrete'}, str(sorted(layers)))
    check('no duplicate chunk within a pool',
          all(len({c['chunk_id'] for c in p}) == len(p) for p in pools))
    sims = [c['similarity'] for p in pools for c in p]
    check('similarities are valid cosines',
          all(-1.001 <= s <= 1.001 for s in sims),
          f'mean={np.mean(sims):.3f} max={max(sims):.3f}  (~0.2 is normal, '
          'and low cosine is NOT the metric to tune)')

    # ─── no self-retrieval ───────────────────────────────────────────
    # Memory is built from train, queries come from test, so a query item must
    # never appear as its own precedent. This is the leakage check that makes
    # claim 2 meaningful.
    leaked = [it['id'] for it, p in zip(probe, pools)
              if any(c['item_id'] == it['id'] for c in p)]
    check('no query retrieves itself (train/test disjointness holds)',
          not leaked, str(leaked[:3]))

    # ─── Stage 1: soft, not a filter ─────────────────────────────────
    it = probe[0]
    pool = r._gather_candidate_pool(qemb[it['id']], pool_size=5)
    same = r._tag_bonus(dict(pool[0], task=it['task'], pattern_type='arithmetic_chain'),
                        it['task'], ['arithmetic_chain'], config.STAGE1_TASK_WEIGHT,
                        config.STAGE1_PATTERN_WEIGHT)
    cross = r._tag_bonus(dict(pool[0], task='__other__', pattern_type='__none__'),
                         it['task'], ['arithmetic_chain'], config.STAGE1_TASK_WEIGHT,
                         config.STAGE1_PATTERN_WEIGHT)
    check('tag bonus rewards agreement', same > cross, f'{same:.2f} vs {cross:.2f}')
    check('tag bonus is bounded (a nudge, not a veto)',
          same <= 0.35, f'max bonus {same:.2f} vs typical cosine spread ~0.3')

    # The mechanism prior must track the data in BOTH directions: fire when a
    # task has a dominant mechanism, stay silent when it does not. The dry-run
    # fabricator assigns pattern_type uniformly at random, so an empty prior here
    # is the *correct* answer — asserting non-empty would have been asserting
    # that the prior hallucinates structure that isn't there.
    flat = {t: r.pattern_prior_for_task(t) for t in sorted({c['task'] for c in r.records})}
    check('prior stays empty on a flat (random) mechanism distribution',
          not any(flat.values()), 'dry-run pattern_types are uniform by construction')

    concentrated = MemoryRetriever.__new__(MemoryRetriever)
    task = r.records[0]['task']
    concentrated.records = (
        [dict(c, task=task, pattern_type='sorting_ordering') for c in r.records[:80]]
        + [dict(c, task=task, pattern_type='elimination') for c in r.records[80:100]]
        + [dict(c, task=task, pattern_type='other') for c in r.records[100:105]])
    got = concentrated._build_pattern_prior().get(task, [])
    check('prior fires on a concentrated distribution',
          got[:2] == ['sorting_ordering', 'elimination'] and 'other' not in got,
          f'{got}  (5/105 = 4.8% "other" correctly below the 15% floor)')

    if not any(flat.values()):
        print('        NOTE: with a real abstractor this must be non-empty for most')
        print('        tasks. An all-empty prior means STAGE1_PATTERN_WEIGHT never')
        print('        applies and the abstractor probably collapsed — check')
        print('        `python -m bbeh.abstract --report-only` before spending on runs.')

    # Cross-task candidates must survive Stage 1, or the transfer claim is
    # untestable: a within-task-only shortlist can only ever show few-shot.
    n_cross = 0
    for q in probe:
        shortlist = r.retrieve_three_stage(q['input'], qemb[q['id']], q['task'],
                                           reranker=None)
        n_cross += sum(1 for c in shortlist if c['task'] != q['task'])
    check('cross-task candidates survive Stage 1', n_cross > 0,
          f'{n_cross} cross-task precedents across {len(probe)} queries')

    # ─── Stage 3: the gate ───────────────────────────────────────────
    def run(mode, **kw):
        j = FakeJudge(mode)
        return [r.retrieve_three_stage(q['input'], qemb[q['id']], q['task'],
                                       reranker=j, **kw) for q in probe]

    passed = run('all_pass')
    check('all_pass injects up to top_k',
          all(0 < len(x) <= config.TOP_K for x in passed),
          f'sizes {[len(x) for x in passed[:6]]}')
    check('all_fail injects nothing (empty is a designed outcome)',
          all(len(x) == 0 for x in run('all_fail')))
    check('borderline (2/5 votes < threshold 3) gates out',
          all(len(x) == 0 for x in run('borderline')))
    check('single passing candidate injects exactly one',
          all(len(x) == 1 for x in run('first_only')))
    dead = run('infra_dead')
    check('judge infra death degrades to no-memory, not to a 0.0 verdict',
          all(len(x) == 0 for x in dead))

    # Monotonicity: raising tau can never admit more. Catches an inverted
    # comparison, which would otherwise look like a plausible tuning result.
    strict = FakeJudge('borderline')
    loose = [len(r.retrieve_three_stage(q['input'], qemb[q['id']], q['task'],
                                        reranker=strict, vote_threshold=2))
             for q in probe]
    tight = [len(r.retrieve_three_stage(q['input'], qemb[q['id']], q['task'],
                                        reranker=strict, vote_threshold=4))
             for q in probe]
    check('gate is monotone in the vote threshold',
          all(t <= l for t, l in zip(tight, loose)) and sum(loose) > sum(tight),
          f'votes>=2 -> {sum(loose)} injected, votes>=4 -> {sum(tight)}')

    # ─── diagnostics survive to the caller ───────────────────────────
    one = run('all_pass')[0][0]
    need = {'similarity', 'tag_bonus', 'blended_score', 'fit_mean', 'fit_votes',
            'fit_tau', 'fit_passed', 'fit_degraded', 'retrieval_layer',
            'chunk_id', 'source_idx', 'item_id', 'task'}
    check('every injected chunk carries its full audit trail',
          need <= set(one), f'missing {sorted(need - set(one))}')

    # ─── misalignment must raise, not truncate ───────────────────────
    import shutil, tempfile, os
    tmp = tempfile.mkdtemp()
    try:
        broken = os.path.join(tmp, 'DRYRUN-broken')
        shutil.copytree(config.version_dir(vid), broken)
        emb = np.load(os.path.join(broken, config.EMBEDDINGS_NPY_NAME))
        np.save(os.path.join(broken, config.EMBEDDINGS_NPY_NAME), emb[:-1])
        orig = config.MEMORY_VERSIONS_DIR
        config.MEMORY_VERSIONS_DIR = tmp
        try:
            MemoryRetriever('DRYRUN-broken')
            check('a misaligned bank refuses to load', False, 'it loaded!')
        except AssertionError as e:
            check('a misaligned bank refuses to load', 'misaligned' in str(e))
        finally:
            config.MEMORY_VERSIONS_DIR = orig
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f'\n{"=" * 68}')
    if failures:
        print(f'{len(failures)} FAILURES: {failures}')
        return 1
    print('all retrieval checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
