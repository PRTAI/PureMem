"""
bbeh/run.py — evaluate one arm on the BBEH test split.

An arm is (reasoner, memory version). ``no_memory`` is the bare student;
``memory`` is the same student with a gated precedent block. Every arm writes
its own run directory:

    runs/<arm_label>_<model>/
        results.jsonl          one record per item (the resume cache)
        summary.json           overall + per-task accuracy, outcome counts
        memory_injections.jsonl  what was injected and why (memory arms only)
        token_usage.json       spend, by model

Resume semantics, learned the hard way:

  * A cached record counts as done only if it is *scorable*. Infra errors are
    re-run on the next invocation — a proxy timeout must not become a permanent
    zero for that item, because a scattering of permanent zeros is invisible in
    aggregate and biases whichever arm happened to run during a bad window.
  * An empty ``response`` is a cache MISS even if the record exists.
  * ``--rescore`` re-applies the scorer to cached responses without any API
    calls. Use it after touching ``official_eval`` or the answer format.
  * ``--no-skip-existing`` forces a full re-solve.

The run aborts if the first ``--abort-after`` items all fail, so a bad key or a
wrong model name costs seconds instead of the whole budget.
"""

import argparse
import json
import logging
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence

from bbeh import config, data, official_eval, prompts, reasoner as reasoner_mod
from bbeh.api_client import GenResult, TokenUsage, build_client, dry_run_solve

_WRITE_LOCK = threading.Lock()


class DryRunClient:
    """Fabricates solve responses. Deterministic per (item, salt); no network.

    Carries the item through ``solve()`` via a thread-local, because the
    reasoner's contract is ``generate_detailed(prompt, ...)`` and honouring that
    contract is what makes the dry run test the real code path.
    """

    def __init__(self, correct_rate: float = 0.35, salt: str = ''):
        self.correct_rate = correct_rate
        self.salt = salt
        self._local = threading.local()

    def set_item(self, item: dict):
        self._local.item = item

    def generate_detailed(self, prompt: str, **kw) -> GenResult:
        item = getattr(self._local, 'item', None) or {'id': 'x', 'target': ''}
        # The precedent block lengthens the prompt; reflect that in the token
        # count so token_usage.json is comparable between arms.
        res = dry_run_solve(item, correct_rate=self.correct_rate, salt=self.salt)
        res.prompt_tokens = len(prompt) // 4
        return res


# ═════════════════════════════════════════════════════════════════════
#  Cache
# ═════════════════════════════════════════════════════════════════════

def load_cache(path: str) -> Dict[str, dict]:
    return data.read_jsonl_indexed(path, key='id')


def is_done(rec: Optional[dict]) -> bool:
    """Only a scorable, non-empty record counts as done."""
    if not rec:
        return False
    if rec.get('outcome') == 'infra_error':
        return False
    return bool((rec.get('response') or '').strip())


def rescore(cache: Dict[str, dict], items: Sequence[dict]) -> int:
    """Re-apply the scorer to cached responses. No API calls."""
    by_id = {it['id']: it for it in items}
    changed = 0
    for rec in cache.values():
        item = by_id.get(rec['id'])
        if not item or not (rec.get('response') or '').strip():
            continue
        correct, pred, ref = official_eval.score_with_detail(
            rec['response'], item['target'])
        if (correct, pred) != (rec.get('correct'), rec.get('prediction')):
            changed += 1
        rec['correct'], rec['prediction'], rec['reference'] = correct, pred, ref
    return changed


# ═════════════════════════════════════════════════════════════════════
#  Summary
# ═════════════════════════════════════════════════════════════════════

def summarize(records: Sequence[dict], arm_label: str, model: str,
              extra: Optional[dict] = None, arm_type: str = '') -> dict:
    # arm_type is carried explicitly rather than sniffed from arm_label: the
    # label picks up prefixes (DRYRUN-) and suffixes (the memory version), so
    # substring tests on it silently misclassify runs.
    arm_type = arm_type or ('memory' if 'memory' in arm_label else 'no_memory')
    scorable = [r for r in records if r.get('outcome') != 'infra_error']
    n_correct = sum(1 for r in scorable if r.get('correct'))
    by_task = defaultdict(lambda: {'n': 0, 'correct': 0, 'infra': 0,
                                   'truncated': 0, 'injected': 0, 'n_inj_items': 0})
    for r in records:
        slot = by_task[r['task']]
        if r.get('outcome') == 'infra_error':
            slot['infra'] += 1
            continue
        slot['n'] += 1
        slot['correct'] += bool(r.get('correct'))
        slot['truncated'] += r.get('outcome') == 'truncated'
        n_inj = r.get('n_injected') or 0
        slot['injected'] += n_inj
        slot['n_inj_items'] += bool(n_inj)

    inj = [r.get('n_injected') or 0 for r in scorable]
    summary = {
        'arm': arm_label,
        'arm_type': arm_type,
        'model': model,
        'n_items': len(records),
        'n_scorable': len(scorable),
        'n_infra_error': len(records) - len(scorable),
        'n_truncated': sum(1 for r in scorable if r.get('outcome') == 'truncated'),
        'n_correct': n_correct,
        # Denominator is scorable attempts. Infra errors are excluded, not
        # counted wrong — see the module docstring.
        'accuracy': n_correct / len(scorable) if scorable else 0.0,
        'n_items_with_memory': sum(1 for x in inj if x),
        'memory_injection_rate': (sum(1 for x in inj if x) / len(inj)) if inj else 0.0,
        'mean_chunks_injected': (sum(inj) / len(inj)) if inj else 0.0,
        'per_task': {t: {**v, 'accuracy': v['correct'] / v['n'] if v['n'] else 0.0}
                     for t, v in sorted(by_task.items())},
    }
    if extra:
        summary.update(extra)
    return summary


def print_summary(s: dict, warns: Sequence[str] = ()) -> None:
    print(f'\n{"=" * 72}\n{s["arm"]}  |  {s["model"]}\n{"=" * 72}')
    print(f'accuracy        {s["accuracy"]:.4f}   '
          f'({s["n_correct"]}/{s["n_scorable"]} scorable)')
    print(f'items           {s["n_items"]}')
    print(f'infra errors    {s["n_infra_error"]}'
          + ('   <- excluded from the denominator; rerun to retry them'
             if s['n_infra_error'] else ''))
    print(f'truncated       {s["n_truncated"]}'
          + ('   <- raise --max-tokens; these are format failures, not reasoning '
             'failures' if s['n_truncated'] > 0.02 * max(1, s['n_scorable']) else ''))
    if s.get('arm_type') == 'memory':
        print(f'memory injected {s["n_items_with_memory"]}/{s["n_scorable"]} items '
              f'({s["memory_injection_rate"]:.1%}), '
              f'mean {s["mean_chunks_injected"]:.2f} chunks/item')
        if s['memory_injection_rate'] < 0.15:
            print('  WARNING: the Stage-3 gate rejected almost everything. This arm')
            print('  is nearly identical to no_memory, so a null result here says')
            print('  nothing about memory — it says the gate is too strict.')

    print(f'\n{"task":36s} {"n":>5s} {"acc":>7s} {"inj":>6s} {"trunc":>6s} {"infra":>6s}')
    for task, v in s['per_task'].items():
        print(f'{task:36s} {v["n"]:5d} {v["accuracy"]:7.3f} '
              f'{v["n_inj_items"]:6d} {v["truncated"]:6d} {v["infra"]:6d}')
    for w in warns:
        print(f'\nWARNING: {w}')


# ═════════════════════════════════════════════════════════════════════
#  Driver
# ═════════════════════════════════════════════════════════════════════

def run_arm(arm: str = 'no_memory',
            memory_version: Optional[str] = None,
            model: str = config.STUDENT_MODEL,
            judge_model: str = config.JUDGE_MODEL,
            arm_label: Optional[str] = None,
            tasks: Optional[Sequence[str]] = None,
            limit_per_task: Optional[int] = None,
            limit_total: Optional[int] = None,
            max_workers: int = 4,
            max_tokens: int = config.SOLVE_MAX_TOKENS,
            temperature: float = config.SOLVE_TEMPERATURE,
            top_k: int = config.TOP_K,
            rerank_samples: int = config.RERANK_SAMPLES_N,
            tau: float = config.RERANK_FIT_TAU,
            vote_threshold: int = config.RERANK_VOTE_THRESHOLD,
            no_reranker: bool = False,
            random_retrieval: bool = False,
            dry_run: bool = False,
            skip_existing: bool = True,
            rescore_only: bool = False,
            report_only: bool = False,
            abort_after: int = 12) -> dict:
    config.ensure_dirs()
    prompts.selftest_arm_parity()      # cheap; the whole comparison rests on it

    if arm == 'memory' and not memory_version:
        raise SystemExit('--arm memory requires --memory-version')
    label = arm_label or (f'memory_{memory_version}' if arm == 'memory' else 'no_memory')
    if dry_run:
        label = f'DRYRUN-{label}'

    items = data.select_items(data.load_split('test'), tasks=tasks,
                              limit_per_task=limit_per_task,
                              limit_total=limit_total)
    if not items:
        raise SystemExit('no test items selected')

    out_dir = config.run_dir(label, model)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, 'results.jsonl')
    cache = load_cache(results_path)
    warns: List[str] = []

    # ─── report / rescore: no API calls at all ───────────────────────
    if report_only or rescore_only:
        if rescore_only:
            changed = rescore(cache, items)
            data.write_jsonl(results_path, [cache[i['id']] for i in items
                                            if i['id'] in cache])
            print(f'rescored {len(cache)} records, {changed} verdicts changed')
        recs = [cache[i['id']] for i in items if i['id'] in cache]
        s = summarize(recs, label, model, arm_type=arm)
        print_summary(s)
        return s

    # ─── memory machinery ────────────────────────────────────────────
    reasoner_kwargs = {}
    judge_usage = TokenUsage()
    judge = None
    if arm == 'memory':
        from bbeh.retriever import MemoryRetriever, QueryEmbedder
        from bbeh.reranker import ApproachFitReranker
        retriever = MemoryRetriever(memory_version)
        warns.extend(retriever.warn_if_degenerate())
        qemb = QueryEmbedder(dry_run=dry_run).embed_items(items)
        if not no_reranker:
            cache_name = ('DRYRUN-' if dry_run else '') + config.RERANK_CACHE_NAME
            judge = ApproachFitReranker(
                model=judge_model,
                cache_path=os.path.join(config.version_dir(memory_version), cache_name),
                n_samples=rerank_samples, usage=judge_usage, dry_run=dry_run)
        reasoner_kwargs = dict(retriever=retriever, query_embeddings=qemb,
                               reranker=judge, top_k=top_k, tau=tau,
                               vote_threshold=vote_threshold,
                               random_retrieval=random_retrieval)

    usage = TokenUsage()
    client = (DryRunClient(salt=label) if dry_run else
              build_client(model, temperature=temperature,
                           max_tokens=max_tokens, usage=usage))
    reasoner = reasoner_mod.build_reasoner(
        arm, client, max_tokens=max_tokens, temperature=temperature,
        **reasoner_kwargs)

    todo = [it for it in items if not (skip_existing and is_done(cache.get(it['id'])))]
    retry = sum(1 for it in items
                if cache.get(it['id']) and not is_done(cache.get(it['id'])))
    logging.info('%s: %d items, %d cached, %d to run (%d retried after infra '
                 'errors)%s', label, len(items), len(items) - len(todo), len(todo),
                 retry, '  [DRY RUN]' if dry_run else '')

    def work(item):
        if dry_run:
            client.set_item(item)
        return reasoner.solve(item)

    if todo:
        inj_path = os.path.join(out_dir, 'memory_injections.jsonl')
        if not skip_existing:
            # When overwriting existing runs (--no-skip-existing), clear old files so
            # results.jsonl and memory_injections.jsonl stay in exact 1:1 parity
            for p in (results_path, inj_path):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            cache.clear()
        n_done = n_err = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(work, it): it for it in todo}
            try:
                for fut in as_completed(futures):
                    res = fut.result()
                    rec = res.to_record()
                    with _WRITE_LOCK:
                        data.append_jsonl(results_path, rec)
                        if arm == 'memory':
                            data.append_jsonl(inj_path, {
                                'id': res.id, 'task': res.task,
                                'n_injected': res.n_injected,
                                'injected': res.injected,
                                'retrieval_error': res.retrieval_error,
                                'correct': res.correct})
                    cache[res.id] = rec
                    n_done += 1
                    n_err += res.outcome == 'infra_error'
                    # Fail fast on a systematic problem (bad key, wrong model)
                    # instead of burning the budget producing zeros.
                    if n_done >= abort_after and n_err == n_done:
                        for f in futures:
                            f.cancel()
                        raise SystemExit(
                            f'\nABORT: first {n_done} items all failed with infra '
                            f'errors.\nLast error: {res.error}\n'
                            'Nothing was scored; fix the endpoint and rerun '
                            '(cached work is preserved).')
                    if n_done % 25 == 0 or n_done == len(todo):
                        eta = (time.time() - t0) / n_done * (len(todo) - n_done)
                        acc = sum(1 for r in cache.values()
                                  if r.get('correct')) / max(1, len(cache))
                        logging.info('  %d/%d | %d infra | running acc %.3f | ETA %.0fm',
                                     n_done, len(todo), n_err, acc, eta / 60)
            except KeyboardInterrupt:
                logging.warning('interrupted — %d results written to %s; '
                                'rerun to resume', n_done, results_path)
                raise

    records = [cache[it['id']] for it in items if it['id'] in cache]
    extra = {
        'memory_version': memory_version,
        'judge_model': judge_model if judge else None,
        'reranker': 'off' if (arm == 'memory' and no_reranker) else
                    ('on' if judge else None),
        'top_k': top_k, 'tau': tau, 'vote_threshold': vote_threshold,
        'rerank_samples': rerank_samples,
        'max_tokens': max_tokens, 'temperature': temperature,
        'n_test_selected': len(items),
        'dry_run': dry_run,
    }
    if judge:
        extra['judge_stats'] = judge.stats()
        if judge.n_infra_failures:
            warns.append(
                f'{judge.n_infra_failures} judge calls failed; those candidates '
                'could not reach the vote threshold and were gated out. The '
                'injection rate above is therefore a floor, not a measurement.')
    s = summarize(records, label, model, extra, arm_type=arm)

    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    if not dry_run:
        merged = usage.snapshot()
        merged['judge'] = judge_usage.snapshot()
        with open(os.path.join(out_dir, 'token_usage.json'), 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

    print_summary(s, warns)
    print(f'\n-> {out_dir}')
    return s


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Evaluate one arm on the BBEH test split',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # plumbing check, no API spend
  python -m bbeh.run --arm memory --memory-version DRYRUN-zpd --dry-run \\
      --tasks bbeh_word_sorting --limit-per-task 5

  # the claim-2 baseline
  python -m bbeh.run --arm no_memory --limit-per-task 30

  # the treatment arm
  python -m bbeh.run --arm memory --memory-version zpd --limit-per-task 30

  # re-score cached responses after touching official_eval (free)
  python -m bbeh.run --arm no_memory --limit-per-task 30 --rescore
""")
    p.add_argument('--arm', choices=('no_memory', 'memory'), default='no_memory')
    p.add_argument('--memory-version', default=None)
    p.add_argument('--model', default=config.STUDENT_MODEL)
    p.add_argument('--judge-model', default=config.JUDGE_MODEL)
    p.add_argument('--arm-label', default=None, help='override the run dir name')
    p.add_argument('--tasks', nargs='*', default=None)
    p.add_argument('--limit-per-task', type=int, default=None)
    p.add_argument('--limit-total', type=int, default=None)
    p.add_argument('--max-workers', type=int, default=4)
    p.add_argument('--max-tokens', type=int, default=config.SOLVE_MAX_TOKENS)
    p.add_argument('--temperature', type=float, default=config.SOLVE_TEMPERATURE)
    p.add_argument('--top-k', type=int, default=config.TOP_K)
    p.add_argument('--rerank-samples', type=int, default=config.RERANK_SAMPLES_N)
    p.add_argument('--tau', type=float, default=config.RERANK_FIT_TAU)
    p.add_argument('--vote-threshold', type=int, default=config.RERANK_VOTE_THRESHOLD)
    p.add_argument('--no-reranker', action='store_true',
                   help='ablate Stage 3: inject the top-k by blended score, ungated')
    p.add_argument('--random-retrieval', action='store_true',
                   help='ablate Stage 2: pick candidate chunks randomly instead of by similarity')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--no-skip-existing', action='store_true')
    p.add_argument('--rescore', action='store_true',
                   help='re-score cached responses; no API calls')
    p.add_argument('--report-only', action='store_true')
    p.add_argument('--abort-after', type=int, default=12)
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    run_arm(arm=a.arm, memory_version=a.memory_version, model=a.model,
            judge_model=a.judge_model, arm_label=a.arm_label, tasks=a.tasks,
            limit_per_task=a.limit_per_task, limit_total=a.limit_total,
            max_workers=a.max_workers, max_tokens=a.max_tokens,
            temperature=a.temperature, top_k=a.top_k,
            rerank_samples=a.rerank_samples, tau=a.tau,
            vote_threshold=a.vote_threshold, no_reranker=a.no_reranker,
            random_retrieval=a.random_retrieval,
            dry_run=a.dry_run, skip_existing=not a.no_skip_existing,
            rescore_only=a.rescore, report_only=a.report_only,
            abort_after=a.abort_after)


if __name__ == '__main__':
    main()
