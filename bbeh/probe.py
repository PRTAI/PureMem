"""
bbeh/probe.py — measure how hard each TRAIN item is for the STUDENT.

This is the only source of difficulty in the whole pipeline, and therefore the
foundation of claim 1. ZPD is a property of the *learner*, so difficulty must be
measured against the student model that will later be evaluated — never against
the teacher, and never against a heuristic like input length or step count.

Method: sample the student ``k`` times per train item at ``PROBE_TEMPERATURE``
(nonzero on purpose — a deterministic sample gives only pass_rate ∈ {0, 1} and
there would be no zone to speak of), score each attempt with the official BBEH
scorer, and report ``pass_rate = n_correct / n_samples``.

Three things this file is careful about, all learned the hard way:

  * **The prompt is the eval prompt.** ``prompts.build_solve_prompt(item)`` with
    no precedents is byte-identical to the ``no_memory`` arm's prompt. If the
    probe asked the question differently, pass_rate would describe a model we
    never evaluate.
  * **Infra failures are not wrong answers.** An empty body or a timeout is
    recorded as ``error`` and excluded from the denominator, rather than
    silently counted as a failed attempt (which would push items toward
    pass_rate 0 and corrupt the band). Items whose probe never completed are
    marked ``complete: false`` and reported.
  * **Truncation is visible.** If the student hits the token ceiling, the answer
    line never appears and the attempt scores 0 for a reason that has nothing to
    do with reasoning. A high truncation rate means the experiment is measuring
    ``SOLVE_MAX_TOKENS``, so the report flags it.

Usage::

    # pilot: 3 tasks x 20 items, no API spend, just exercise the plumbing
    python -m bbeh.probe --tasks bbeh_word_sorting bbeh_time_arithmetic \\
                                 bbeh_boolean_expressions \\
                         --limit-per-task 20 --dry-run

    # real pilot
    python -m bbeh.probe --tasks bbeh_word_sorting --limit-per-task 20

    # full probe
    python -m bbeh.probe --k 5 --max-workers 8

    # read the landscape without calling anything
    python -m bbeh.probe --report-only
"""

import argparse
import json
import logging
import os
import threading
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from bbeh import config, data, official_eval, prompts
from bbeh.api_client import GenResult, TokenUsage, build_client, dry_run_solve

_WRITE_LOCK = threading.Lock()

# If this many attempts fail in a row with zero successes, something is wrong
# with the credentials / model name / base URL. Fail loudly instead of burning
# an hour producing an all-zero difficulty file that looks like "BBEH is hard".
_ABORT_AFTER_CONSECUTIVE_ERRORS = 15


# ═════════════════════════════════════════════════════════════════════
#  Work planning
# ═════════════════════════════════════════════════════════════════════

def _sample_key(item_id: str, sample_idx: int) -> str:
    return f'{item_id}|{sample_idx}'


def plan_work(items: Sequence[dict], k: int, done: Dict[str, dict],
              redo_errors: bool = True) -> List[Tuple[dict, int, str]]:
    """Which (item, attempt) pairs still need a call.

    Attempts recorded as infra errors are retried by default. Rationale: a
    cached error costs a permanently under-sampled pass_rate feeding the ZPD
    band, while retrying costs a few tokens — and the record already survived
    ``API_MAX_RETRIES`` rounds of backoff inside the call, so a fresh
    invocation is a genuinely new attempt, not a tight loop.
    """
    todo = []
    for it in items:
        for s in range(k):
            key = _sample_key(it['id'], s)
            rec = done.get(key)
            if rec is not None and not (redo_errors and rec.get('error')):
                continue
            todo.append((it, s, key))
    return todo


# ═════════════════════════════════════════════════════════════════════
#  One attempt
# ═════════════════════════════════════════════════════════════════════

def probe_one(item: dict, sample_idx: int, key: str, client,
              temperature: float, max_tokens: int, dry_run: bool) -> dict:
    """Run one student attempt and score it. Never raises."""
    prompt = prompts.build_solve_prompt(item)          # == the no_memory prompt

    if dry_run:
        # correct_rate spread across the pool so the fabricated difficulty
        # landscape actually contains a ZPD band to test the selector against.
        # zlib.crc32, not hash(): builtin hash() of a str is salted by
        # PYTHONHASHSEED, which would make dry runs irreproducible across
        # processes — the one property a plumbing test must have.
        rate = 0.15 + 0.7 * ((zlib.crc32(item['id'].encode()) % 100) / 100.0)
        res: GenResult = dry_run_solve(item, correct_rate=rate,
                                       salt=f'probe|{sample_idx}')
    else:
        res = client.generate_detailed(prompt, max_tokens=max_tokens,
                                       temperature=temperature)

    rec = {
        'key': key,
        'id': item['id'],
        'task': item['task'],
        'sample_idx': sample_idx,
        'correct': False,
        'prediction': '',
        'error': res.error or '',
        'truncated': bool(res.truncated),
        'attempts': res.attempts,
        'prompt_tokens': res.prompt_tokens,
        'completion_tokens': res.completion_tokens,
    }
    if res.ok:
        correct, pred, _ref = official_eval.score_with_detail(res.text, item['target'])
        rec['correct'] = bool(correct)
        rec['prediction'] = pred[:200]
    return rec


# ═════════════════════════════════════════════════════════════════════
#  Aggregation
# ═════════════════════════════════════════════════════════════════════

def aggregate(samples: Dict[str, dict], items: Sequence[dict],
              k: int) -> List[dict]:
    """Collapse per-attempt records into one difficulty record per item.

    ``pass_rate`` divides by the number of attempts that actually produced a
    response, not by ``k``. An item with 2/3 correct and 2 infra errors has
    pass_rate 0.667 and ``complete: false`` — an honest estimate flagged as
    under-sampled, rather than 2/5 = 0.4, which would be a fabricated number.
    """
    by_item: Dict[str, List[dict]] = defaultdict(list)
    for rec in samples.values():
        by_item[rec['id']].append(rec)

    out = []
    for it in items:
        recs = sorted(by_item.get(it['id'], []), key=lambda r: r['sample_idx'])
        ok = [r for r in recs if not r.get('error')]
        n_ok = len(ok)
        n_correct = sum(1 for r in ok if r.get('correct'))
        out.append({
            'id': it['id'],
            'task': it['task'],
            'k_requested': k,
            'n_samples': n_ok,
            'n_correct': n_correct,
            'n_error': len(recs) - n_ok,
            'n_truncated': sum(1 for r in recs if r.get('truncated')),
            'pass_rate': (n_correct / n_ok) if n_ok else None,
            'complete': n_ok >= k,
            # Kept for triage: seeing what the model actually answered on an
            # item it never got right is how you catch format problems that
            # look like reasoning problems.
            'predictions': [r.get('prediction', '') for r in ok][:k],
        })
    return out


def merge_aggregate(out_path: str, new_records: Sequence[dict]) -> List[dict]:
    """Merge fresh per-item records into the difficulty file. Never shrinks it.

    The file must accumulate, because ``aggregate`` only ever sees the items
    *this* invocation selected. Overwriting instead of merging means a narrow
    follow-up probe (say 3 tasks, to deepen a pilot) silently deletes the other
    20 tasks' difficulty from the aggregate. Nothing downstream would complain:
    ``teacher --select zpd`` and ``selector.py`` both read this file, so they
    would just quietly select from a third of the corpus and report a
    plausible-looking bank. The raw attempts survive in the per-sample cache, so
    the loss is recoverable — but only if you notice it, and there is no signal
    that you should.

    Records for items in ``new_records`` win; everything else is preserved as
    written, keeping each item's own ``k_requested`` rather than relabelling it
    against whatever k this invocation happened to use.
    """
    merged = {r['id']: r for r in data.read_jsonl(out_path)} if os.path.exists(out_path) else {}
    n_before = len(merged)
    merged.update({r['id']: r for r in new_records})
    records = [merged[i] for i in sorted(merged)]
    data.write_jsonl(out_path, records)
    n_kept = len(records) - len(new_records)
    if n_before and n_kept > 0:
        logging.info('difficulty file: %d records (%d updated/added this run, '
                     '%d preserved from earlier probes)',
                     len(records), len(new_records), n_kept)
    return records


# ═════════════════════════════════════════════════════════════════════
#  Driver
# ═════════════════════════════════════════════════════════════════════

def run_probe(model: str = config.STUDENT_MODEL,
              k: int = config.PROBE_K,
              split: str = 'train',
              tasks: Optional[Sequence[str]] = None,
              limit_per_task: Optional[int] = None,
              limit_total: Optional[int] = None,
              max_workers: int = 4,
              temperature: float = config.PROBE_TEMPERATURE,
              max_tokens: int = config.SOLVE_MAX_TOKENS,
              dry_run: bool = False,
              redo_errors: bool = True,
              report_only: bool = False) -> List[dict]:
    """Probe the student and write ``difficulty_<model>.jsonl``."""
    config.ensure_dirs()

    items = data.select_items(data.load_split(split), tasks,
                              limit_per_task, limit_total)
    if not items:
        raise SystemExit('no items selected — check --tasks / --limit-per-task')

    # A dry run must NEVER write to the real cache path. If it did, a
    # subsequent real run would see the fabricated attempts as "cached" and
    # silently treat invented numbers as measured difficulty — the single
    # nastiest failure mode we hit on the PuzzleWorld runs, because nothing
    # about the output looks wrong.
    label = f'DRYRUN-{model}' if dry_run else model
    samples_path = config.probe_samples_path(label)
    out_path = config.probe_path(label)
    done = data.read_jsonl_indexed(samples_path, key='key')

    todo = plan_work(items, k, done, redo_errors)
    n_cached = len(items) * k - len(todo)
    logging.info('probe %s: %d items x k=%d = %d attempts; %d cached, %d to run%s',
                 model, len(items), k, len(items) * k, n_cached, len(todo),
                 '  [DRY RUN]' if dry_run else '')

    if report_only:
        if not done:
            raise SystemExit(f'no probe samples at {samples_path} yet')
        records = merge_aggregate(out_path, aggregate(done, items, k))
        report(records, out_path)
        return records

    if todo:
        usage = TokenUsage()
        client = None if dry_run else build_client(
            model, temperature=temperature, max_tokens=max_tokens, usage=usage)

        n_done = n_err = n_correct = n_scored = 0
        consecutive_errors = 0
        t0 = time.time()

        def _submit(unit):
            it, s, key = unit
            return probe_one(it, s, key, client, temperature, max_tokens, dry_run)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_submit, u): u for u in todo}
            try:
                for fut in as_completed(futures):
                    rec = fut.result()
                    with _WRITE_LOCK:
                        data.append_jsonl(samples_path, rec)
                    done[rec['key']] = rec

                    n_done += 1
                    if rec['error']:
                        n_err += 1
                        consecutive_errors += 1
                    else:
                        consecutive_errors = 0
                        n_scored += 1
                        n_correct += int(rec['correct'])

                    if (consecutive_errors >= _ABORT_AFTER_CONSECUTIVE_ERRORS
                            and n_scored == 0):
                        for f in futures:
                            f.cancel()
                        raise SystemExit(
                            f'\nABORT: first {consecutive_errors} attempts all failed '
                            f'with no successes.\nLast error: {rec["error"]}\n'
                            f'Check BBEH_BASE_URL / BBEH_API_KEY / model name '
                            f'({model!r}) before spending more.'
                        )

                    if n_done % 25 == 0 or n_done == len(todo):
                        rate = n_correct / n_scored if n_scored else 0.0
                        eta = (time.time() - t0) / n_done * (len(todo) - n_done)
                        logging.info('  %d/%d attempts | acc %.3f | %d errors | ETA %.0fm',
                                     n_done, len(todo), rate, n_err, eta / 60)
            except KeyboardInterrupt:
                logging.warning('interrupted — %d attempts already on disk at %s; '
                                'rerun the same command to resume', n_done, samples_path)
                raise

        if not dry_run:
            usage.write(os.path.join(config.WORK_DIR,
                                     f'probe_usage_{config._slug(model)}.json'))

    records = merge_aggregate(out_path, aggregate(done, items, k))
    report(records, out_path)
    return records


# ═════════════════════════════════════════════════════════════════════
#  Report
# ═════════════════════════════════════════════════════════════════════

def report(records: Sequence[dict], out_path: str,
           zpd_low: float = config.ZPD_LOW,
           zpd_high: float = config.ZPD_HIGH) -> None:
    """Print the difficulty landscape. Read this before spending teacher tokens."""
    n = len(records)
    probed = [r for r in records if r['pass_rate'] is not None]
    incomplete = [r for r in records if not r['complete']]
    trunc = sum(r['n_truncated'] for r in records)
    total_attempts = sum(r['n_samples'] + r['n_error'] for r in records)

    print(f'\n{"=" * 68}')
    print(f'probe -> {out_path}')
    print(f'{"=" * 68}')
    print(f'items                 {n}')
    print(f'items with a rate     {len(probed)}')
    print(f'incomplete probes     {len(incomplete)}'
          + ('   <- rerun the same command to retry the failed attempts'
             if incomplete else ''))

    if not probed:
        print('\nNo scored attempts at all. Nothing to report.')
        return

    micro = (sum(r['n_correct'] for r in probed)
             / max(1, sum(r['n_samples'] for r in probed)))
    macro = sum(r['pass_rate'] for r in probed) / len(probed)
    print(f'mean attempt accuracy {micro:.3f}   (per-attempt, temp>0 — NOT the '
          f'temp-0 no_memory baseline)')
    print(f'mean item pass_rate   {macro:.3f}')
    if total_attempts:
        print(f'truncated attempts    {trunc} ({trunc / total_attempts:.1%})'
              + ('   <- raise SOLVE_MAX_TOKENS; these score 0 for the wrong reason'
                 if trunc / total_attempts > 0.05 else ''))

    # ─── the histogram that decides whether claim 1 is testable ──────
    hist = Counter(round(r['pass_rate'], 3) for r in probed)
    print('\npass_rate histogram:')
    for rate in sorted(hist):
        bar = '#' * min(50, hist[rate] * 50 // max(1, len(probed)))
        tag = '  <- ZPD' if zpd_low <= rate <= zpd_high else ''
        print(f'  {rate:5.2f}  {hist[rate]:5d}  {bar}{tag}')

    n_band = sum(1 for r in probed if zpd_low <= r['pass_rate'] <= zpd_high)
    n_strict = sum(1 for r in probed if 0.0 < r['pass_rate'] < 1.0)
    n_floor = sum(1 for r in probed if r['pass_rate'] == 0.0)
    n_ceil = sum(1 for r in probed if r['pass_rate'] == 1.0)
    print(f'\nZPD band [{zpd_low}, {zpd_high}]   {n_band:5d}  ({n_band / len(probed):5.1%})')
    print(f'strict 0 < p < 1        {n_strict:5d}  ({n_strict / len(probed):5.1%})')
    print(f'floor  p == 0           {n_floor:5d}  ({n_floor / len(probed):5.1%})')
    print(f'ceil   p == 1           {n_ceil:5d}  ({n_ceil / len(probed):5.1%})')

    print('\nper-task pass_rate (mean) and ZPD yield:')
    by_task: Dict[str, List[dict]] = defaultdict(list)
    for r in probed:
        by_task[r['task']].append(r)
    for task in sorted(by_task):
        rs = by_task[task]
        mean = sum(x['pass_rate'] for x in rs) / len(rs)
        band = sum(1 for x in rs if zpd_low <= x['pass_rate'] <= zpd_high)
        print(f'  {task:36s} p={mean:5.3f}  zpd={band:3d}/{len(rs):3d}')

    # ─── interpretation, so the number is not read naively ──────────
    print(f'\n{"-" * 68}')
    if n_band < 50:
        print(f'WARNING: only {n_band} items in the band. Claim 1 needs the ZPD arm to be')
        print('  big enough to build a memory bank from. Options, in order of preference:')
        print('  (a) raise k so the band resolves more finely (k=5 gives only 6 values);')
        print('  (b) widen the band, or use --zpd-strict (0<p<1);')
        print('  (c) report honestly that this student is at floor on BBEH and the')
        print('      ZPD framing does not apply at this difficulty.')
    if n_floor / len(probed) > 0.6:
        print(f'NOTE: {n_floor / len(probed):.0%} of items are never solved. A memory bank')
        print('  built from those carries teacher reasoning the student cannot yet use,')
        print('  which is exactly what hard_only is designed to test. Good for the')
        print('  contrast, bad if it swallows the whole train set.')
    if n_ceil / len(probed) > 0.5:
        print(f'NOTE: {n_ceil / len(probed):.0%} of items are always solved — headroom for')
        print('  claim 2 is small on those tasks. Consider reporting claim 2 restricted')
        print('  to the non-saturated tasks as well as overall.')
    print(f'{"-" * 68}')


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Probe student difficulty on the BBEH train split (ZPD basis)')
    p.add_argument('--model', default=config.STUDENT_MODEL)
    p.add_argument('--k', type=int, default=config.PROBE_K,
                   help='attempts per item; raising it later reuses cached attempts')
    p.add_argument('--split', default='train', choices=['train', 'test'])
    p.add_argument('--tasks', nargs='*', default=None)
    p.add_argument('--limit-per-task', type=int, default=None)
    p.add_argument('--limit-total', type=int, default=None)
    p.add_argument('--max-workers', type=int, default=4)
    p.add_argument('--temperature', type=float, default=config.PROBE_TEMPERATURE)
    p.add_argument('--max-tokens', type=int, default=config.SOLVE_MAX_TOKENS)
    p.add_argument('--dry-run', action='store_true',
                   help='fabricate responses locally; zero API spend')
    p.add_argument('--keep-errors', action='store_true',
                   help='do NOT retry attempts previously recorded as infra '
                        'errors (default is to retry them, since a cached error '
                        'permanently under-samples that item)')
    p.add_argument('--report-only', action='store_true',
                   help='re-aggregate and print from cached samples; no calls')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    run_probe(model=args.model, k=args.k, split=args.split, tasks=args.tasks,
              limit_per_task=args.limit_per_task, limit_total=args.limit_total,
              max_workers=args.max_workers, temperature=args.temperature,
              max_tokens=args.max_tokens, dry_run=args.dry_run,
              redo_errors=not args.keep_errors, report_only=args.report_only)


if __name__ == '__main__':
    main()
