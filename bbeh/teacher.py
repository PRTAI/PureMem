"""
bbeh/teacher.py — author the reasoning that becomes memory.

BBEH ships ``{input, target}`` and nothing else: no reference solutions, no
rationales. So the content of the memory bank has to be generated, and the only
defensible way to generate it is **rejection sampling against the gold target**
— the teacher solves a train item, and its reasoning is admitted to the bank
only if its final answer matches ``target`` under the official scorer.

Why that matters: a memory bank containing confidently-wrong reasoning is worse
than an empty one, because Stage-3 will happily rank a fluent wrong precedent as
a good fit. Verification is the only thing standing between "retrieval helps"
and "retrieval injects plausible nonsense".

Each admitted item yields a list of ``(state, action, next_state)`` triples,
requested in the *same* call as the solution so we pay once rather than solving
and then re-reading the solution to decompose it.

Cost staging. ``full`` (every train item) is the most expensive artifact in the
project and is only needed for claim 1's reference arm. The ``--select`` filter
lets you buy the cheap arms first::

    # 0. what would this cost?
    python -m bbeh.teacher --select zpd --estimate

    # 1. the hypothesis arm and the pools its controls draw from
    python -m bbeh.teacher --select zpd

    # 2. only once claim 2 looks alive, buy the reference arm
    python -m bbeh.teacher --select all

Resume policy differs from ``probe.py`` on purpose:
  * infra errors are retried on the next invocation (transient);
  * items the teacher genuinely failed R times are NOT retried unless
    ``--retry-failed``. Those R failures are evidence about the item, and
    teacher tokens are the expensive kind.
"""

import argparse
import logging
import os
import threading
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

from bbeh import config, data, jsonutil, official_eval, prompts, selector
from bbeh.api_client import GenResult, TokenUsage, build_client, dry_run_teacher

_WRITE_LOCK = threading.Lock()

_ABORT_AFTER_CONSECUTIVE_ERRORS = 10

# Structural bounds on an admissible CoT.
MIN_STEPS = 2        # a 1-step "chain" teaches nothing about decomposition
MAX_STEPS = 16       # zebra puzzles legitimately need many; beyond this it is padding
MIN_ACTION_CHARS = 15

SELECT_MODES = ('all', 'zpd', 'zpd-strict', 'easy', 'hard', 'unprobed')


# ═════════════════════════════════════════════════════════════════════
#  Validation
# ═════════════════════════════════════════════════════════════════════

def validate_steps(raw) -> Tuple[Optional[List[dict]], str]:
    """Normalise and structurally validate the teacher's step list.

    Returns ``(steps, '')`` or ``(None, reason)``. Rejection reasons are
    recorded in the bank so the report can distinguish "teacher can't solve
    this" from "teacher solved it but wouldn't format it".
    """
    if not isinstance(raw, list):
        return None, 'steps_not_a_list'

    steps: List[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            return None, 'step_not_an_object'
        state = str(entry.get('state', '') or '').strip()
        action = str(entry.get('action', '') or '').strip()
        nxt = str(entry.get('next_state', '') or '').strip()
        if not (state and action and nxt):
            return None, 'step_missing_field'
        if len(action) < MIN_ACTION_CHARS:
            # "Compute it." is not a reusable reasoning move.
            return None, 'action_too_short'
        steps.append({'state': state, 'action': action, 'next_state': nxt})

    # Collapse consecutive identical actions: a repeated mechanical move should
    # be one chunk, not N copies inflating this item's memory weight (which
    # would also distort the chunk-matched controls in selector.py).
    deduped: List[dict] = []
    for s in steps:
        if deduped and s['action'] == deduped[-1]['action']:
            continue
        deduped.append(s)

    if len(deduped) < MIN_STEPS:
        return None, 'too_few_steps'
    if len(deduped) > MAX_STEPS:
        return None, 'too_many_steps'
    return deduped, ''


# ═════════════════════════════════════════════════════════════════════
#  One item
# ═════════════════════════════════════════════════════════════════════

def teach_one(item: dict, client, max_attempts: int, max_tokens: int,
              temperature: float, dry_run: bool) -> dict:
    """Rejection-sample a verified CoT for one item. Never raises."""
    prompt = prompts.TPL_TEACHER_COT.format(question=item['input'].strip())
    rec = {
        'id': item['id'],
        'task': item['task'],
        'target': item['target'],
        'verified': False,
        'reason': '',
        'error': '',
        'n_attempts': 0,
        'answer': '',
        'steps': [],
        'n_steps': 0,
        'prompt_tokens': 0,
        'completion_tokens': 0,
    }
    reasons: List[str] = []

    for attempt in range(1, max_attempts + 1):
        rec['n_attempts'] = attempt
        if dry_run:
            # Vary the step count per item, deterministically. A fixed count
            # would make every item contribute equally and leave the
            # chunk-matched controls in selector.py untested — which is exactly
            # the confound they exist to remove.
            res: GenResult = dry_run_teacher(
                item, n_steps=2 + zlib.crc32(item['id'].encode()) % 7)
        else:
            res = client.generate_detailed(prompt, max_tokens=max_tokens,
                                           temperature=temperature)
        rec['prompt_tokens'] += res.prompt_tokens
        rec['completion_tokens'] += res.completion_tokens

        if not res.ok:
            # Infra failure. Do not count it as a teacher failure — an empty
            # body says nothing about whether the teacher can solve the item.
            rec['error'] = res.error
            reasons.append('infra_error')
            continue
        rec['error'] = ''

        if res.truncated:
            reasons.append('truncated')
            continue

        payload = jsonutil.extract_json(res.text, 'object')
        if not isinstance(payload, dict):
            reasons.append('bad_json')
            continue

        answer = str(payload.get('answer', '') or '').strip()
        if not answer:
            reasons.append('no_answer')
            continue

        # ─── the gate: does the teacher's answer match gold? ─────────
        if not official_eval.evaluate_correctness(answer, item['target']):
            reasons.append('wrong_answer')
            continue

        steps, why = validate_steps(payload.get('steps'))
        if steps is None:
            reasons.append(why)
            continue

        rec.update({'verified': True, 'reason': '', 'answer': answer[:500],
                    'steps': steps, 'n_steps': len(steps),
                    'attempt_reasons': reasons})
        return rec

    rec['reason'] = reasons[-1] if reasons else 'unknown'
    rec['attempt_reasons'] = reasons
    return rec


# ═════════════════════════════════════════════════════════════════════
#  Item selection
# ═════════════════════════════════════════════════════════════════════

def filter_by_probe(items: Sequence[dict], mode: str,
                    difficulty: Dict[str, dict],
                    zpd_low: float, zpd_high: float,
                    probe_path: str = '') -> List[dict]:
    """Restrict the items to teach, so spend can be staged."""
    if mode == 'all':
        return list(items)
    if not difficulty:
        # Name the path we actually looked at. "Run the probe" is unhelpful when
        # the probe HAS been run and the reason for the miss is a model-name or
        # dry-run-prefix mismatch, which is the usual cause.
        raise SystemExit(
            f'--select {mode} needs a difficulty probe first.\n'
            f'  looked for: {probe_path or "(unknown)"}\n'
            f'  produce it: python -m bbeh.probe --model {config.STUDENT_MODEL}'
            + ('  --dry-run' if 'DRYRUN' in probe_path else '')
        )

    def rate(it):
        rec = difficulty.get(it['id'])
        if rec is None:
            return None
        r = rec.get('pass_rate')
        return None if r is None else float(r)

    if mode == 'unprobed':
        return [it for it in items if rate(it) is None]
    out = []
    for it in items:
        r = rate(it)
        if r is None:
            continue
        if mode == 'zpd' and selector.in_zpd(r, zpd_low, zpd_high):
            out.append(it)
        elif mode == 'zpd-strict' and 0.0 < r < 1.0:
            out.append(it)
        elif mode == 'easy' and r == 1.0:
            out.append(it)
        elif mode == 'hard' and r == 0.0:
            out.append(it)
    return out


def plan_work(items: Sequence[dict], done: Dict[str, dict],
              retry_failed: bool) -> List[dict]:
    """Which items still need teaching."""
    todo = []
    for it in items:
        rec = done.get(it['id'])
        if rec is None:
            todo.append(it)
        elif rec.get('verified'):
            continue
        elif rec.get('error'):
            todo.append(it)          # infra failure: always worth another go
        elif retry_failed:
            todo.append(it)          # genuine teacher failure: opt-in only
    return todo


# ═════════════════════════════════════════════════════════════════════
#  Driver
# ═════════════════════════════════════════════════════════════════════

def run_teacher(model: str = config.TEACHER_MODEL,
                student_model: str = config.STUDENT_MODEL,
                select: str = 'all',
                split: str = 'train',
                tasks: Optional[Sequence[str]] = None,
                limit_per_task: Optional[int] = None,
                limit_total: Optional[int] = None,
                ids_file: Optional[str] = None,
                max_attempts: int = 3,
                max_workers: int = 4,
                temperature: float = config.TEACHER_TEMPERATURE,
                max_tokens: int = config.TEACHER_MAX_TOKENS,
                zpd_low: float = config.ZPD_LOW,
                zpd_high: float = config.ZPD_HIGH,
                dry_run: bool = False,
                retry_failed: bool = False,
                estimate: bool = False,
                report_only: bool = False) -> List[dict]:
    """Generate and verify teacher CoTs; write ``cot_bank_<teacher>.jsonl``."""
    config.ensure_dirs()

    items = data.select_items(data.load_split(split), tasks,
                              limit_per_task, limit_total)

    # The DRYRUN- prefix has to be applied on the READ side too, matching
    # abstract.py and build_memory.py. A dry run must be hermetic: it reads the
    # fabricated difficulty its own dry probe wrote, never the real one. Reading
    # the real file here would make the dry run silently depend on a real probe
    # having been done, so it would pass on this machine and fail on a clean one.
    probe_label = f'DRYRUN-{student_model}' if dry_run else student_model
    probe_file = config.probe_path(probe_label)
    difficulty = data.read_jsonl_indexed(probe_file, key='id')
    items = filter_by_probe(items, select, difficulty, zpd_low, zpd_high, probe_file)

    if ids_file:
        with open(ids_file, 'r', encoding='utf-8') as f:
            keep = {ln.strip() for ln in f if ln.strip()}
        items = [it for it in items if it['id'] in keep]

    if not items:
        raise SystemExit(f'no items selected (--select {select}) — nothing to teach')

    # Same dry-run isolation as probe.py: fabricated CoTs must never land in
    # the path a real run will treat as cache.
    label = f'DRYRUN-{model}' if dry_run else model
    bank_path = config.cot_bank_path(label)
    done = data.read_jsonl_indexed(bank_path, key='id')
    todo = plan_work(items, done, retry_failed)

    mean_chars = sum(len(it['input']) for it in items) / max(1, len(items))
    logging.info('teacher %s: %d items selected (--select %s), %d already banked, '
                 '%d to run%s', model, len(items), select,
                 len(items) - len(todo), len(todo), '  [DRY RUN]' if dry_run else '')

    if estimate:
        _print_estimate(model, todo, mean_chars, max_attempts, max_tokens)
        return []

    if report_only:
        records = [done[it['id']] for it in items if it['id'] in done]
        report(records, bank_path, difficulty)
        return records

    if todo:
        usage = TokenUsage()
        client = None if dry_run else build_client(
            model, temperature=temperature, max_tokens=max_tokens, usage=usage)

        n_done = n_ok = n_err = 0
        consecutive_errors = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(teach_one, it, client, max_attempts,
                                   max_tokens, temperature, dry_run): it
                       for it in todo}
            try:
                for fut in as_completed(futures):
                    rec = fut.result()
                    with _WRITE_LOCK:
                        data.append_jsonl(bank_path, rec)
                    done[rec['id']] = rec

                    n_done += 1
                    if rec['verified']:
                        n_ok += 1
                        consecutive_errors = 0
                    elif rec['error']:
                        n_err += 1
                        consecutive_errors += 1
                    else:
                        consecutive_errors = 0

                    if (consecutive_errors >= _ABORT_AFTER_CONSECUTIVE_ERRORS
                            and n_ok == 0):
                        for f in futures:
                            f.cancel()
                        raise SystemExit(
                            f'\nABORT: {consecutive_errors} consecutive infra failures, '
                            f'0 verified.\nLast error: {rec["error"]}\n'
                            f'Check the teacher model name ({model!r}) and credentials.'
                        )

                    if n_done % 10 == 0 or n_done == len(todo):
                        eta = (time.time() - t0) / n_done * (len(todo) - n_done)
                        logging.info('  %d/%d | verified %d (%.1f%%) | %d infra err '
                                     '| ETA %.0fm', n_done, len(todo), n_ok,
                                     100 * n_ok / n_done, n_err, eta / 60)
            except KeyboardInterrupt:
                logging.warning('interrupted — %d items already banked at %s; '
                                'rerun to resume', n_done, bank_path)
                raise

        if not dry_run:
            usage.write(os.path.join(config.WORK_DIR,
                                     f'teacher_usage_{config._slug(model)}.json'))

    records = [done[it['id']] for it in items if it['id'] in done]
    report(records, bank_path, difficulty)
    return records


def _print_estimate(model: str, todo: Sequence[dict], mean_chars: float,
                    max_attempts: int, max_tokens: int) -> None:
    """Projected token spend. Rough, but the right order of magnitude."""
    n = len(todo)
    chars_per_token = 3.6
    prompt_tok = sum(len(it['input']) for it in todo) / chars_per_token
    prompt_tok += n * 600 / chars_per_token          # the instruction block
    # Most items verify on attempt 1; assume 1.35 attempts on average, and that
    # a structured CoT uses roughly a third of the ceiling.
    exp_attempts = 1.35
    completion_tok = n * exp_attempts * max_tokens * 0.33

    print(f'\n{"=" * 60}\nteacher cost estimate — {model}\n{"=" * 60}')
    print(f'items to run          {n}')
    print(f'mean input chars      {mean_chars:,.0f}')
    print(f'expected attempts     {exp_attempts:.2f} / item (max {max_attempts})')
    print(f'prompt tokens        ~{prompt_tok * exp_attempts / 1e6:.2f}M')
    print(f'completion tokens    ~{completion_tok / 1e6:.2f}M')
    print('\nMultiply by your per-token price. Nothing was called.')
    print(f'{"=" * 60}\n')


# ═════════════════════════════════════════════════════════════════════
#  Report
# ═════════════════════════════════════════════════════════════════════

def report(records: Sequence[dict], bank_path: str,
           difficulty: Optional[Dict[str, dict]] = None,
           zpd_low: float = config.ZPD_LOW,
           zpd_high: float = config.ZPD_HIGH) -> None:
    n = len(records)
    if not n:
        print('no records in the bank yet')
        return
    ok = [r for r in records if r.get('verified')]
    steps = [r['n_steps'] for r in ok]

    print(f'\n{"=" * 68}\nteacher bank -> {bank_path}\n{"=" * 68}')
    print(f'items attempted       {n}')
    print(f'verified CoTs         {len(ok)}  ({len(ok) / n:.1%})   <- the memory pool')
    print(f'total memory chunks   {sum(steps)}')
    if steps:
        srt = sorted(steps)
        print(f'steps per item        mean {sum(steps) / len(steps):.2f}  '
              f'median {srt[len(srt) // 2]}  min {srt[0]}  max {srt[-1]}')

    fails = Counter(r.get('reason', '') for r in records if not r.get('verified'))
    if fails:
        print('\nrejection reasons:')
        for reason, c in fails.most_common():
            note = ''
            if reason == 'wrong_answer':
                note = '   (teacher could not solve it — a real ceiling)'
            elif reason in ('bad_json', 'no_answer', 'steps_not_a_list',
                            'step_missing_field', 'action_too_short',
                            'too_few_steps', 'too_many_steps'):
                note = '   (format, not capability — fixable via the prompt)'
            elif reason == 'truncated':
                note = '   (raise TEACHER_MAX_TOKENS)'
            elif reason == 'infra_error':
                note = '   (rerun to retry)'
            print(f'  {reason or "(none)":22s} {c:5d}{note}')

    print('\nper-task verification rate:')
    by_task: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_task[r['task']].append(r)
    for task in sorted(by_task):
        rs = by_task[task]
        v = sum(1 for x in rs if x.get('verified'))
        ns = [x['n_steps'] for x in rs if x.get('verified')]
        tail = f'  mean_steps={sum(ns) / len(ns):5.2f}' if ns else ''
        print(f'  {task:36s} {v:3d}/{len(rs):3d} ({v / len(rs):5.1%}){tail}')

    # ─── the cross-tab that decides whether the arms are buildable ────
    if difficulty:
        print(f'\n{"-" * 68}')
        print('teacher success vs STUDENT difficulty '
              '(does the pool survive in each band?)')
        bands = [('floor  p == 0', lambda p: p == 0.0),
                 (f'ZPD [{zpd_low},{zpd_high}]', lambda p: zpd_low <= p <= zpd_high),
                 ('ceil   p == 1', lambda p: p == 1.0)]
        for name, pred in bands:
            rs = []
            for r in records:
                d = difficulty.get(r['id'])
                if not d or d.get('pass_rate') is None:
                    continue
                if pred(float(d['pass_rate'])):
                    rs.append(r)
            if not rs:
                print(f'  {name:16s} no probed items')
                continue
            v = sum(1 for x in rs if x.get('verified'))
            ch = sum(x['n_steps'] for x in rs if x.get('verified'))
            print(f'  {name:16s} {v:4d}/{len(rs):4d} verified ({v / len(rs):5.1%}), '
                  f'{ch:5d} chunks')
        print('  If the floor band verifies far worse than the ZPD band, the')
        print('  hard_only control will be small — say so rather than comparing')
        print('  a 900-chunk zpd arm against a 200-chunk hard_only arm.')
        print(f'{"-" * 68}')

    _verdict(n, len(ok), fails)


# Rejection reasons grouped by what the remedy is. The grouping is the whole
# point: a 40% verification rate made of format errors is a prompt bug worth
# twenty minutes, and a 40% rate made of wrong answers is a teacher that cannot
# out-solve the student. Identical headline number, opposite decisions — and the
# flat reason table above leaves the reader to work that out.
_CAPABILITY = {'wrong_answer'}
_FORMAT = {'bad_json', 'no_answer', 'steps_not_a_list', 'step_missing_field',
           'action_too_short', 'too_few_steps', 'too_many_steps'}
_TRANSIENT = {'infra_error', 'truncated'}

VERIFY_FLOOR = 0.50


def _verdict(n: int, n_ok: int, fails: Counter) -> None:
    """State whether this bank is worth building on, and what to change if not.

    Exists because the numbers above are individually readable and jointly easy
    to misread. The failure this guards is spending the dominant line of the
    budget on a teacher that is a peer of the student: rejection sampling
    re-spends on every failure, so a weak teacher costs *more* than a strong one
    while producing a thinner bank.
    """
    rate = n_ok / n if n else 0.0
    cap = sum(c for r, c in fails.items() if r in _CAPABILITY)
    fmt = sum(c for r, c in fails.items() if r in _FORMAT)
    tra = sum(c for r, c in fails.items() if r in _TRANSIENT)
    other = sum(fails.values()) - cap - fmt - tra

    print(f'\n{"=" * 68}\nVERDICT\n{"=" * 68}')
    print(f'verification {rate:.1%} of {n} items'
          f'   (floor for a fundable full run: {VERIFY_FLOOR:.0%})')
    if sum(fails.values()):
        parts = [f'{k} {v}' for k, v in
                 (('capability', cap), ('format', fmt), ('transient', tra),
                  ('unclassified', other)) if v]
        print('rejections:  ' + ',  '.join(parts))

    if rate >= VERIFY_FLOOR and cap <= fmt:
        print('\nOK — the pool is thick enough and failures are not dominated by')
        print('the teacher being unable to solve the items. Proceed.')
    elif tra > cap + fmt:
        print('\nINCONCLUSIVE — most rejections are transient (infra or truncation),')
        print('so this rate is not a measurement of the teacher yet. Rerun to retry')
        print('the infra errors, raise TEACHER_MAX_TOKENS for the truncations, then')
        print('read this block again.')
    elif fmt > cap:
        print('\nFIXABLE — failures are mostly format, not capability. The teacher can')
        print('solve these items but is not emitting the schema. Fix the prompt and')
        print('rerun with --retry-failed; do not upgrade the teacher for this.')
    else:
        print('\nTEACHER TOO WEAK — most rejections are wrong answers, meaning it')
        print('cannot out-solve the student on these items. This is the one failure')
        print('that spending more does not fix: rejection sampling re-spends on every')
        print('failure, so a peer teacher is more expensive than a stronger one AND')
        print('yields a thinner bank. Raise BBEH_TEACHER_MODEL a tier before funding')
        print('the full pass. Note the caveat this puts on claim 2 either way: the')
        print('harness supplies the student with a stronger model\'s traces, so a')
        print('teacher at the student\'s own level has little to transfer.')
    print('=' * 68)


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Generate target-verified teacher CoTs for the memory bank')
    p.add_argument('--model', default=config.TEACHER_MODEL)
    p.add_argument('--student-model', default=config.STUDENT_MODEL,
                   help='whose probe defines the ZPD band for --select')
    p.add_argument('--select', default='all', choices=SELECT_MODES,
                   help='stage spend: teach only the items an arm needs')
    p.add_argument('--split', default='train', choices=['train', 'test'])
    p.add_argument('--tasks', nargs='*', default=None)
    p.add_argument('--limit-per-task', type=int, default=None)
    p.add_argument('--limit-total', type=int, default=None)
    p.add_argument('--ids-file', default=None,
                   help='newline-separated item ids to restrict to')
    p.add_argument('--max-attempts', type=int, default=3,
                   help='rejection-sampling budget per item')
    p.add_argument('--max-workers', type=int, default=4)
    p.add_argument('--temperature', type=float, default=config.TEACHER_TEMPERATURE)
    p.add_argument('--max-tokens', type=int, default=config.TEACHER_MAX_TOKENS)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--retry-failed', action='store_true',
                   help='re-attempt items the teacher previously failed to solve')
    p.add_argument('--estimate', action='store_true',
                   help='print a token estimate and exit without calling anything')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    run_teacher(model=args.model, student_model=args.student_model,
                select=args.select, split=args.split, tasks=args.tasks,
                limit_per_task=args.limit_per_task, limit_total=args.limit_total,
                ids_file=args.ids_file, max_attempts=args.max_attempts,
                max_workers=args.max_workers, temperature=args.temperature,
                max_tokens=args.max_tokens, dry_run=args.dry_run,
                retry_failed=args.retry_failed, estimate=args.estimate,
                report_only=args.report_only)


if __name__ == '__main__':
    main()
