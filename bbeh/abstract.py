"""
bbeh/abstract.py — turn verified concrete steps into reusable patterns.

Stage 2 of the retriever recalls from two pools: the concrete steps and their
abstractions. This module produces the second pool.

Two design decisions, both about cost and both important:

  * **One call per item, not per step.** All of an item's steps are abstracted
    together (~2.3k calls for the full train set instead of ~11k), and the
    neighbouring steps give the abstractor the context to name a move precisely
    ("the second of two passes") instead of generically.
  * **The cache is arm-independent.** Abstractions are keyed by item id and
    shared by every memory version — ``full``, ``zpd``, ``random_matched`` and
    the rest all draw from the same ``abstracts_<model>.jsonl``. Re-abstracting
    per arm would multiply the bill by the number of arms for zero benefit.

A degenerate abstractor is the specific way this stage failed before: told to
"remove all specific content", it collapses every distinct mechanism into
"gather the items -> combine them -> get the answer", after which Stage 3 cannot
tell a good precedent from a bad one and the whole retriever becomes noise. So
the report measures collapse directly — distinct-action ratio and a degenerate
phrase rate — and says so out loud when the numbers look bad.

Never drop a verified CoT because the abstractor misbehaved: the concrete step
is the ground truth, an abstraction is an enhancement. Missing abstractions are
recorded as ``None`` and the chunk still enters the concrete pool.
"""

import argparse
import logging
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence

from bbeh import config, data, jsonutil, prompts
from bbeh.api_client import GenResult, TokenUsage, build_client, dry_run_abstract

_WRITE_LOCK = threading.Lock()
_ABORT_AFTER_CONSECUTIVE_ERRORS = 10

_VALID_TYPES = set(prompts.PATTERN_TYPES)

# Phrases that signal the abstraction collapsed into a content-free shell.
_DEGENERATE_RE = re.compile(
    r'^(?:'
    r'(?:carefully\s+)?(?:analyse|analyze|examine|review|consider|look at|read)'
    r'(?:\s+the)?\s+(?:data|information|input|problem|question|items|list)'
    r'|process (?:the|it)'
    r'|(?:compute|calculate|determine|find|get|obtain) (?:the )?(?:answer|result|value)'
    r'|apply (?:the )?(?:logic|reasoning|rule)'
    r'|draw (?:a |the )?conclusion'
    r'|solve (?:the|it)'
    r')\.?$',
    re.IGNORECASE,
)
_MIN_ABSTRACT_ACTION_CHARS = 20


def abstracts_path(model: str) -> str:
    return os.path.join(config.WORK_DIR, f'abstracts_{config._slug(model)}.jsonl')


# ═════════════════════════════════════════════════════════════════════
#  Parsing / validation
# ═════════════════════════════════════════════════════════════════════

def _clean_pattern(entry: dict) -> Optional[dict]:
    """Normalise one abstraction object, or None if it is unusable."""
    if not isinstance(entry, dict):
        return None
    st = str(entry.get('abstract_state', '') or '').strip()
    ac = str(entry.get('abstract_action', '') or '').strip()
    nx = str(entry.get('abstract_next_state', '') or '').strip()
    if not ac:
        return None
    ptype = str(entry.get('pattern_type', '') or '').strip().lower().replace(' ', '_')
    coerced = ptype not in _VALID_TYPES
    return {
        'abstract_state': st,
        'abstract_action': ac,
        'abstract_next_state': nx,
        'pattern_type': ptype if not coerced else 'other',
        'pattern_type_raw': ptype if coerced else '',
        'degenerate': bool(_DEGENERATE_RE.match(ac)
                           or len(ac) < _MIN_ABSTRACT_ACTION_CHARS),
    }


def align_patterns(parsed, n_steps: int) -> List[Optional[dict]]:
    """Map the model's array onto the item's steps, positionally or by ``step``.

    Returns a list of length ``n_steps``; entries may be ``None``. A partial
    result is kept rather than discarded — one missing abstraction costs one
    chunk its abstract pool entry, whereas rejecting the item would cost the
    whole verified CoT.
    """
    out: List[Optional[dict]] = [None] * n_steps
    if not isinstance(parsed, list):
        return out

    # If every entry carries an integer ``step``, trust the labels over
    # position: models occasionally skip a step but still label the rest
    # correctly. When a label is out of range or duplicated we drop that entry
    # rather than falling back to its position — misfiling step 9's content into
    # slot 2 would attach the wrong abstraction to a concrete step, which is
    # worse than having no abstraction for it.
    entries = [e for e in parsed if isinstance(e, dict)]
    use_labels = bool(entries) and len(entries) == len(parsed) and all(
        isinstance(e.get('step'), int) and not isinstance(e.get('step'), bool)
        for e in entries
    )

    seen = set()
    for pos, entry in enumerate(parsed):
        cleaned = _clean_pattern(entry)
        if cleaned is None:
            continue
        if use_labels:
            slot = entry['step'] - 1
            if slot in seen:
                continue
            seen.add(slot)
        else:
            slot = pos
        if 0 <= slot < n_steps:
            out[slot] = cleaned
    return out


# ═════════════════════════════════════════════════════════════════════
#  One item
# ═════════════════════════════════════════════════════════════════════

def abstract_one(bank_rec: dict, client, max_attempts: int, max_tokens: int,
                 temperature: float, dry_run: bool) -> dict:
    """Abstract every step of one verified item. Never raises."""
    steps = bank_rec['steps']
    prompt = prompts.build_abstract_prompt(steps)
    rec = {
        'id': bank_rec['id'],
        'task': bank_rec['task'],
        'n_steps': len(steps),
        'patterns': [None] * len(steps),
        'n_filled': 0,
        'error': '',
        'n_attempts': 0,
        'prompt_tokens': 0,
        'completion_tokens': 0,
    }

    best: List[Optional[dict]] = [None] * len(steps)
    for attempt in range(1, max_attempts + 1):
        rec['n_attempts'] = attempt
        if dry_run:
            res: GenResult = dry_run_abstract(list(prompts.PATTERN_TYPES),
                                              n=len(steps), salt=bank_rec['id'])
        else:
            res = client.generate_detailed(prompt, max_tokens=max_tokens,
                                           temperature=temperature)
        rec['prompt_tokens'] += res.prompt_tokens
        rec['completion_tokens'] += res.completion_tokens

        if not res.ok:
            rec['error'] = res.error
            continue
        rec['error'] = ''

        parsed = jsonutil.extract_json(res.text, 'array')
        if not isinstance(parsed, list):
            parsed = jsonutil.extract_json_objects(res.text) or None
        aligned = align_patterns(parsed, len(steps))

        # Keep the best attempt so far: a retry that fills fewer slots must not
        # overwrite a better earlier one.
        if sum(x is not None for x in aligned) > sum(x is not None for x in best):
            best = aligned
        if all(x is not None for x in best):
            break

    rec['patterns'] = best
    rec['n_filled'] = sum(x is not None for x in best)
    return rec


def plan_work(bank: Sequence[dict], done: Dict[str, dict],
              redo_partial: bool) -> List[dict]:
    """Verified items still needing abstraction."""
    todo = []
    for rec in bank:
        prev = done.get(rec['id'])
        if prev is None:
            todo.append(rec)
        elif prev.get('error'):
            todo.append(rec)                                  # transient
        elif redo_partial and prev.get('n_filled', 0) < prev.get('n_steps', 0):
            todo.append(rec)
        elif prev.get('n_steps') != rec.get('n_steps'):
            # The CoT was regenerated (e.g. --retry-failed) and no longer
            # matches the cached abstraction's shape. Stale cache: redo it.
            todo.append(rec)
    return todo


# ═════════════════════════════════════════════════════════════════════
#  Driver
# ═════════════════════════════════════════════════════════════════════

def run_abstract(model: str = config.JUDGE_MODEL,
                 teacher_model: str = config.TEACHER_MODEL,
                 tasks: Optional[Sequence[str]] = None,
                 limit_total: Optional[int] = None,
                 ids_file: Optional[str] = None,
                 max_attempts: int = 2,
                 max_workers: int = 4,
                 temperature: float = config.ABSTRACT_TEMPERATURE,
                 max_tokens: int = config.ABSTRACT_MAX_TOKENS,
                 dry_run: bool = False,
                 redo_partial: bool = False,
                 report_only: bool = False) -> List[dict]:
    """Abstract every verified CoT in the teacher bank."""
    config.ensure_dirs()

    bank_label = f'DRYRUN-{teacher_model}' if dry_run else teacher_model
    bank_path = config.cot_bank_path(bank_label)
    bank = [r for r in data.read_jsonl_indexed(bank_path, key='id').values()
            if r.get('verified')]
    if tasks:
        keep = set(tasks)
        bank = [r for r in bank if r['task'] in keep]
    if ids_file:
        with open(ids_file, 'r', encoding='utf-8') as f:
            ids = {ln.strip() for ln in f if ln.strip()}
        bank = [r for r in bank if r['id'] in ids]
    bank.sort(key=lambda r: r['id'])
    if limit_total:
        bank = bank[:limit_total]
    if not bank:
        raise SystemExit(
            f'no verified CoTs in {bank_path} — run bbeh.teacher first')

    label = f'DRYRUN-{model}' if dry_run else model
    out_path = abstracts_path(label)
    done = data.read_jsonl_indexed(out_path, key='id')
    todo = plan_work(bank, done, redo_partial)

    logging.info('abstract %s: %d verified items, %d cached, %d to run%s',
                 model, len(bank), len(bank) - len(todo), len(todo),
                 '  [DRY RUN]' if dry_run else '')

    if report_only:
        report([done[r['id']] for r in bank if r['id'] in done], out_path)
        return []

    if todo:
        usage = TokenUsage()
        client = None if dry_run else build_client(
            model, temperature=temperature, max_tokens=max_tokens, usage=usage)
        n_done = n_err = 0
        consecutive_errors = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(abstract_one, r, client, max_attempts,
                                   max_tokens, temperature, dry_run): r
                       for r in todo}
            try:
                for fut in as_completed(futures):
                    rec = fut.result()
                    with _WRITE_LOCK:
                        data.append_jsonl(out_path, rec)
                    done[rec['id']] = rec
                    n_done += 1
                    if rec['error']:
                        n_err += 1
                        consecutive_errors += 1
                    else:
                        consecutive_errors = 0
                    if (consecutive_errors >= _ABORT_AFTER_CONSECUTIVE_ERRORS
                            and n_done == consecutive_errors):
                        for f in futures:
                            f.cancel()
                        raise SystemExit(
                            f'\nABORT: {consecutive_errors} consecutive failures.\n'
                            f'Last error: {rec["error"]}')
                    if n_done % 25 == 0 or n_done == len(todo):
                        eta = (time.time() - t0) / n_done * (len(todo) - n_done)
                        logging.info('  %d/%d | %d errors | ETA %.0fm',
                                     n_done, len(todo), n_err, eta / 60)
            except KeyboardInterrupt:
                logging.warning('interrupted — %d items written to %s; rerun to resume',
                                n_done, out_path)
                raise

        if not dry_run:
            usage.write(os.path.join(config.WORK_DIR,
                                     f'abstract_usage_{config._slug(model)}.json'))

    records = [done[r['id']] for r in bank if r['id'] in done]
    report(records, out_path)
    return records


# ═════════════════════════════════════════════════════════════════════
#  Report — is the abstractor actually abstracting?
# ═════════════════════════════════════════════════════════════════════

def report(records: Sequence[dict], out_path: str) -> None:
    if not records:
        print('no abstractions yet')
        return
    total_steps = sum(r['n_steps'] for r in records)
    filled = [p for r in records for p in r['patterns'] if p]
    n_filled = len(filled)

    print(f'\n{"=" * 68}\nabstractions -> {out_path}\n{"=" * 68}')
    print(f'items                 {len(records)}')
    print(f'steps                 {total_steps}')
    print(f'abstracted            {n_filled}  ({n_filled / max(1, total_steps):.1%})')
    partial = [r for r in records if r['n_filled'] < r['n_steps']]
    print(f'items partially done  {len(partial)}'
          + ('   <- rerun with --redo-partial to fill the gaps' if partial else ''))
    if not filled:
        return

    # ─── collapse diagnostics: the v8.6.2 failure mode ───────────────
    actions = [p['abstract_action'] for p in filled]
    distinct = len(set(a.lower() for a in actions))
    degenerate = sum(1 for p in filled if p['degenerate'])
    dup_top = Counter(a.lower() for a in actions).most_common(3)

    print(f'\ncollapse diagnostics (the failure mode that killed the last attempt):')
    print(f'  distinct actions      {distinct}/{n_filled}  '
          f'({distinct / n_filled:.1%} unique)')
    print(f'  degenerate actions    {degenerate}  ({degenerate / n_filled:.1%})')
    print(f'  mean action length    {sum(len(a) for a in actions) / n_filled:.0f} chars')
    if dup_top and dup_top[0][1] > 1:
        print('  most repeated actions:')
        for text, c in dup_top:
            if c > 1:
                print(f'    {c:4d}x  {text[:70]}')

    if distinct / n_filled < 0.6 or degenerate / n_filled > 0.15:
        print('\n  WARNING: the abstractions are collapsing toward a content-free')
        print('  shell. Stage 3 cannot rank precedents it cannot tell apart, so the')
        print('  retriever degrades to noise. Fix TPL_ABSTRACT_PATTERN before')
        print('  building memory versions — do not proceed on these.')

    types = Counter(p['pattern_type'] for p in filled)
    coerced = sum(1 for p in filled if p.get('pattern_type_raw'))
    print(f'\npattern_type distribution ({len(types)}/{len(prompts.PATTERN_TYPES)} '
          f'labels used, {coerced} coerced to "other"):')
    for t, c in types.most_common():
        bar = '#' * min(40, c * 40 // max(1, n_filled))
        print(f'  {t:26s} {c:5d}  {bar}')
    if types.get('other', 0) / n_filled > 0.25:
        print('  NOTE: >25% "other" — the enum does not fit BBEH\'s mechanisms well.')
        print('  Stage-1 pattern bonus will be near-useless; consider revising')
        print('  PATTERN_TYPES in prompts.py.')
    if coerced:
        raw = Counter(p['pattern_type_raw'] for p in filled if p.get('pattern_type_raw'))
        print(f'  labels invented by the model: {dict(raw.most_common(5))}')


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Abstract verified teacher CoTs into reusable patterns')
    p.add_argument('--model', default=config.JUDGE_MODEL,
                   help='abstractor model (cheap is fine)')
    p.add_argument('--teacher-model', default=config.TEACHER_MODEL,
                   help='whose CoT bank to read')
    p.add_argument('--tasks', nargs='*', default=None)
    p.add_argument('--limit-total', type=int, default=None)
    p.add_argument('--ids-file', default=None)
    p.add_argument('--max-attempts', type=int, default=2)
    p.add_argument('--max-workers', type=int, default=4)
    p.add_argument('--temperature', type=float, default=config.ABSTRACT_TEMPERATURE)
    p.add_argument('--max-tokens', type=int, default=config.ABSTRACT_MAX_TOKENS)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--redo-partial', action='store_true',
                   help='re-attempt items whose steps were only partly abstracted')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    run_abstract(model=args.model, teacher_model=args.teacher_model,
                 tasks=args.tasks, limit_total=args.limit_total,
                 ids_file=args.ids_file, max_attempts=args.max_attempts,
                 max_workers=args.max_workers, temperature=args.temperature,
                 max_tokens=args.max_tokens, dry_run=args.dry_run,
                 redo_partial=args.redo_partial, report_only=args.report_only)


if __name__ == '__main__':
    main()
