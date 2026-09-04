"""
bbeh/data.py — BBEH task loading and train/test splitting.

BBEH ships as 23 eval-only task files of ``{"examples": [{"input", "target"}],
"canary": ...}``. There is no official train split and no reference reasoning,
so we carve our own split. Two properties matter more than anything else here:

  * **Disjointness.** The memory bank is built from train and evaluated on
    test. Any overlap makes claim 2 vacuous (the harness would be retrieving
    the answer). We enforce it twice: by index, and by exact input hash.
  * **Determinism.** The split is a function of ``(seed, task)`` only, so it
    reproduces regardless of dict ordering, filesystem order, or PYTHONHASHSEED
    (``random.Random`` seeded with a *string* is stable across processes).

Item schema used everywhere downstream::

    {"id": "bbeh_word_sorting#0042",   # task + original index, stable & auditable
     "task": "bbeh_word_sorting",
     "input": "...",                    # verbatim from task.json
     "target": "2"}                     # verbatim from task.json
"""

import argparse
import hashlib
import json
import logging
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

from bbeh import config


# ═════════════════════════════════════════════════════════════════════
#  Loading the vendored benchmark
# ═════════════════════════════════════════════════════════════════════

def list_tasks() -> List[str]:
    """Sorted list of BBEH task names (directory names under benchmark_tasks/)."""
    if not os.path.isdir(config.BBEH_TASKS_DIR):
        raise FileNotFoundError(
            f'BBEH tasks not found at {config.BBEH_TASKS_DIR}. '
            'Expected the vendored bbeh-main/ checkout at the repo root.'
        )
    names = []
    for name in sorted(os.listdir(config.BBEH_TASKS_DIR)):
        if os.path.isfile(os.path.join(config.BBEH_TASKS_DIR, name, 'task.json')):
            names.append(name)
    return names


def load_task(task: str) -> List[dict]:
    """Load one task's examples as harness items, in original file order."""
    path = os.path.join(config.BBEH_TASKS_DIR, task, 'task.json')
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    examples = payload.get('examples') or []
    items = []
    for idx, ex in enumerate(examples):
        items.append({
            'id': f'{task}#{idx:04d}',
            'task': task,
            'input': ex['input'],
            'target': ex['target'],
        })
    return items


def load_all_tasks(tasks: Optional[Sequence[str]] = None) -> Dict[str, List[dict]]:
    """``{task: [item, ...]}`` for the requested tasks (default: all 23)."""
    return {t: load_task(t) for t in (tasks or list_tasks())}


# ═════════════════════════════════════════════════════════════════════
#  Splitting
# ═════════════════════════════════════════════════════════════════════

def _input_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()[:16]


def build_splits(seed: int = config.SPLIT_SEED,
                 train_per_task: int = config.TRAIN_PER_TASK,
                 test_per_task: int = config.TEST_PER_TASK,
                 tasks: Optional[Sequence[str]] = None) -> dict:
    """Carve a deterministic, disjoint per-task train/test split and write it.

    Per task: shuffle the example indices with a seed derived from
    ``f'{seed}|{task}'``, then take ``train_per_task`` for train and
    ``test_per_task`` for test from the *remaining* items. Tasks with fewer
    than ``train+test`` examples (bbeh_disambiguation_qa has 120) are split
    proportionally, half and half, so train never eats into test.

    Returns the metadata dict that is also written to ``split_meta.json``.
    """
    config.ensure_dirs()
    task_names = list(tasks or list_tasks())

    train_items: List[dict] = []
    test_items: List[dict] = []
    per_task_meta = {}

    for task in task_names:
        items = load_task(task)
        n = len(items)

        # Deterministic permutation from a string seed: stable across processes
        # and unaffected by PYTHONHASHSEED.
        order = list(range(n))
        random.Random(f'{seed}|{task}').shuffle(order)

        want_train, want_test = train_per_task, test_per_task
        if want_train + want_test > n:
            # Not enough examples: halve, favouring test (it is what we report).
            want_test = min(want_test, n // 2)
            want_train = min(want_train, n - want_test)

        train_idx = order[:want_train]
        test_idx = order[want_train:want_train + want_test]

        for i in train_idx:
            train_items.append(dict(items[i], split='train'))
        for i in test_idx:
            test_items.append(dict(items[i], split='test'))

        per_task_meta[task] = {
            'n_available': n,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
        }

    # ─── Guard 1: index-level disjointness ───────────────────────────
    train_ids = {it['id'] for it in train_items}
    test_ids = {it['id'] for it in test_items}
    id_overlap = train_ids & test_ids
    if id_overlap:
        raise AssertionError(
            f'train/test id overlap ({len(id_overlap)} items), e.g. '
            f'{sorted(id_overlap)[:3]} — split logic is broken'
        )

    # ─── Guard 2: exact-input disjointness ───────────────────────────
    # Distinct indices can still carry byte-identical inputs (duplicated
    # examples inside a task). Those would leak the answer into memory, so we
    # drop the *train* copy and keep test intact.
    test_hashes = {_input_hash(it['input']) for it in test_items}
    kept, dropped = [], []
    for it in train_items:
        if _input_hash(it['input']) in test_hashes:
            dropped.append(it['id'])
        else:
            kept.append(it)
    if dropped:
        logging.warning(
            'Dropped %d train items whose input is byte-identical to a test '
            'item (duplicate examples in the source task files): %s%s',
            len(dropped), dropped[:5], ' ...' if len(dropped) > 5 else ''
        )
        for it_id in dropped:
            task = it_id.split('#')[0]
            per_task_meta[task]['n_train'] -= 1
            per_task_meta[task].setdefault('n_train_dropped_dup', 0)
            per_task_meta[task]['n_train_dropped_dup'] += 1
    train_items = kept

    # Also report intra-train duplicates: they inflate a "size-matched" subset
    # with redundant memory, which matters for the claim-1 controls.
    train_dup = sum(
        c - 1 for c in Counter(_input_hash(it['input']) for it in train_items).values()
        if c > 1
    )

    write_jsonl(config.TRAIN_JSONL, train_items)
    write_jsonl(config.TEST_JSONL, test_items)

    meta = {
        'seed': seed,
        'train_per_task': train_per_task,
        'test_per_task': test_per_task,
        'n_tasks': len(task_names),
        'n_train': len(train_items),
        'n_test': len(test_items),
        'n_train_dropped_as_test_duplicate': len(dropped),
        'n_intra_train_duplicate_inputs': train_dup,
        'per_task': per_task_meta,
    }
    with open(config.SPLIT_META_JSON, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logging.info('Split: %d train / %d test across %d tasks -> %s',
                 len(train_items), len(test_items), len(task_names), config.SPLITS_DIR)
    return meta


# ═════════════════════════════════════════════════════════════════════
#  Reading splits back
# ═════════════════════════════════════════════════════════════════════

def load_split(name: str) -> List[dict]:
    """Load ``'train'`` or ``'test'``. Raises if the split hasn't been built."""
    path = {'train': config.TRAIN_JSONL, 'test': config.TEST_JSONL}[name]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found. Run:  python -m bbeh.data build-splits'
        )
    return read_jsonl(path)


def select_items(items: Sequence[dict],
                 tasks: Optional[Sequence[str]] = None,
                 limit_per_task: Optional[int] = None,
                 limit_total: Optional[int] = None) -> List[dict]:
    """Filter a split down to a pilot subset, deterministically.

    ``limit_per_task`` keeps the FIRST n items of each task *as they appear in
    the split file* — the split file order is already the shuffled order, so
    this is an unbiased sample, and it is nested: the 30-per-task pilot is a
    strict subset of the 100-per-task full run. That nesting is what lets a
    pilot's cached solves be reused verbatim when scaling up.
    """
    out = list(items)
    if tasks:
        keep = set(tasks)
        out = [it for it in out if it['task'] in keep]
    if limit_per_task is not None:
        seen = defaultdict(int)
        limited = []
        for it in out:
            if seen[it['task']] < limit_per_task:
                limited.append(it)
                seen[it['task']] += 1
        out = limited
    if limit_total is not None:
        out = out[:limit_total]
    return out


# ═════════════════════════════════════════════════════════════════════
#  Embedding text
# ═════════════════════════════════════════════════════════════════════

def embed_text(item_or_input) -> str:
    """Text handed to the sentence encoder for a BBEH item.

    MiniLM truncates at ~256 word-pieces, and BBEH inputs run to 32k chars, so
    embedding the raw input would encode nothing but the task preamble — which
    is *identical* across all 200 items of a task and therefore carries zero
    discriminative signal. We take a head+tail window: the head keeps the task
    framing, the tail keeps the actual question, which in BBEH is almost always
    last.
    """
    text = item_or_input if isinstance(item_or_input, str) else item_or_input['input']
    text = (text or '').strip()
    head_n, tail_n = config.EMBED_HEAD_CHARS, config.EMBED_TAIL_CHARS
    if len(text) <= head_n + tail_n:
        return text
    return text[:head_n].strip() + '\n...\n' + text[-tail_n:].strip()


# ═════════════════════════════════════════════════════════════════════
#  JSONL helpers  (always explicit UTF-8 — Windows defaults to GBK and
#  silently corrupts non-ASCII targets such as bbeh_linguini's "els llaços")
# ═════════════════════════════════════════════════════════════════════

def read_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def read_jsonl_indexed(path: str, key: str = 'id') -> Dict[str, dict]:
    """Read a JSONL keyed by ``key``; later records win (append-log semantics)."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if key in rec:
                out[rec[key]] = rec
    return out


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='BBEH data / split utilities')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_build = sub.add_parser('build-splits', help='carve deterministic train/test splits')
    p_build.add_argument('--seed', type=int, default=config.SPLIT_SEED)
    p_build.add_argument('--train-per-task', type=int, default=config.TRAIN_PER_TASK)
    p_build.add_argument('--test-per-task', type=int, default=config.TEST_PER_TASK)
    p_build.add_argument('--tasks', nargs='*', default=None)

    sub.add_parser('stats', help='print vendored benchmark statistics')
    sub.add_parser('verify', help='re-check an existing split for leakage')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    if args.cmd == 'build-splits':
        meta = build_splits(args.seed, args.train_per_task,
                            args.test_per_task, args.tasks)
        print(json.dumps({k: v for k, v in meta.items() if k != 'per_task'}, indent=2))
        print(f'\nper-task: {len(meta["per_task"])} tasks')
        for task, m in list(meta['per_task'].items())[:5]:
            print(f'  {task:34s} avail={m["n_available"]:4d} '
                  f'train={m["n_train"]:4d} test={m["n_test"]:4d}')
        print('  ...')

    elif args.cmd == 'stats':
        total = 0
        for task in list_tasks():
            items = load_task(task)
            total += len(items)
            lens = [len(it['input']) for it in items]
            print(f'{task:34s} n={len(items):4d} '
                  f'input_chars med={sorted(lens)[len(lens)//2]:6d} max={max(lens):6d}')
        print(f'\nTOTAL {total} examples across {len(list_tasks())} tasks')

    elif args.cmd == 'verify':
        train, test = load_split('train'), load_split('test')
        tr_ids, te_ids = {i['id'] for i in train}, {i['id'] for i in test}
        tr_h = {_input_hash(i['input']) for i in train}
        te_h = {_input_hash(i['input']) for i in test}
        print(f'train={len(train)}  test={len(test)}')
        print(f'id overlap:          {len(tr_ids & te_ids)}   (must be 0)')
        print(f'input-hash overlap:  {len(tr_h & te_h)}   (must be 0)')
        by_task = Counter(i['task'] for i in test)
        print(f'tasks in test: {len(by_task)}; '
              f'min/max per task: {min(by_task.values())}/{max(by_task.values())}')
        ok = not (tr_ids & te_ids) and not (tr_h & te_h)
        print('VERDICT:', 'OK' if ok else 'LEAKAGE DETECTED')
        raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
