"""
bbeh/build_memory.py — assemble a memory version directory. No API calls.

Inputs (all cached, all arm-independent):
    splits/train.jsonl                  the items
    work/difficulty_<student>.jsonl     pass_rate, from probe.py
    work/cot_bank_<teacher>.jsonl       verified CoTs, from teacher.py
    work/abstracts_<model>.jsonl        patterns, from abstract.py

Output — one directory per experimental arm::

    memory_banks/versions/<version_id>/
        demos.jsonl                 row i  <-> source_idx i  <-> train item
        question_embeddings.npy     (n_demos,   D)   row i = demo i's question
        memory.jsonl                row j  <-> chunk_id j
        embeddings.npy              (n_chunks,  D)   row j = chunk j
        abstract_memory.jsonl       row k  <-> abstract_id k
        abstract_embeddings.npy     (n_abstr,   D)   row k = abstract k
        meta.json                   what this arm is and how it was selected

**The alignment invariant.** Retrieval mixes three index spaces, and a silent
off-by-one between them produces a retriever that returns confident nonsense
while every score still looks plausible — the hardest kind of bug to notice from
outputs alone. So it is stated once, enforced on write, and re-checkable::

    memory.jsonl[j]['chunk_id']            == j
    memory.jsonl[j]['source_idx']          == i  where demos.jsonl[i]['id'] is
                                                 the train item chunk j came from
    demos.jsonl[i]['source_idx']           == i
    abstract_memory.jsonl[k]['abstract_id']== k
    abstract_memory.jsonl[k]['chunk_id']   == some j, and its item_id agrees
    question_embeddings.npy.shape[0]       == len(demos.jsonl)
    embeddings.npy.shape[0]                == len(memory.jsonl)
    abstract_embeddings.npy.shape[0]       == len(abstract_memory.jsonl)

``verify_version()`` asserts all of it and is run automatically after every
build, and available as ``python -m bbeh.build_memory verify <version_id>``.

Because this stage costs nothing, build every arm here and spend only on the
eval. Size-matched controls read their target from the arm they match::

    python -m bbeh.build_memory build --method zpd            --version-id zpd
    python -m bbeh.build_memory build --method random_matched --version-id rnd \\
                                      --match-version zpd --match-on chunks
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from bbeh import abstract as abstract_mod
from bbeh import config, data, selector


# ═════════════════════════════════════════════════════════════════════
#  Embedding
# ═════════════════════════════════════════════════════════════════════

class Embedder:
    """SentenceTransformer wrapper producing L2-normalised float32 rows.

    Normalising on write means every downstream similarity is a plain dot
    product, so no stage can accidentally compare a normalised query against
    unnormalised memory (which silently ranks by vector magnitude).
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        logging.info('loading embedder %s', model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(list(texts), batch_size=batch_size,
                                 show_progress_bar=len(texts) > 500,
                                 convert_to_numpy=True,
                                 normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


class HashEmbedder:
    """Deterministic fake embedder — DRY RUN ONLY.

    Exists so the assembly, alignment and retrieval plumbing can be exercised
    without torch. The vectors carry a little lexical signal (hashed token
    buckets) so retrieval returns *something* stable, but they are NOT
    semantic. Any version built with this must live under a ``DRYRUN-`` id so
    it can never be mistaken for a real bank.
    """

    def __init__(self, dim: int = config.EMBEDDING_DIM):
        self.dim = dim
        self.model_name = f'HASH-FAKE-{dim}d'

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        import zlib
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in (text or '').lower().split():
                h = zlib.crc32(tok.encode())
                out[i, h % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


def chunk_text(chunk: dict) -> str:
    """Text embedded for the concrete pool."""
    return f"{chunk['state']} -> {chunk['action']} -> {chunk['next_state']}"


def abstract_text(pattern: dict) -> str:
    """Text embedded for the abstract pool."""
    return (f"{pattern['abstract_state']} -> {pattern['abstract_action']}"
            f" -> {pattern['abstract_next_state']}")


# ═════════════════════════════════════════════════════════════════════
#  Pool assembly
# ═════════════════════════════════════════════════════════════════════

def load_pool(student_model: str, teacher_model: str, abstractor_model: str,
              tasks: Optional[Sequence[str]] = None,
              dry_run: bool = False) -> Tuple[List[dict], Dict[str, dict],
                                              Dict[str, dict], Dict[str, dict]]:
    """``(pool, difficulty, bank, abstracts)``.

    ``pool`` holds one entry per train item that has a verified CoT, carrying
    the ``n_steps`` that ``selector`` needs to size-match arms.
    """
    lab = (lambda m: f'DRYRUN-{m}') if dry_run else (lambda m: m)

    train = {it['id']: it for it in data.load_split('train')}
    difficulty = data.read_jsonl_indexed(config.probe_path(lab(student_model)), key='id')
    bank = {k: v for k, v in
            data.read_jsonl_indexed(config.cot_bank_path(lab(teacher_model)),
                                    key='id').items()
            if v.get('verified')}
    abstracts = data.read_jsonl_indexed(
        abstract_mod.abstracts_path(lab(abstractor_model)), key='id')

    if not bank:
        raise SystemExit('no verified CoTs — run bbeh.teacher first')

    pool = []
    for item_id, rec in bank.items():
        if item_id not in train:
            # A CoT for an item that is not in the current train split: the
            # split was rebuilt with a different seed. Refuse to guess.
            raise SystemExit(
                f'CoT bank contains {item_id!r}, which is not in the current '
                'train split. The split was rebuilt since the bank was made — '
                'either restore the old split or regenerate the bank.'
            )
        if tasks and rec['task'] not in set(tasks):
            continue
        pool.append({
            'id': item_id,
            'task': rec['task'],
            'n_steps': int(rec['n_steps']),
        })
    pool.sort(key=lambda r: r['id'])          # deterministic before any shuffle
    return pool, difficulty, bank, abstracts


def assemble(selected: Sequence[dict], bank: Dict[str, dict],
             abstracts: Dict[str, dict],
             train: Dict[str, dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    """Build ``(demos, chunks, abstract_records)`` with indices already assigned.

    This function is the sole place where ``source_idx``, ``chunk_id`` and
    ``abstract_id`` are minted, which is why the invariant is checkable at all.
    """
    demos: List[dict] = []
    chunks: List[dict] = []
    abstract_records: List[dict] = []

    for source_idx, sel in enumerate(sorted(selected, key=lambda r: r['id'])):
        item_id = sel['id']
        rec = bank[item_id]
        item = train[item_id]

        demos.append({
            'source_idx': source_idx,
            'id': item_id,
            'task': rec['task'],
            'n_steps': int(rec['n_steps']),
            # Kept for auditing only; retrieval uses question_embeddings.npy.
            'question_head': item['input'].strip()[:300],
        })

        patterns = (abstracts.get(item_id) or {}).get('patterns') or []
        for step_idx, step in enumerate(rec['steps']):
            pattern = patterns[step_idx] if step_idx < len(patterns) else None
            chunk_id = len(chunks)
            chunks.append({
                'chunk_id': chunk_id,
                'source_idx': source_idx,
                'item_id': item_id,
                'task': rec['task'],
                'step_idx': step_idx,
                'n_steps_in_item': int(rec['n_steps']),
                'state': step['state'],
                'action': step['action'],
                'next_state': step['next_state'],
                # Denormalised onto the chunk so Stage 1's pattern bonus and the
                # prompt formatter never need a second lookup.
                'pattern_type': (pattern or {}).get('pattern_type', ''),
                'abstract_pattern': pattern,
            })
            if pattern:
                abstract_records.append({
                    'abstract_id': len(abstract_records),
                    'chunk_id': chunk_id,
                    'source_idx': source_idx,
                    'item_id': item_id,
                    'task': rec['task'],
                    'step_idx': step_idx,
                    'abstract_state': pattern['abstract_state'],
                    'abstract_action': pattern['abstract_action'],
                    'abstract_next_state': pattern['abstract_next_state'],
                    'pattern_type': pattern['pattern_type'],
                })

    return demos, chunks, abstract_records


# ═════════════════════════════════════════════════════════════════════
#  Build
# ═════════════════════════════════════════════════════════════════════

def build_version(version_id: str,
                  method: str = 'zpd',
                  student_model: str = config.STUDENT_MODEL,
                  teacher_model: str = config.TEACHER_MODEL,
                  abstractor_model: str = config.JUDGE_MODEL,
                  tasks: Optional[Sequence[str]] = None,
                  size: Optional[int] = None,
                  match_version: Optional[str] = None,
                  match_on: str = 'chunks',
                  seed: int = config.SPLIT_SEED,
                  zpd_low: float = config.ZPD_LOW,
                  zpd_high: float = config.ZPD_HIGH,
                  zpd_strict: bool = False,
                  balance_tasks: bool = False,
                  dry_run: bool = False,
                  overwrite: bool = False) -> str:
    """Select an arm, embed it, write the version directory, verify it."""
    config.ensure_dirs()
    if dry_run and not version_id.startswith('DRYRUN-'):
        version_id = f'DRYRUN-{version_id}'

    vdir = config.version_dir(version_id)
    if os.path.exists(os.path.join(vdir, config.META_JSON_NAME)) and not overwrite:
        raise SystemExit(
            f'{version_id} already exists at {vdir}\n'
            'Pass --overwrite to rebuild it. (Rebuilding invalidates any run '
            'that used it — rerun those with a fresh --run-label.)'
        )

    pool, difficulty, bank, abstracts = load_pool(
        student_model, teacher_model, abstractor_model, tasks, dry_run)
    train = {it['id']: it for it in data.load_split('train')}

    # ─── size-match against a previously built arm ───────────────────
    if match_version and size is None:
        ref_meta_path = os.path.join(config.version_dir(match_version),
                                     config.META_JSON_NAME)
        if not os.path.exists(ref_meta_path):
            raise SystemExit(f'--match-version {match_version}: no meta.json at '
                             f'{ref_meta_path}; build that arm first')
        with open(ref_meta_path, 'r', encoding='utf-8') as f:
            ref = json.load(f)
        size = ref['n_chunks'] if match_on == 'chunks' else ref['n_demos']
        logging.info('size-matching %s to %s: %d %s',
                     version_id, match_version, size, match_on)

    selected, sel_info = selector.select_subset(
        method, pool, difficulty, size=size, seed=seed,
        zpd_low=zpd_low, zpd_high=zpd_high, zpd_strict=zpd_strict,
        balance_tasks=balance_tasks, match_on=match_on)
    if not selected:
        raise SystemExit(
            f'method {method!r} selected 0 items. Check the probe: '
            f'python -m bbeh.probe --report-only')

    demos, chunks, abstract_records = assemble(selected, bank, abstracts, train)

    # ─── embed the three pools ───────────────────────────────────────
    embedder = HashEmbedder() if dry_run else Embedder()
    t0 = time.time()
    q_emb = embedder.encode([data.embed_text(train[d['id']]) for d in demos])
    c_emb = embedder.encode([chunk_text(c) for c in chunks])
    a_emb = embedder.encode([abstract_text(a) for a in abstract_records])
    logging.info('embedded %d questions / %d chunks / %d abstracts in %.1fs',
                 len(q_emb), len(c_emb), len(a_emb), time.time() - t0)

    # ─── write ───────────────────────────────────────────────────────
    os.makedirs(vdir, exist_ok=True)
    data.write_jsonl(os.path.join(vdir, config.DEMOS_JSONL_NAME), demos)
    data.write_jsonl(os.path.join(vdir, config.MEMORY_JSONL_NAME), chunks)
    data.write_jsonl(os.path.join(vdir, config.ABSTRACT_MEMORY_JSONL_NAME),
                     abstract_records)
    np.save(os.path.join(vdir, config.QUESTION_EMBEDDINGS_NPY_NAME), q_emb)
    np.save(os.path.join(vdir, config.EMBEDDINGS_NPY_NAME), c_emb)
    np.save(os.path.join(vdir, config.ABSTRACT_EMBEDDINGS_NPY_NAME), a_emb)

    meta = {
        'version_id': version_id,
        'method': method,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'dry_run': bool(dry_run),
        'n_demos': len(demos),
        'n_chunks': len(chunks),
        'n_abstracts': len(abstract_records),
        'abstract_coverage': (len(abstract_records) / len(chunks)) if chunks else 0.0,
        'embedding_model': embedder.model_name,
        'embedding_dim': int(embedder.dim),
        'normalized': True,
        'student_model': student_model,
        'teacher_model': teacher_model,
        'abstractor_model': abstractor_model,
        'pool_size_items': len(pool),
        'pool_size_chunks': sum(p['n_steps'] for p in pool),
        'match_version': match_version,
        'selector': sel_info,
    }
    with open(os.path.join(vdir, config.META_JSON_NAME), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    verify_version(version_id, verbose=False)
    logging.info('built %s: %d demos / %d chunks / %d abstracts -> %s',
                 version_id, len(demos), len(chunks), len(abstract_records), vdir)
    print_version(version_id)
    return version_id


# ═════════════════════════════════════════════════════════════════════
#  Verification — the invariant, enforced
# ═════════════════════════════════════════════════════════════════════

def verify_version(version_id: str, verbose: bool = True) -> dict:
    """Assert every alignment property. Raises AssertionError on violation."""
    vdir = config.version_dir(version_id)
    missing = [n for n in config.VERSION_FILES
               if not os.path.exists(os.path.join(vdir, n))]
    if missing:
        raise AssertionError(f'{version_id}: missing files {missing}')

    demos = data.read_jsonl(os.path.join(vdir, config.DEMOS_JSONL_NAME))
    chunks = data.read_jsonl(os.path.join(vdir, config.MEMORY_JSONL_NAME))
    abstracts = data.read_jsonl(os.path.join(vdir, config.ABSTRACT_MEMORY_JSONL_NAME))
    q_emb = np.load(os.path.join(vdir, config.QUESTION_EMBEDDINGS_NPY_NAME))
    c_emb = np.load(os.path.join(vdir, config.EMBEDDINGS_NPY_NAME))
    a_emb = np.load(os.path.join(vdir, config.ABSTRACT_EMBEDDINGS_NPY_NAME))

    # ─── row counts line up with their matrices ──────────────────────
    assert q_emb.shape[0] == len(demos), \
        f'question_embeddings rows {q_emb.shape[0]} != demos {len(demos)}'
    assert c_emb.shape[0] == len(chunks), \
        f'embeddings rows {c_emb.shape[0]} != chunks {len(chunks)}'
    assert a_emb.shape[0] == len(abstracts), \
        f'abstract_embeddings rows {a_emb.shape[0]} != abstracts {len(abstracts)}'
    dims = {q_emb.shape[1], c_emb.shape[1]} | ({a_emb.shape[1]} if len(a_emb) else set())
    assert len(dims) == 1, f'mixed embedding dims across pools: {dims}'

    # ─── ids equal their row index ───────────────────────────────────
    for i, d in enumerate(demos):
        assert d['source_idx'] == i, f'demos[{i}].source_idx == {d["source_idx"]}'
    for j, c in enumerate(chunks):
        assert c['chunk_id'] == j, f'memory[{j}].chunk_id == {c["chunk_id"]}'
    for k, a in enumerate(abstracts):
        assert a['abstract_id'] == k, f'abstract[{k}].abstract_id == {a["abstract_id"]}'

    # ─── the cross-space invariant ───────────────────────────────────
    demo_by_idx = {d['source_idx']: d for d in demos}
    for c in chunks:
        d = demo_by_idx.get(c['source_idx'])
        assert d is not None, f'chunk {c["chunk_id"]} points at missing source_idx'
        assert d['id'] == c['item_id'], (
            f'chunk {c["chunk_id"]} claims item {c["item_id"]} but source_idx '
            f'{c["source_idx"]} is item {d["id"]} — MISALIGNED')
    for a in abstracts:
        c = chunks[a['chunk_id']]
        assert c['item_id'] == a['item_id'], (
            f'abstract {a["abstract_id"]} -> chunk {a["chunk_id"]} item mismatch: '
            f'{a["item_id"]} vs {c["item_id"]} — MISALIGNED')
        assert c['source_idx'] == a['source_idx'], (
            f'abstract {a["abstract_id"]} source_idx {a["source_idx"]} != '
            f'chunk source_idx {c["source_idx"]} — MISALIGNED')

    # ─── every demo actually contributed its chunks ──────────────────
    from collections import Counter
    per_demo = Counter(c['source_idx'] for c in chunks)
    for d in demos:
        assert per_demo[d['source_idx']] == d['n_steps'], (
            f'demo {d["id"]} declares {d["n_steps"]} steps but contributed '
            f'{per_demo[d["source_idx"]]} chunks')

    # ─── embeddings are normalised, as meta claims ───────────────────
    for name, emb in (('question', q_emb), ('chunk', c_emb), ('abstract', a_emb)):
        if len(emb) == 0:
            continue
        norms = np.linalg.norm(emb, axis=1)
        nonzero = norms[norms > 0]
        if len(nonzero):
            assert np.allclose(nonzero, 1.0, atol=1e-3), (
                f'{name} embeddings are not L2-normalised '
                f'(min {nonzero.min():.4f}, max {nonzero.max():.4f})')

    # ─── no train item appears twice ─────────────────────────────────
    ids = [d['id'] for d in demos]
    assert len(ids) == len(set(ids)), 'duplicate train item in demos.jsonl'

    result = {'version_id': version_id, 'n_demos': len(demos),
              'n_chunks': len(chunks), 'n_abstracts': len(abstracts),
              'dim': int(q_emb.shape[1]) if len(q_emb) else 0}
    if verbose:
        print(f'{version_id}: OK — {result["n_demos"]} demos, '
              f'{result["n_chunks"]} chunks, {result["n_abstracts"]} abstracts, '
              f'dim {result["dim"]}; all alignment assertions passed')
    return result


def print_version(version_id: str) -> None:
    vdir = config.version_dir(version_id)
    with open(os.path.join(vdir, config.META_JSON_NAME), 'r', encoding='utf-8') as f:
        meta = json.load(f)
    si = meta['selector']
    print(f'\n{"=" * 68}\nmemory version  {version_id}\n{"=" * 68}')
    print(f'method                {meta["method"]}'
          + ('   [DRY RUN — fake embeddings]' if meta['dry_run'] else ''))
    print(f'demos (train items)   {meta["n_demos"]}')
    print(f'chunks (memory units) {meta["n_chunks"]}')
    print(f'abstract coverage     {meta["n_abstracts"]}/{meta["n_chunks"]} '
          f'({meta["abstract_coverage"]:.1%})')
    print(f'drawn from a pool of  {si["n_candidates_eligible"]} eligible '
          f'/ {meta["pool_size_items"]} with a verified CoT')
    if si.get('target_size') is not None:
        print(f'size target           {si["target_size"]} {si["match_on"]}'
              + ('   *** SHORTFALL — arms are NOT matched ***'
                 if si['shortfall'] else '  (met)'))
    if si.get('zpd_band'):
        print(f'zpd band              {si["zpd_band"]}')
    if si.get('pass_rate_hist'):
        print(f'pass_rate spread      {si["pass_rate_hist"]}')
    print(f'tasks represented     {len(si["per_task"])}')
    print(f'embedder              {meta["embedding_model"]} (dim {meta["embedding_dim"]})')


def list_versions() -> List[str]:
    if not os.path.isdir(config.MEMORY_VERSIONS_DIR):
        return []
    return sorted(
        name for name in os.listdir(config.MEMORY_VERSIONS_DIR)
        if os.path.exists(os.path.join(config.MEMORY_VERSIONS_DIR, name,
                                       config.META_JSON_NAME))
    )


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='Assemble BBEH memory versions (no API calls)')
    sub = p.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='build one memory version (one arm)')
    b.add_argument('--version-id', required=True)
    b.add_argument('--method', default='zpd', choices=selector.METHODS)
    b.add_argument('--student-model', default=config.STUDENT_MODEL)
    b.add_argument('--teacher-model', default=config.TEACHER_MODEL)
    b.add_argument('--abstractor-model', default=config.JUDGE_MODEL)
    b.add_argument('--tasks', nargs='*', default=None)
    b.add_argument('--size', type=int, default=None)
    b.add_argument('--match-version', default=None,
                   help='take the size target from this already-built arm')
    b.add_argument('--match-on', default='chunks', choices=['items', 'chunks'],
                   help='chunks is the honest default: harder items have longer '
                        'CoTs, so equal item counts are not equal memory volume')
    b.add_argument('--seed', type=int, default=config.SPLIT_SEED)
    b.add_argument('--zpd-low', type=float, default=config.ZPD_LOW)
    b.add_argument('--zpd-high', type=float, default=config.ZPD_HIGH)
    b.add_argument('--zpd-strict', action='store_true', help='band becomes 0 < p < 1')
    b.add_argument('--balance-tasks', action='store_true')
    b.add_argument('--dry-run', action='store_true',
                   help='fake embeddings, DRYRUN- prefixed version id')
    b.add_argument('--overwrite', action='store_true')

    v = sub.add_parser('verify', help='re-check a version\'s alignment invariant')
    v.add_argument('version_id', nargs='?', default=None,
                   help='omit to verify every version')

    sub.add_parser('list', help='list built versions')

    s = sub.add_parser('show', help='print a version\'s meta')
    s.add_argument('version_id')

    pr = sub.add_parser('pool', help='difficulty landscape of the verified-CoT pool')
    pr.add_argument('--student-model', default=config.STUDENT_MODEL)
    pr.add_argument('--teacher-model', default=config.TEACHER_MODEL)
    pr.add_argument('--abstractor-model', default=config.JUDGE_MODEL)
    pr.add_argument('--dry-run', action='store_true')

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    if args.cmd == 'build':
        build_version(
            version_id=args.version_id, method=args.method,
            student_model=args.student_model, teacher_model=args.teacher_model,
            abstractor_model=args.abstractor_model, tasks=args.tasks,
            size=args.size, match_version=args.match_version,
            match_on=args.match_on, seed=args.seed,
            zpd_low=args.zpd_low, zpd_high=args.zpd_high,
            zpd_strict=args.zpd_strict, balance_tasks=args.balance_tasks,
            dry_run=args.dry_run, overwrite=args.overwrite)

    elif args.cmd == 'verify':
        targets = [args.version_id] if args.version_id else list_versions()
        if not targets:
            raise SystemExit('no versions built yet')
        bad = 0
        for vid in targets:
            try:
                verify_version(vid)
            except AssertionError as e:
                bad += 1
                print(f'{vid}: FAILED — {e}')
        raise SystemExit(1 if bad else 0)

    elif args.cmd == 'list':
        for vid in list_versions():
            print(vid)

    elif args.cmd == 'show':
        print_version(args.version_id)

    elif args.cmd == 'pool':
        pool, difficulty, _bank, _abs = load_pool(
            args.student_model, args.teacher_model, args.abstractor_model,
            dry_run=args.dry_run)
        selector.print_pool_report(pool, difficulty)


if __name__ == '__main__':
    main()
