"""
bbeh/retriever.py — three-stage retrieval over a BBEH memory version.

Ported from ``adaptive_memory/retriever.py``. What changed and why:

  * **Text only.** ``_load_visual_content`` / ``_query_visual_text`` /
    ``puzzle_dir`` are gone; BBEH has no images.
  * **Explicit version loading.** The original hardcoded a scan of
    ``('v_best', 'v0000')``, so it could silently load a *different* memory bank
    than the one under test. Here the version id is a required argument, and the
    alignment invariant is asserted at load time.
  * **No silent truncation.** The original trimmed ``records`` and
    ``embeddings`` to their shorter length on mismatch. That converts a
    corrupted bank into a *misaligned* one and hides the bug behind plausible
    scores. Here a mismatch raises and points at ``build_memory verify``.
  * **Stage-1 tags are BBEH-native.** modality/skills become ``task`` and
    ``pattern_type``. Still additive, never a hard filter — a hard within-task
    filter would quietly reduce this to a per-task few-shot retriever, which is
    a much weaker claim than cross-task mechanism transfer.
  * **Abstract hits point at their own chunk.** Our abstractions are per-step
    (1:1 with a chunk), not per-demo, so an abstract hit resolves directly to
    the chunk it abstracts. The two pools are therefore two *views* of the same
    memory: recall by concrete wording, or recall by abstracted mechanism.

A caveat worth stating plainly: BBEH is solved in one pass, so the query is
always a full question, while chunk embeddings are step text. Question-to-step
cosine is a weak signal — which is precisely why Layer 1 (question-to-question)
gates the candidate pool and why Stage 3's judge, not the cosine, decides what
gets injected. Expect raw similarities around 0.2; that is normal and is not the
number to tune against.
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from bbeh import config, data


# A task's mechanism prior keeps at most this many labels, each covering at
# least this share of the task's chunks. Loose on purpose: the prior only needs
# to be right about "this task is mostly sorting" to be useful, and a wrong
# label costs 0.10 of blended score, which Stage 3 can still veto.
_PATTERN_PRIOR_TOP_N = 3
_PATTERN_PRIOR_MIN_SHARE = 0.15


def _l2(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-8)


def _l2_rows(mat: np.ndarray) -> np.ndarray:
    if len(mat) == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class MemoryRetriever:
    """Three-stage retrieval over one memory version directory."""

    def __init__(self, version_id: str, strict: bool = True):
        self.version_id = version_id
        vdir = config.version_dir(version_id)
        if not os.path.isdir(vdir):
            raise FileNotFoundError(
                f'memory version {version_id!r} not found at {vdir}. '
                'Build it: python -m bbeh.build_memory build '
                f'--version-id {version_id} --method <arm>'
            )

        self.records = data.read_jsonl(os.path.join(vdir, config.MEMORY_JSONL_NAME))
        self.demos = data.read_jsonl(os.path.join(vdir, config.DEMOS_JSONL_NAME))
        self.abstracts = data.read_jsonl(
            os.path.join(vdir, config.ABSTRACT_MEMORY_JSONL_NAME))

        self.chunk_emb = _l2_rows(np.load(os.path.join(vdir, config.EMBEDDINGS_NPY_NAME)))
        self.question_emb = _l2_rows(
            np.load(os.path.join(vdir, config.QUESTION_EMBEDDINGS_NPY_NAME)))
        self.abstract_emb = _l2_rows(
            np.load(os.path.join(vdir, config.ABSTRACT_EMBEDDINGS_NPY_NAME)))

        # ─── alignment, asserted rather than papered over ─────────────
        problems = []
        if self.chunk_emb.shape[0] != len(self.records):
            problems.append(f'{self.chunk_emb.shape[0]} chunk embeddings vs '
                            f'{len(self.records)} memory records')
        if self.question_emb.shape[0] != len(self.demos):
            problems.append(f'{self.question_emb.shape[0]} question embeddings vs '
                            f'{len(self.demos)} demos')
        if self.abstract_emb.shape[0] != len(self.abstracts):
            problems.append(f'{self.abstract_emb.shape[0]} abstract embeddings vs '
                            f'{len(self.abstracts)} abstract records')
        if problems and strict:
            raise AssertionError(
                f'memory version {version_id} is misaligned: ' + '; '.join(problems)
                + '\nRun: python -m bbeh.build_memory verify ' + version_id
                + '\n(Do NOT truncate to match — that hides the misalignment and '
                'yields confident nonsense.)'
            )
        elif problems:
            logging.error('version %s misaligned (strict=False): %s',
                          version_id, problems)

        self.demo_to_chunks: Dict[int, List[int]] = defaultdict(list)
        for i, rec in enumerate(self.records):
            self.demo_to_chunks[int(rec['source_idx'])].append(i)

        self.n_demos = len(self.demos)
        self._pattern_prior = self._build_pattern_prior()
        logging.info('loaded memory %s: %d chunks from %d demos, %d abstracts, dim %d',
                     version_id, len(self.records), self.n_demos,
                     len(self.abstracts),
                     self.chunk_emb.shape[1] if len(self.chunk_emb) else 0)

    # ─────────────────────────────────────────────────────────────────
    #  Stage 1: soft tag bonus
    # ─────────────────────────────────────────────────────────────────
    def _build_pattern_prior(self) -> Dict[str, List[str]]:
        """``{task: [dominant pattern_type, ...]}`` learned from this bank.

        A test item has no abstraction — it has no CoT, that is the whole point —
        so there is nothing to compare a candidate's ``pattern_type`` against,
        and the Stage-1 pattern bonus would be dead code. But the *task* label is
        known at query time, and BBEH tasks are mechanistically homogeneous: the
        mechanism distribution of a task's train chunks is a strong, free prior
        for its test items.

        This is not circular and it is not leakage: it reads train-side
        abstractions plus a label we are given. Crucially it is also the only
        thing that makes the bonus interesting — a same-task candidate already
        earns ``task_w``, so what this buys is a *cross-task* chunk carrying the
        right mechanism earning ``pattern_w`` despite the task mismatch. That is
        exactly the transfer the claim is about.

        Derived per version, so a size-matched arm that happens to drop a task
        gets an empty prior there and simply scores 0 — correct, not a bug.
        """
        from collections import Counter
        by_task: Dict[str, Counter] = defaultdict(Counter)
        for rec in self.records:
            ptype = rec.get('pattern_type') or ''
            if ptype:
                by_task[rec['task']][ptype] += 1
        prior: Dict[str, List[str]] = {}
        for task, counts in by_task.items():
            total = sum(counts.values())
            prior[task] = [t for t, c in counts.most_common(_PATTERN_PRIOR_TOP_N)
                           if c / total >= _PATTERN_PRIOR_MIN_SHARE]
        return prior

    def pattern_prior_for_task(self, task: Optional[str]) -> List[str]:
        return list(self._pattern_prior.get(task or '', ()))

    def _tag_bonus(self, chunk: dict, query_task: Optional[str],
                   query_pattern_types: Optional[Sequence[str]],
                   task_w: float, pattern_w: float) -> float:
        """Additive bonus for BBEH-native tag agreement.

        Soft on purpose. ``task`` is a coarse but highly informative tag; making
        it a hard filter would turn the whole system into per-task few-shot
        retrieval and forfeit the cross-task transfer result.
        """
        bonus = 0.0
        if query_task and chunk.get('task') == query_task:
            bonus += task_w
        if query_pattern_types:
            ctype = chunk.get('pattern_type') or ''
            if ctype and ctype in set(query_pattern_types):
                bonus += pattern_w
        return bonus

    # ─────────────────────────────────────────────────────────────────
    #  Stage 2: content recall over both pools
    # ─────────────────────────────────────────────────────────────────
    def _gather_candidate_pool(self, query_emb: np.ndarray, pool_size: int,
                               top_n_demos: int = config.TOP_N_DEMOS) -> List[dict]:
        """Broad candidate pool from the abstract and concrete pools.

        Concrete side: Layer 1 picks the top-N most similar demo *questions*,
        then chunks within those demos are ranked against the query. Abstract
        side: the top patterns by mechanism similarity, each resolving to the
        chunk it abstracts. Deduplicated by chunk index, abstract view first.
        """
        q = _l2(query_emb)
        pool: List[dict] = []
        seen = set()

        # ─── abstract pool ───────────────────────────────────────────
        if len(self.abstract_emb):
            sims = self.abstract_emb @ q
            for ai in np.argsort(sims)[::-1][:pool_size]:
                arec = self.abstracts[int(ai)]
                cidx = int(arec['chunk_id'])
                if cidx in seen or cidx >= len(self.records):
                    continue
                rec = dict(self.records[cidx])
                rec['similarity'] = float(sims[int(ai)])
                rec['retrieval_layer'] = 'abstract'
                rec['matched_abstract_id'] = int(arec['abstract_id'])
                pool.append(rec)
                seen.add(cidx)

        # ─── concrete pool ───────────────────────────────────────────
        if len(self.question_emb):
            q_sim = self.question_emb @ q
            cand: List[int] = []
            for di in np.argsort(q_sim)[::-1][:top_n_demos]:
                cand.extend(self.demo_to_chunks.get(int(di), []))
            if cand:
                sims = self.chunk_emb[cand] @ q
                for local in np.argsort(sims)[::-1]:
                    gidx = cand[int(local)]
                    if gidx in seen:
                        continue
                    rec = dict(self.records[gidx])
                    rec['similarity'] = float(sims[int(local)])
                    rec['retrieval_layer'] = 'concrete'
                    rec['demo_question_similarity'] = float(
                        q_sim[int(rec['source_idx'])])
                    pool.append(rec)
                    seen.add(gidx)
                    if len(pool) >= pool_size * 2:
                        break
        return pool

    # ─────────────────────────────────────────────────────────────────
    #  The full three-stage pipeline
    # ─────────────────────────────────────────────────────────────────
    def retrieve_three_stage(self, query_text: str, query_emb: np.ndarray,
                             query_task: Optional[str] = None,
                             query_pattern_types: Optional[Sequence[str]] = None,
                             reranker=None,
                             top_k_chunks: int = config.TOP_K,
                             candidates_m: int = config.RERANK_CANDIDATES_M,
                             tau: float = config.RERANK_FIT_TAU,
                             vote_threshold: int = config.RERANK_VOTE_THRESHOLD,
                             task_w: float = config.STAGE1_TASK_WEIGHT,
                             pattern_w: float = config.STAGE1_PATTERN_WEIGHT,
                             top_n_demos: int = config.TOP_N_DEMOS,
                             random_retrieval: bool = False) -> List[dict]:
        """Stage 2 recall -> Stage 1 tag reweighting -> Stage 3 judge + gate.

        Returns 0..``top_k_chunks`` chunk dicts. **Returning an empty list is a
        legitimate, designed outcome**: if no candidate clears the fit gate, the
        solver is better served by no precedent than by a misfitting one
        ("宁空勿滥"). ``run.py`` must therefore treat an empty injection as a
        normal case, and the analysis must report the gated fraction — a memory
        arm that gates 95% of items is not really being tested.

        Every returned chunk carries ``similarity`` (raw cosine),
        ``tag_bonus``, ``blended_score`` and the ``fit_*`` family, so the
        injection log can be audited without rerunning anything.
        """
        if random_retrieval:
            import random
            rng = random.Random(f'random_retrieval|{query_text}')
            indices = rng.sample(range(len(self.records)), min(max(candidates_m, 6), len(self.records)))
            pool = []
            for idx in indices:
                rec = dict(self.records[idx])
                rec['similarity'] = 0.5
                rec['retrieval_layer'] = 'random'
                pool.append(rec)
        else:
            pool = self._gather_candidate_pool(
                query_emb, pool_size=max(candidates_m, 6), top_n_demos=top_n_demos)
        if not pool:
            return []

        # ─── Stage 1: reweight, then shortlist ───────────────────────
        if query_pattern_types is None:
            query_pattern_types = self.pattern_prior_for_task(query_task)
        for rec in pool:
            rec['tag_bonus'] = self._tag_bonus(
                rec, query_task, query_pattern_types, task_w, pattern_w)
            rec['blended_score'] = rec.get('similarity', 0.0) + rec['tag_bonus']
        pool.sort(key=lambda r: r['blended_score'], reverse=True)
        shortlist = pool[:candidates_m]

        # ─── Stage 3: LLM approach-fit rerank with a majority-vote gate ──
        if reranker is None:
            # No judge: fall back to blended order with no gate. This is the
            # "Stage 3 ablated" condition, not the main arm.
            for rec in shortlist:
                rec['fit'] = None
                rec['fit_passed'] = True
                rec['gate'] = 'no_reranker'
            return shortlist[:top_k_chunks]

        summaries = reranker.score(query_text, query_task or '', shortlist)
        for rec, summ in zip(shortlist, summaries):
            summ = summ if isinstance(summ, dict) else {}
            samples = summ.get('samples', []) or []
            votes = sum(1 for x in samples if (x or 0.0) >= tau)
            rec['fit'] = summ.get('mean', 0.0)
            rec['fit_mean'] = rec['fit']
            rec['fit_std'] = summ.get('std', 0.0)
            rec['fit_samples'] = samples
            rec['fit_votes'] = votes
            rec['fit_n'] = summ.get('n', len(samples))
            rec['fit_tau'] = tau
            # Judge calls that failed outright are dropped rather than scored 0,
            # so a degraded candidate has too few samples to clear the vote and
            # is gated out. Recorded so the analysis can separate "the judge said
            # no" from "the judge never answered".
            rec['fit_degraded'] = bool(summ.get('degraded'))
            # Majority vote rather than a bare mean: with N=5 samples a single
            # sample drifting across the boundary should not flip the decision.
            rec['fit_passed'] = votes >= vote_threshold
            rec['gate'] = 'vote'

        survivors = [r for r in shortlist if r.get('fit_passed')]
        survivors.sort(key=lambda r: (r.get('fit_votes', 0), r.get('fit_mean', 0.0)),
                       reverse=True)
        return survivors[:top_k_chunks]

    # ─────────────────────────────────────────────────────────────────
    #  Diagnostics
    # ─────────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        from collections import Counter
        tasks = Counter(r['task'] for r in self.records)
        with_prior = sum(1 for t in tasks if self._pattern_prior.get(t))
        return {
            'version_id': self.version_id,
            'n_chunks': len(self.records),
            'n_demos': self.n_demos,
            'n_abstracts': len(self.abstracts),
            'tasks': len(tasks),
            'tasks_with_pattern_prior': with_prior,
            'pattern_prior': {t: v for t, v in self._pattern_prior.items() if v},
            'pattern_types': dict(
                Counter(r.get('pattern_type', '') for r in self.records).most_common()),
        }

    def warn_if_degenerate(self) -> List[str]:
        """Conditions that make retrieval quietly useless. Check before spending.

        Each of these still produces plausible-looking output — a run completes,
        numbers appear, and nothing errors — which is exactly why they need an
        explicit check rather than being left to show up as a disappointing
        result that gets attributed to the method.
        """
        warns = []
        tasks = {r['task'] for r in self.records}
        with_prior = sum(1 for t in tasks if self._pattern_prior.get(t))
        if tasks and with_prior / len(tasks) < 0.5:
            warns.append(
                f'only {with_prior}/{len(tasks)} tasks have a mechanism prior, so '
                f'STAGE1_PATTERN_WEIGHT ({config.STAGE1_PATTERN_WEIGHT}) is mostly '
                'inert — the abstractor likely produced diffuse or collapsed labels')
        if self.abstracts and len(self.abstracts) < 0.5 * len(self.records):
            warns.append(
                f'only {len(self.abstracts)}/{len(self.records)} chunks have an '
                'abstraction; the abstract pool is half-empty and Stage 2 recall '
                'is effectively concrete-only')
        if not self.abstracts:
            warns.append('no abstractions at all — this is a one-pool retriever')
        return warns


# ═════════════════════════════════════════════════════════════════════
#  Query embeddings, cached across arms
# ═════════════════════════════════════════════════════════════════════

class QueryEmbedder:
    """Embeds test questions, with a disk cache shared by every arm.

    The same 2260 test questions are embedded identically for all seven arms, so
    encoding them once and reusing the matrix saves both time and any risk of an
    arm being evaluated against subtly different query vectors.
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL,
                 dry_run: bool = False):
        self.dry_run = dry_run
        self.model_name = f'HASH-FAKE-{config.EMBEDDING_DIM}d' if dry_run else model_name
        self._embedder = None
        self.cache_path = os.path.join(
            config.WORK_DIR,
            f'query_emb_{config._slug("DRYRUN" if dry_run else model_name)}.npz')
        self._by_id: Dict[str, np.ndarray] = {}
        if os.path.exists(self.cache_path):
            with np.load(self.cache_path, allow_pickle=False) as npz:
                ids = list(npz['ids'])
                mat = npz['emb']
            self._by_id = {str(i): mat[k] for k, i in enumerate(ids)}
            logging.info('query embedding cache: %d vectors from %s',
                         len(self._by_id), self.cache_path)

    def _ensure_model(self):
        if self._embedder is None:
            from bbeh.build_memory import Embedder, HashEmbedder
            self._embedder = HashEmbedder() if self.dry_run else Embedder(self.model_name)
        return self._embedder

    def embed_items(self, items: Sequence[dict]) -> Dict[str, np.ndarray]:
        """Return ``{item_id: vector}``, encoding and caching whatever is missing."""
        missing = [it for it in items if it['id'] not in self._by_id]
        if missing:
            emb = self._ensure_model().encode([data.embed_text(it) for it in missing])
            for it, vec in zip(missing, emb):
                self._by_id[it['id']] = vec
            self._save()
        return {it['id']: self._by_id[it['id']] for it in items}

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_path) or '.', exist_ok=True)
        ids = sorted(self._by_id)
        mat = np.stack([self._by_id[i] for i in ids]).astype(np.float32)
        np.savez(self.cache_path, ids=np.array(ids), emb=mat)


def main():
    import argparse
    from bbeh.build_memory import list_versions
    p = argparse.ArgumentParser(description='Inspect a memory version / smoke-test retrieval')
    p.add_argument('version_id', nargs='?', default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    for vid in ([args.version_id] if args.version_id else list_versions()):
        r = MemoryRetriever(vid)
        s = r.stats()
        print(f"{vid:26s} chunks={s['n_chunks']:5d} demos={s['n_demos']:4d} "
              f"abstracts={s['n_abstracts']:5d} tasks={s['tasks']} "
              f"prior={s['tasks_with_pattern_prior']}/{s['tasks']}")
        for w in r.warn_if_degenerate():
            print(f'    WARNING: {w}')


if __name__ == '__main__':
    main()
