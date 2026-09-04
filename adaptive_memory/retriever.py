"""
retriever.py — Two-layer retrieval:
  Layer 1: Full question text → top-N similar demos (question-level)
  Layer 2: Current compact state → top-K chunks within Layer 1's demos (step-level)

Uses numpy cosine similarity (no FAISS).
"""

import json
import logging
import os
import sys
from collections import defaultdict
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from adaptive_memory.config import (
    MEMORY_JSONL, EMBEDDINGS_NPY, QUESTION_EMBEDDINGS_NPY,
    TOP_K, TOP_N_DEMOS, SIMILARITY_THRESHOLD, TOP_N_ABSTRACTS,
)

# Abstract memory paths (optional, loaded if available).
# PuzzleWorld memory bank lives under adaptive_memory/puzzle_memory/versions/.
_BASE = os.path.dirname(os.path.abspath(__file__))
MEMORY_VERSIONS_DIR = os.path.join(_BASE, 'puzzle_memory', 'versions')


def _read_jsonl(path) -> List[dict]:
    """Read a JSONL file robustly against legacy encodings.

    Files should be UTF-8. Some older memory versions were written with the
    Windows locale codec (GBK/cp936), leaving stray bytes like 0xa1 for
    characters such as '×' and '→'. We try UTF-8 first, then GBK (which
    round-trips those files losslessly), and only as a last resort drop
    undecodable bytes so a single bad record can't take down the whole load.
    """
    for encoding in ('utf-8', 'gbk'):
        records = []
        try:
            with open(path, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            if encoding != 'utf-8':
                logging.warning(
                    '%s is not valid UTF-8; decoded as %s. '
                    'Consider re-saving it as UTF-8.', path, encoding
                )
            return records
        except UnicodeDecodeError:
            continue

    logging.warning('Could not decode %s as UTF-8 or GBK; '
                    'dropping undecodable bytes.', path)
    records = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


class MemoryRetriever:
    """Two-layer retrieval over memory bank.

    Layer 1 (question-level): retrieves top-N demos by full question similarity.
    Layer 2 (chunk-level): retrieves top-K chunks within those N demos by
    current compact state similarity.
    """

    def __init__(self,
                 memory_file=MEMORY_JSONL,
                 chunk_embeddings_file=EMBEDDINGS_NPY,
                 question_embeddings_file=QUESTION_EMBEDDINGS_NPY):
        """
        Args:
            memory_file: Path to memory.jsonl (each record has source_idx linking to a demo question).
            chunk_embeddings_file: Path to chunk embeddings.npy (one per record, same order).
            question_embeddings_file: Path to question embeddings.npy (one per unique demo).
        """
        self.records = _read_jsonl(memory_file)

        # Chunk embeddings (one per record)
        self.chunk_embeddings = np.load(chunk_embeddings_file)

        # Handle mismatch due to encoding issues in memory.jsonl
        if self.chunk_embeddings.shape[0] != len(self.records):
            logging.warning(
                f'Mismatch: {self.chunk_embeddings.shape[0]} embeddings but {len(self.records)} records. '
                f'Truncating to match.'
            )
            min_size = min(self.chunk_embeddings.shape[0], len(self.records))
            self.chunk_embeddings = self.chunk_embeddings[:min_size]
            self.records = self.records[:min_size]

        assert self.chunk_embeddings.shape[0] == len(self.records)
        chunk_norms = np.linalg.norm(self.chunk_embeddings, axis=1, keepdims=True)
        chunk_norms[chunk_norms == 0] = 1.0
        self.normalized_chunks = self.chunk_embeddings / chunk_norms

        # Question embeddings (one per unique source_idx)
        self.question_embeddings = np.load(question_embeddings_file)
        q_norms = np.linalg.norm(self.question_embeddings, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1.0
        self.normalized_questions = self.question_embeddings / q_norms

        # Map source_idx → chunk record indices
        self.demo_to_chunks = defaultdict(list)
        for i, rec in enumerate(self.records):
            self.demo_to_chunks[rec['source_idx']].append(i)

        self.n_demos = len(self.question_embeddings)
        logging.info('Loaded %d chunk records from %d demos, '
                    'chunk dim=%d, question dim=%d',
                    len(self.records), self.n_demos,
                    self.chunk_embeddings.shape[1],
                    self.question_embeddings.shape[1])

        # ─── Abstract memory (v8.6) ──────────────────────────────
        self.abstract_patterns = []
        self.abstract_embeddings = None
        self.normalized_abstracts = None

        self._load_abstract_memory()

    def _load_visual_content(self, puzzle_dir: str) -> Optional[dict]:
        """Load visual_content.json for a puzzle if available.

        Args:
            puzzle_dir: Path to puzzle directory (can be in ref_puzzles or hf_full/data)

        Returns:
            Dict with 'summary' key, or None if file not found
        """
        if not puzzle_dir:
            return None

        from pathlib import Path
        puzzle_path = Path(puzzle_dir)
        visual_file = puzzle_path / 'visual_content.json'

        if not visual_file.exists():
            return None

        try:
            with open(visual_file, 'r', encoding='utf-8') as f:
                return json.loads(f.read())
        except Exception as e:
            logging.warning(f'Failed to load visual_content from {visual_file}: {e}')
            return None

    def _query_visual_text(self, puzzle_dir: str) -> str:
        """Solve-time-visible image text for the QUERY puzzle.

        Concatenates the description + key_elements of every content*.png image,
        which is exactly what the solver itself sees. Deliberately EXCLUDES:
          - figure*.png  → that is the answer/solution page (leaks the answer)
          - the top-level 'summary' → it is generated over ALL images (incl.
            figures) and can dribble the final answer, so we don't feed it into
            query-side matching/reranking.
        Returns '' when no usable description exists.
        """
        vc = self._load_visual_content(puzzle_dir)
        if not vc:
            return ''
        parts = []
        for fname, info in (vc.get('images') or {}).items():
            if fname.lower().startswith('figure'):
                continue  # answer key — never expose on the query side
            if not isinstance(info, dict):
                continue
            desc = info.get('description', '')
            if desc:
                parts.append(desc)
            for ke in info.get('key_elements', []) or []:
                parts.append(str(ke))
        return " ".join(parts).strip()

    def _tag_bonus(self, chunk: dict, query_modality, query_skills,
                   modality_w: float, skill_w: float) -> float:
        """Stage-1 soft tag pre-filter: bonus for modality/skill tag overlap.

        Soft (additive) rather than a hard cut, so a cross-type precedent that
        is genuinely useful is not silently dropped; the reranker still gets the
        final say. Returns an additive bonus applied to the blended score.
        """
        bonus = 0.0
        if query_modality:
            qm = set(query_modality)
            cm = set(chunk.get('modality') or [])
            bonus += modality_w * len(qm & cm)
        if query_skills:
            qs = set(query_skills)
            cs = set(chunk.get('skills') or [])
            bonus += skill_w * len(qs & cs)
        return bonus

    def _load_abstract_memory(self):
        """Load abstract patterns and their embeddings if available."""
        # Check v_best first, then v0000
        for version_id in ('v_best', 'v0000'):
            version_dir = os.path.join(MEMORY_VERSIONS_DIR, version_id)
            abstract_file = os.path.join(version_dir, 'abstract_memory.jsonl')
            embedding_file = os.path.join(version_dir, 'abstract_embeddings.npy')

            if os.path.exists(abstract_file):
                self.abstract_patterns.extend(_read_jsonl(abstract_file))

                if os.path.exists(embedding_file):
                    self.abstract_embeddings = np.load(embedding_file)
                    if self.abstract_embeddings.shape[0] > 0:
                        a_norms = np.linalg.norm(
                            self.abstract_embeddings, axis=1, keepdims=True
                        )
                        a_norms[a_norms == 0] = 1.0
                        self.normalized_abstracts = self.abstract_embeddings / a_norms

                if self.abstract_patterns:
                    logging.info('Loaded %d abstract patterns from version %s',
                                len(self.abstract_patterns), version_id)
                break

    def retrieve(self, question_text: str, question_emb: np.ndarray,
                 state_emb: Optional[np.ndarray] = None,
                 top_n_demos: int = TOP_N_DEMOS,
                 top_k_chunks: int = TOP_K,
                 puzzle_dir: Optional[str] = None) -> List[dict]:
        """Two-layer retrieval.

        Args:
            question_text: Full question text (for embedding if state_emb provided).
            question_emb: Pre-computed question embedding (for Layer 1).
            state_emb: Pre-computed current state embedding (for Layer 2).
                       If None, returns top-N demo questions only.
            top_n_demos: Number of demos to retrieve in Layer 1.
            top_k_chunks: Number of chunks to retrieve in Layer 2.
            puzzle_dir: Optional path to puzzle directory containing visual_content.json.
                       If provided and visual_content exists, visual summary is appended
                       to question_text for enhanced embedding (multimodal matching).

        Returns:
            List of top-K chunk record dicts with similarity scores.
        """
        # Optionally enhance question text with visual summary
        if puzzle_dir:
            visual = self._load_visual_content(puzzle_dir)
            if visual and visual.get('summary'):
                question_text = f"{question_text}\n[Visual: {visual['summary']}]"
        # Layer 1: Retrieve top-N demos by question similarity
        q = question_emb / (np.linalg.norm(question_emb) + 1e-8)
        q_sim = self.normalized_questions @ q
        top_demo_indices = np.argsort(q_sim)[::-1][:top_n_demos]

        # Gather chunk indices belonging to these demos
        candidate_chunk_indices = []
        for di in top_demo_indices:
            candidate_chunk_indices.extend(self.demo_to_chunks.get(int(di), []))

        if not candidate_chunk_indices or state_emb is None:
            return []

        # Layer 2: Retrieve top-K chunks by state similarity
        s = state_emb / (np.linalg.norm(state_emb) + 1e-8)
        candidate_chunks = self.normalized_chunks[candidate_chunk_indices]
        state_similarities = candidate_chunks @ s

        # Get top-K within candidates
        local_top = np.argsort(state_similarities)[::-1][:top_k_chunks]

        results = []
        for local_idx in local_top:
            global_idx = candidate_chunk_indices[local_idx]
            record = dict(self.records[global_idx])
            record['similarity'] = float(state_similarities[local_idx])
            results.append(record)

        return results

    def retrieve_with_abstracts(self, question_text: str, question_emb: np.ndarray,
                                state_emb: Optional[np.ndarray] = None,
                                top_n_demos: int = TOP_N_DEMOS,
                                top_k_chunks: int = TOP_K,
                                top_n_abstracts: int = TOP_N_ABSTRACTS,
                                puzzle_dir: Optional[str] = None) -> List[dict]:
        """Three-layer retrieval: abstract pattern → question → state.

        Layer 0: Match state against abstract patterns; from each matched
                 pattern's source_demo, pick the ONE most-similar chunk.
        Layer 1: Retrieve top-N demos by question similarity
        Layer 2: Retrieve top-K chunks by state similarity within those demos

        FIXED (v8.6.1):
        - Layer 0 no longer floods results with ALL chunks of matched demos.
          Only the single best chunk per matched pattern is kept.
        - Final result is STRICTLY trimmed to top_k_chunks (was buggy
          `max(top_k_chunks, len)` which effectively disabled trimming).
        - Abstract and concrete chunks split the top-K budget evenly.

        Args:
            puzzle_dir: Optional path to puzzle directory for visual_content.json

        Returns:
            List of ≤ top_k_chunks chunk record dicts. Chunks matched via
            abstract patterns carry an 'abstract_pattern' field.
        """
        if state_emb is None:
            state_emb = question_emb  # fallback

        # Optionally enhance question text with visual summary
        if puzzle_dir:
            visual = self._load_visual_content(puzzle_dir)
            if visual and visual.get('summary'):
                question_text = f"{question_text}\n[Visual: {visual['summary']}]"

        s = state_emb / (np.linalg.norm(state_emb) + 1e-8)

        # ─── Layer 0: Abstract pattern matching (1 chunk per pattern) ───
        abstract_hits = []  # list of {chunk_idx, pattern, similarity}

        if self.normalized_abstracts is not None and len(self.abstract_patterns) > 0:
            pattern_similarities = self.normalized_abstracts @ s
            top_pattern_indices = np.argsort(pattern_similarities)[::-1][:top_n_abstracts]

            for pi in top_pattern_indices:
                pattern = self.abstract_patterns[int(pi)]
                source_demo = pattern.get('source_demo_idx')
                if source_demo is None:
                    continue

                demo_chunk_indices = self.demo_to_chunks.get(int(source_demo), [])
                if not demo_chunk_indices:
                    continue

                # From this demo's chunks, pick the ONE most similar to current state
                demo_chunks_emb = self.normalized_chunks[demo_chunk_indices]
                chunk_sims = demo_chunks_emb @ s
                best_local = int(np.argmax(chunk_sims))
                best_chunk_idx = demo_chunk_indices[best_local]

                abstract_hits.append({
                    'chunk_idx': best_chunk_idx,
                    'pattern': pattern,
                    'similarity': float(chunk_sims[best_local]),
                })

        # ─── Layer 1 + 2: Standard two-layer concrete retrieval ─────────
        concrete_results = self.retrieve(
            question_text=question_text,
            question_emb=question_emb,
            state_emb=state_emb,
            top_n_demos=top_n_demos,
            top_k_chunks=top_k_chunks,
            puzzle_dir=puzzle_dir,  # Pass through for consistency
        )

        # ─── Merge with strict top-K budget ─────────────────────────────
        # Split the budget: half for abstract, half for concrete.
        # For top_k=2 this gives 1 + 1. If abstract is empty, concrete fills.
        max_abstract = max(1, top_k_chunks // 2)

        seen_ids = set()
        seen_chunk_indices = set()
        merged_results = []

        # 1) Abstract hits (highest chunk-similarity first)
        abstract_hits.sort(key=lambda x: x['similarity'], reverse=True)
        for item in abstract_hits[:max_abstract]:
            chunk_idx = item['chunk_idx']
            if chunk_idx in seen_chunk_indices:
                continue
            record = dict(self.records[chunk_idx])
            record['abstract_pattern'] = item['pattern']
            record['similarity'] = item['similarity']
            record['retrieval_layer'] = 'abstract'
            merged_results.append(record)
            seen_chunk_indices.add(chunk_idx)
            if record.get('id') is not None:
                seen_ids.add(record['id'])

        # 2) Concrete results (dedupe against abstract hits)
        for r in concrete_results:
            if len(merged_results) >= top_k_chunks:
                break
            r_id = r.get('id')
            if r_id is not None and r_id in seen_ids:
                continue
            r['retrieval_layer'] = 'concrete'
            merged_results.append(r)
            if r_id is not None:
                seen_ids.add(r_id)

        # ✅ Strict trim: never exceed top_k_chunks
        return merged_results[:top_k_chunks]

    def _gather_candidate_pool(self, question_emb: np.ndarray,
                               state_emb: np.ndarray, pool_size: int) -> List[dict]:
        """Collect a broad candidate pool (abstract + concrete) for reranking.

        Returns up to `pool_size` chunk-record dicts, each annotated with a raw
        content-similarity 'similarity' and (for abstract hits) 'abstract_pattern'.
        This is Stage-2 recall BEFORE tag weighting / rerank.
        """
        s = state_emb / (np.linalg.norm(state_emb) + 1e-8)
        q = question_emb / (np.linalg.norm(question_emb) + 1e-8)
        pool = []
        seen_chunk_idx = set()

        # Abstract candidates: best chunk per top-N abstract pattern.
        if self.normalized_abstracts is not None and len(self.abstract_patterns) > 0:
            pattern_similarities = self.normalized_abstracts @ s
            top_pattern_indices = np.argsort(pattern_similarities)[::-1][:pool_size]
            for pi in top_pattern_indices:
                pattern = self.abstract_patterns[int(pi)]
                source_demo = pattern.get('source_demo_idx')
                if source_demo is None:
                    continue
                demo_chunk_indices = self.demo_to_chunks.get(int(source_demo), [])
                if not demo_chunk_indices:
                    continue
                demo_chunks_emb = self.normalized_chunks[demo_chunk_indices]
                chunk_sims = demo_chunks_emb @ s
                best_local = int(np.argmax(chunk_sims))
                best_chunk_idx = demo_chunk_indices[best_local]
                if best_chunk_idx in seen_chunk_idx:
                    continue
                rec = dict(self.records[best_chunk_idx])
                rec['abstract_pattern'] = pattern
                rec['similarity'] = float(chunk_sims[best_local])
                rec['retrieval_layer'] = 'abstract'
                pool.append(rec)
                seen_chunk_idx.add(best_chunk_idx)

        # Concrete candidates: top chunks within top demos by state similarity.
        q_sim = self.normalized_questions @ q
        top_demo_indices = np.argsort(q_sim)[::-1][:TOP_N_DEMOS]
        candidate_chunk_indices = []
        for di in top_demo_indices:
            candidate_chunk_indices.extend(self.demo_to_chunks.get(int(di), []))
        if candidate_chunk_indices:
            cand_emb = self.normalized_chunks[candidate_chunk_indices]
            state_sims = cand_emb @ s
            order = np.argsort(state_sims)[::-1]
            for local_idx in order:
                global_idx = candidate_chunk_indices[int(local_idx)]
                if global_idx in seen_chunk_idx:
                    continue
                rec = dict(self.records[global_idx])
                rec['similarity'] = float(state_sims[int(local_idx)])
                rec['retrieval_layer'] = 'concrete'
                pool.append(rec)
                seen_chunk_idx.add(global_idx)
                if len(pool) >= pool_size * 2:
                    break
        return pool

    def retrieve_three_stage(self, question_text: str, question_emb: np.ndarray,
                             state_emb: Optional[np.ndarray] = None,
                             query_modality: Optional[List[str]] = None,
                             query_skills: Optional[List[str]] = None,
                             query_visual_text: str = '',
                             reranker=None,
                             top_k_chunks: int = TOP_K,
                             candidates_m: int = None,
                             tau: float = None,
                             vote_threshold: int = None,
                             modality_w: float = None,
                             skill_w: float = None,
                             title: str = '', flavor: str = '') -> List[dict]:
        """Three-stage retrieval: tag pre-filter → content recall → LLM rerank.

        Stage 1 (soft tag pre-filter): blend content similarity with a bonus for
                 modality/skill tag overlap (additive, not a hard cut).
        Stage 2 (content recall): keep the top-M candidates by blended score.
        Stage 3 (LLM rerank): score each candidate's approach-fit in [0,1];
                 inject only those with fit >= tau. If none clear tau, return []
                 (degrade to no-memory) rather than injecting a misfit precedent.

        Every returned chunk carries a 'fit' field (the reranker's score) and
        'similarity' (raw content cosine) for logging/inspection.
        """
        from adaptive_memory.config import (
            RERANK_CANDIDATES_M, RERANK_FIT_TAU, RERANK_VOTE_THRESHOLD,
            STAGE1_MODALITY_WEIGHT, STAGE1_SKILL_WEIGHT,
        )
        if state_emb is None:
            state_emb = question_emb
        m = candidates_m if candidates_m is not None else RERANK_CANDIDATES_M
        tau = tau if tau is not None else RERANK_FIT_TAU
        vote_threshold = (vote_threshold if vote_threshold is not None
                          else RERANK_VOTE_THRESHOLD)
        modality_w = modality_w if modality_w is not None else STAGE1_MODALITY_WEIGHT
        skill_w = skill_w if skill_w is not None else STAGE1_SKILL_WEIGHT

        # Stage 2 recall (broad pool), then Stage 1 tag-weighted reorder.
        pool = self._gather_candidate_pool(question_emb, state_emb, pool_size=max(m, 6))
        if not pool:
            return []

        for rec in pool:
            rec['tag_bonus'] = self._tag_bonus(
                rec, query_modality, query_skills, modality_w, skill_w
            )
            rec['blended_score'] = rec.get('similarity', 0.0) + rec['tag_bonus']
        pool.sort(key=lambda r: r['blended_score'], reverse=True)
        shortlist = pool[:m]

        # Stage 3: LLM approach-fit rerank + tau gate.
        if reranker is None:
            # No reranker available → fall back to blended-score order, no gate.
            for rec in shortlist:
                rec['fit'] = None
            return shortlist[:top_k_chunks]

        # reranker.score returns, per candidate, an N-sample summary dict:
        #   {'samples': [f1..fN], 'mean': .., 'std': .., 'n': N}
        # Gate by MAJORITY VOTE: a candidate is injected only if at least
        # `vote_threshold` of its N samples clear tau. This is robust to a
        # single sample drifting across the boundary, unlike a bare mean gate.
        summaries = reranker.score(title, flavor, query_visual_text, shortlist)
        for rec, summ in zip(shortlist, summaries):
            samples = summ.get('samples', []) if isinstance(summ, dict) else []
            votes = sum(1 for x in samples if (x or 0.0) >= tau)
            rec['fit'] = summ.get('mean', 0.0) if isinstance(summ, dict) else 0.0
            rec['fit_mean'] = rec['fit']
            rec['fit_std'] = summ.get('std', 0.0) if isinstance(summ, dict) else 0.0
            rec['fit_samples'] = samples
            rec['fit_votes'] = votes
            rec['fit_n'] = summ.get('n', len(samples)) if isinstance(summ, dict) else 0
            rec['fit_tau'] = tau
            rec['fit_passed'] = votes >= vote_threshold

        survivors = [r for r in shortlist if r.get('fit_passed')]
        # Rank survivors by vote count, then mean fit (both descending).
        survivors.sort(key=lambda r: (r.get('fit_votes', 0), r.get('fit_mean', 0.0)),
                       reverse=True)
        return survivors[:top_k_chunks]
