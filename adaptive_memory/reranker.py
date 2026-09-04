"""
reranker.py — Stage 3 of three-stage retrieval: LLM "approach-fit" reranker.

Stages 1 (tag pre-filter) and 2 (content similarity) produce a shortlist of
candidate memory chunks that are *lexically/semantically* near the query
puzzle. But we established empirically that title/flavor similarity does NOT
predict whether a precedent's *solving approach* actually helps: a misfit
precedent with decent cosine similarity can actively derail the solver
(e.g. Arithmetic Island 0.43 -> 0.0 once an unrelated path-tracing precedent
was injected).

This module asks a cheap judge model (haiku) the question the embeddings
cannot answer: "Given this puzzle, how applicable is this precedent's solving
approach?" It returns fit scores in [0, 1] per candidate.

Stochastic gating (reproducible)
--------------------------------
A single judge call is not reproducible: at nonzero temperature the fit score
drifts run-to-run, and even at temperature 0 the third-party proxy is not
bit-stable, so a candidate whose fit sits near the gate flips inject/skip
between runs. We therefore:

  1. Sample the judge N times at a nonzero temperature (config.RERANK_SAMPLES_N,
     config.RERANK_TEMPERATURE) and return all N scores per candidate. The
     caller gates by MAJORITY VOTE (>= RERANK_VOTE_THRESHOLD of N samples clear
     tau), which is robust to a single sample drifting across the boundary.
  2. Cache the N sampled scores on disk keyed by (prompt-template, query,
     candidate) — NOT by tau or the vote threshold. A rerun reads back the
     identical samples and reaches the identical gate decision, so the whole
     pipeline is reproducible. Set ADAPTIVE_RERANK_FRESH=1 (or delete the cache
     file) to force fresh sampling.

IMPORTANT (no answer leakage): the query-side context handed to the judge must
contain ONLY solve-time-visible information — title, flavor, and content*.png
descriptions (never figure*.png, which is the answer key). The memory side may
carry its source puzzle's full solution, because that is what a memory is
*supposed* to provide.
"""

import hashlib
import json
import logging
import os
import re
from typing import List, Optional


_RERANK_PROMPT = """You are ranking reasoning precedents retrieved from a memory bank for their usefulness in solving a NEW puzzle.

You will see the new puzzle (its title, flavor text, and a description of what its images contain) and a numbered list of candidate precedents. Each precedent is a (state -> action -> result) reasoning step drawn from a DIFFERENT, already-solved puzzle. Some candidates come with BOTH an abstract pattern AND a concrete example — the concrete example is the ground truth of what mechanism the precedent actually demonstrates.

For EACH candidate, judge how well its SOLVING APPROACH (the kind of reasoning move it demonstrates) would transfer to the new puzzle. Ignore surface-level word overlap; judge the method, not the topic.

Scoring rubric (0.0 to 1.0):
- 1.0  the approach is exactly the move this puzzle needs
- 0.7  the approach is the right family and likely helps
- 0.4  loosely related; might occasionally help
- 0.2  weak/unclear connection
- 0.0  irrelevant or would mislead the solver

CRITICAL — mechanism mismatch (score <= 0.3 when any of these apply):
- ABSTRACT SHELL vs CONCRETE MECHANISM MISMATCH. A precedent's abstract may sound relevant (e.g. "collect letters and concatenate to form the answer") while its concrete example demonstrates an incompatible mechanism (e.g. recombining top/bottom halves of words, or filling in one missing letter per row). When the abstract and the concrete disagree, TRUST THE CONCRETE — score by what the concrete example actually does, not by the abstract's generic phrasing.
- WRONG SPATIAL READING DIRECTION. If the query puzzle needs "read down column N" / "read along the diagonal" / "read every other row", a precedent that just says "extract / collect / gather letters" without specifying that same direction is NOT the right mechanism — score <= 0.3.
- WRONG CARDINALITY. If the query needs to fill 5 letters per row but the precedent identifies 1 missing letter per row (or vice versa), the fill rule is different — score <= 0.3 even if both are "letter-extraction" flavored.
- WRONG OPERATION FAMILY. Half-word-recombining, missing-letter-completion, column-read acrostic, indexed extraction, and cipher decoding are DIFFERENT families; they collide only through the generic "letters in / letters out" shell. Do not conflate them.

Prefer the empty set over a misfit: injecting a precedent whose mechanism doesn't match actively derails the solver. When in doubt between "loosely related" (0.4) and "would mislead" (0.0), pick 0.0.

NEW PUZZLE
Title: {title}
Flavor: {flavor}
Image content: {visual}

CANDIDATE PRECEDENTS
{candidates}

Respond with ONLY a JSON array of objects, one per candidate, in the same order:
[{{"idx": 1, "fit": 0.0}}, {{"idx": 2, "fit": 0.0}}, ...]
No prose, no explanation."""


# Hash of the prompt template — part of the cache key so that editing the rubric
# transparently invalidates every cached score (they were judged by old rules).
_PROMPT_HASH = hashlib.sha256(_RERANK_PROMPT.encode('utf-8')).hexdigest()[:16]


def _format_candidate(i: int, chunk: dict) -> str:
    """Render one candidate precedent for the reranker prompt (with index)."""
    if chunk.get('abstract_pattern'):
        pat = chunk['abstract_pattern']
        approach = (
            f"{pat.get('abstract_state', '')} -> "
            f"{pat.get('abstract_action', '')} -> "
            f"{pat.get('abstract_next_state', '')}"
        )
        ptype = pat.get('pattern_type', 'general')
    else:
        approach = (
            f"{chunk.get('state', '')} -> "
            f"{chunk.get('action', '')} -> "
            f"{chunk.get('next_state', '')}"
        )
        ptype = chunk.get('pattern_type', 'general')
    src = chunk.get('source_puzzle', '?')
    return f"[{i}] (type={ptype}, from='{src}') {approach}"


def _candidate_signature(chunk: dict) -> str:
    """Order-independent identity of a candidate, for the cache key.

    Deliberately excludes the prompt index [i]: the same precedent must hash to
    the same key regardless of where it lands in the shortlist, so reruns whose
    shortlist ordering shifts still hit the cache.
    """
    if chunk.get('abstract_pattern'):
        pat = chunk['abstract_pattern']
        return (
            "A|" + str(pat.get('pattern_type', '')) + "|"
            + str(pat.get('abstract_state', '')) + "->"
            + str(pat.get('abstract_action', '')) + "->"
            + str(pat.get('abstract_next_state', '')) + "|"
            + str(chunk.get('source_puzzle', ''))
        )
    return (
        "C|" + str(chunk.get('pattern_type', '')) + "|"
        + str(chunk.get('state', '')) + "->"
        + str(chunk.get('action', '')) + "->"
        + str(chunk.get('next_state', '')) + "|"
        + str(chunk.get('source_puzzle', ''))
    )


def _parse_fit_scores(raw: str, n: int) -> List[float]:
    """Parse the judge's JSON array into n fit scores, robust to noise.

    Falls back to 0.0 for any candidate whose score can't be recovered, so a
    malformed reply degrades gracefully (candidate simply won't clear tau)
    rather than crashing the run.
    """
    scores = [0.0] * n
    if not raw:
        return scores

    # Try strict JSON first, then salvage the first [...] block.
    payload = None
    try:
        payload = json.loads(raw)
    except Exception:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try:
                payload = json.loads(m.group(0))
            except Exception:
                payload = None

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            idx = item.get('idx')
            fit = item.get('fit')
            if isinstance(idx, (int, float)) and 1 <= int(idx) <= n:
                try:
                    scores[int(idx) - 1] = max(0.0, min(1.0, float(fit)))
                except (TypeError, ValueError):
                    continue
        return scores

    # Last-ditch: pull bare floats in order.
    floats = re.findall(r'[01](?:\.\d+)?', raw)
    for i, f in enumerate(floats[:n]):
        try:
            scores[i] = max(0.0, min(1.0, float(f)))
        except ValueError:
            pass
    return scores


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    """Population standard deviation over the samples (0.0 for <2 samples)."""
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


class ApproachFitReranker:
    """Stage-3 LLM reranker scoring approach-fit of candidates to a query.

    score() returns, per candidate, a dict {'samples','mean','std','n'} holding
    N stochastic judge scores. The caller applies the tau + majority-vote gate.
    """

    def __init__(self, gen_client=None, cache_path: Optional[str] = None,
                 n_samples: Optional[int] = None,
                 temperature: Optional[float] = None):
        """
        Args:
            gen_client: an adaptive_memory.api_client.GenClient (or anything
                with .generate(prompt, max_tokens, temperature) -> str).
                If None, a client is built lazily from config.API.
            cache_path: JSONL file to persist sampled scores for reproducibility.
                If None, no caching (every call re-samples).
            n_samples: samples per candidate (default config.RERANK_SAMPLES_N).
            temperature: sampling temperature (default config.RERANK_TEMPERATURE).
        """
        self._client = gen_client
        self.cache_path = cache_path
        from adaptive_memory.config import RERANK_SAMPLES_N, RERANK_TEMPERATURE
        self.n_samples = int(n_samples if n_samples is not None else RERANK_SAMPLES_N)
        self.temperature = float(
            temperature if temperature is not None else RERANK_TEMPERATURE
        )
        self._fresh = os.environ.get('ADAPTIVE_RERANK_FRESH', '') \
            not in ('', '0', 'false', 'False')
        self._cache = {}
        if self.cache_path and not self._fresh:
            self._load_cache()

    # ─── cache ───────────────────────────────────────────────────────
    def _load_cache(self):
        try:
            if not os.path.exists(self.cache_path):
                return
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    k = rec.get('key')
                    s = rec.get('samples')
                    if k and isinstance(s, list):
                        self._cache[k] = [float(x) for x in s]  # last wins
            logging.info('Rerank cache: loaded %d entries from %s',
                         len(self._cache), self.cache_path)
        except Exception as e:
            logging.warning('Rerank cache load failed (%s); starting empty', e)

    def _append_cache(self, key: str, samples: List[float], meta: dict):
        if not self.cache_path:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            rec = {'key': key, 'samples': samples}
            if meta:
                rec.update(meta)
            with open(self.cache_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logging.warning('Rerank cache append failed (%s)', e)

    def _cache_key(self, query_sig: str, cand_sig: str) -> str:
        h = hashlib.sha256()
        h.update(_PROMPT_HASH.encode('utf-8'))
        h.update(b'\x1f')
        h.update(query_sig.encode('utf-8'))
        h.update(b'\x1f')
        h.update(cand_sig.encode('utf-8'))
        return h.hexdigest()

    # ─── client ──────────────────────────────────────────────────────
    def _get_client(self):
        if self._client is None:
            from adaptive_memory.api_client import GenClient
            from adaptive_memory.config import API
            self._client = GenClient(
                base_url=API['base_url'],
                api_key=API['api_key'],
                model=API['gen_model'],
                protocol=API.get('api_protocol', 'chat'),
                temperature=self.temperature,
                max_tokens=256,
            )
        return self._client

    def _one_sample(self, prompt: str, n_cand: int) -> List[float]:
        """One judge call → fit vector over candidates (all 0.0 on failure)."""
        try:
            raw = self._get_client().generate(
                prompt, max_tokens=256, temperature=self.temperature
            )
        except Exception as e:
            logging.warning('Reranker call failed (%s); scoring all 0.0', e)
            return [0.0] * n_cand
        return _parse_fit_scores(raw, n_cand)

    # ─── scoring ─────────────────────────────────────────────────────
    def score(self, title: str, flavor: str, visual: str,
              candidates: List[dict]) -> List[dict]:
        """Return, per candidate, {'samples','mean','std','n'} of N judge scores.

        Cached candidates return their stored samples untouched (reproducible);
        only cache-missing candidates trigger fresh judge calls. On total
        failure a candidate gets all-0.0 samples (→ 0 votes → no-memory).
        """
        if not candidates:
            return []

        n_cand = len(candidates)
        query_sig = ((title or '').strip() + "\x1f"
                     + (flavor or '').strip() + "\x1f"
                     + (visual or '').strip())
        keys = [self._cache_key(query_sig, _candidate_signature(c))
                for c in candidates]

        # Frozen cache hits stay frozen; only recompute genuine misses.
        cached = [None if self._fresh else self._cache.get(k) for k in keys]
        need = [i for i, s in enumerate(cached)
                if (not s) or len(s) < self.n_samples]

        if need:
            cand_text = "\n".join(
                _format_candidate(i + 1, c) for i, c in enumerate(candidates)
            )
            prompt = _RERANK_PROMPT.format(
                title=(title or '').strip(),
                flavor=(flavor or '').strip(),
                visual=(visual or '(no image description available)').strip(),
                candidates=cand_text,
            )
            # N batched samples: each call scores all candidates at once.
            matrix = [self._one_sample(prompt, n_cand)
                      for _ in range(self.n_samples)]
            for i in need:
                samples_i = [matrix[s][i] for s in range(self.n_samples)]
                self._cache[keys[i]] = samples_i
                cached[i] = samples_i
                self._append_cache(keys[i], samples_i, {
                    'title': (title or '').strip()[:120],
                    'candidate': _candidate_signature(candidates[i])[:200],
                })

        results = []
        for s in cached:
            s = s or [0.0] * self.n_samples
            results.append({
                'samples': s,
                'mean': _mean(s),
                'std': _std(s),
                'n': len(s),
            })
        return results
