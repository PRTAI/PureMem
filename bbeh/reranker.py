"""
bbeh/reranker.py — Stage 3: LLM approach-fit reranker for BBEH text reasoning.

Stages 2 and 1 give us candidates that are semantically near the query. They
cannot answer the question that actually matters: *would this precedent's
reasoning move help here?* Cosine similarity rewards topical overlap, and a
topically-similar precedent demonstrating the wrong mechanism is worse than no
precedent at all — it gives the solver a confident wrong plan. So a cheap judge
scores approach-fit in [0,1], N times, and the caller gates by majority vote.

Two fixes relative to the PuzzleWorld original:

  * **The concrete step is now actually shown.** The old rubric instructed the
    judge to "trust the concrete when abstract and concrete disagree", but
    ``_format_candidate`` rendered the abstract *instead of* the concrete
    whenever an abstraction existed. The rule was unenforceable — the judge
    never saw the thing it was told to trust. Both are rendered here.
  * **Infra failures are no longer cached as 0.0.** A timeout used to be stored
    as a genuine "irrelevant" verdict, permanently and invisibly, so a network
    blip would silently disable memory for that item in every later run. Failed
    samples are not written to the cache, and a candidate short of its full
    sample count is marked ``degraded`` — it cannot reach the vote threshold, so
    the failure degrades toward no-memory (safe) and the next run retries it.

Reproducibility: the N sampled scores are cached on disk keyed by
(prompt-template hash, query excerpt, candidate identity) — deliberately NOT by
tau or the vote threshold, so those can be re-swept over a fixed set of samples
without re-spending. Editing the rubric changes ``_PROMPT_HASH`` and invalidates
every entry, which is the intended behaviour: old scores were judged by old
rules. ``BBEH_RERANK_FRESH=1`` forces resampling.
"""

import hashlib
import json
import logging
import os
from typing import List, Optional, Sequence

from bbeh import config, jsonutil

# The judge needs to see what the question asks, but BBEH inputs run to 32k
# chars and paying that 5x per candidate set would make Stage 3 cost more than
# the solve pass itself. Head+tail: the head carries the task framing, the tail
# carries the actual instruction, which in BBEH is nearly always last.
QUERY_HEAD_CHARS = 1200
QUERY_TAIL_CHARS = 1800


_RERANK_PROMPT = """You are ranking reasoning precedents retrieved from a memory bank for their usefulness in solving a NEW reasoning problem.

You will see the new problem and a numbered list of candidate precedents. Each precedent is one (state -> action -> result) reasoning step taken from a DIFFERENT problem that was already solved correctly. Each candidate shows its CONCRETE step and, where available, an ABSTRACT description of the move.

For EACH candidate, judge how well its REASONING MOVE would transfer to the new problem. Judge the method, not the topic: a precedent about trains and a problem about pipes can share the identical mechanism, while two problems both about calendars can need completely different moves.

Scoring rubric (0.0 to 1.0):
- 1.0  this is exactly the move the new problem needs
- 0.7  right family of move, likely helps
- 0.4  loosely related, might occasionally help
- 0.2  weak or unclear connection
- 0.0  irrelevant, or would actively mislead the solver

Score <= 0.3 whenever any of these apply:
- ABSTRACT/CONCRETE MISMATCH. The abstract may sound relevant ("combine the values to get the result") while the concrete step does something incompatible. TRUST THE CONCRETE. Score what the concrete step actually does, never the abstract's generic phrasing.
- WRONG OPERATION FAMILY. Summing a list, sorting a list, counting occurrences, propagating a constraint, tracking a changing state, and eliminating candidates are DIFFERENT families. They collide only through generic "values in / value out" wording. Do not conflate them.
- WRONG DIRECTION OR ORDER. If the new problem requires processing in a specific order (chronological, reverse, nested innermost-first) and the precedent does not demonstrate that same ordering, it is not the right move.
- WRONG GRANULARITY. A precedent that handles one element when the problem needs a rule applied across all of them (or vice versa) is a different move.
- BOOKKEEPING ONLY. A step that merely restates the given data, sets up notation, or announces an intention teaches nothing transferable. Score 0.0.

Prefer the empty set over a misfit. Injecting a precedent whose mechanism does not match measurably derails the solver, whereas injecting nothing simply leaves it as capable as it was. When torn between "loosely related" (0.4) and "would mislead" (0.0), choose 0.0.

NEW PROBLEM
Task family: {task}
Problem:
{query}

CANDIDATE PRECEDENTS
{candidates}

Respond with ONLY a JSON array of objects, one per candidate, in the same order:
[{{"idx": 1, "fit": 0.0}}, {{"idx": 2, "fit": 0.0}}]
No prose, no explanation."""


# Part of the cache key: editing the rubric invalidates every cached score,
# because those were judged under different instructions.
_PROMPT_HASH = hashlib.sha256(_RERANK_PROMPT.encode('utf-8')).hexdigest()[:16]


def query_excerpt(text: str) -> str:
    text = (text or '').strip()
    if len(text) <= QUERY_HEAD_CHARS + QUERY_TAIL_CHARS:
        return text
    return (text[:QUERY_HEAD_CHARS].strip()
            + '\n[... middle omitted ...]\n'
            + text[-QUERY_TAIL_CHARS:].strip())


def format_candidate(i: int, chunk: dict) -> str:
    """Render one candidate: concrete step always, abstract as annotation."""
    concrete = (f"{chunk.get('state', '')} -> {chunk.get('action', '')} "
                f"-> {chunk.get('next_state', '')}")
    pat = chunk.get('abstract_pattern') or {}
    ptype = (pat.get('pattern_type') or chunk.get('pattern_type') or 'general')
    lines = [f"[{i}] (type={ptype}, from task '{chunk.get('task', '?')}')",
             f"    CONCRETE: {concrete}"]
    if pat.get('abstract_action'):
        lines.append(
            f"    ABSTRACT: {pat.get('abstract_state', '')} -> "
            f"{pat.get('abstract_action', '')} -> "
            f"{pat.get('abstract_next_state', '')}")
    return '\n'.join(lines)


def candidate_signature(chunk: dict) -> str:
    """Order-independent identity of a candidate, for the cache key.

    Excludes the shortlist position on purpose: the same precedent must hash to
    the same key wherever it lands, so a rerun whose ordering shifts still hits
    the cache. Includes the abstract, because the abstract is part of what the
    judge saw.
    """
    pat = chunk.get('abstract_pattern') or {}
    return '\x1f'.join(str(x) for x in (
        chunk.get('task', ''),
        chunk.get('state', ''), chunk.get('action', ''), chunk.get('next_state', ''),
        pat.get('pattern_type', ''), pat.get('abstract_state', ''),
        pat.get('abstract_action', ''), pat.get('abstract_next_state', ''),
    ))


def parse_fit_scores(raw: str, n: int) -> List[Optional[float]]:
    """Judge reply -> n scores. ``None`` where a score could not be recovered.

    ``None`` rather than 0.0 matters: an unparseable reply is not evidence that
    a precedent is irrelevant, and conflating the two would let parse failures
    masquerade as confident vetoes — the same class of bug that produced phantom
    0.0 scores in the PuzzleWorld eval.
    """
    scores: List[Optional[float]] = [None] * n
    if not raw:
        return scores

    payload = jsonutil.extract_json(raw, 'array')
    if payload is None:
        payload = jsonutil.extract_json_objects(raw) or None

    def clamp(v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return None

    if isinstance(payload, list) and payload:
        by_idx = 0
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            idx = entry.get('idx')
            if idx is None:
                # No label at all: trust position, which is what the prompt asks
                # for anyway.
                if by_idx >= n:
                    continue
                slot = by_idx
            elif (isinstance(idx, (int, float)) and not isinstance(idx, bool)
                    and 1 <= int(idx) <= n):
                slot = int(idx) - 1
            else:
                # A label that exists but is out of range means the judge lost
                # track of the shortlist. Drop the entry — silently reinterpreting
                # it positionally would attach one candidate's verdict to another.
                continue
            by_idx = slot + 1
            val = clamp(entry.get('fit'))
            if val is not None:
                scores[slot] = val
        # A well-formed array is authoritative even when every ``fit`` inside it
        # was unusable. Falling through to the regex here would scrape the *idx*
        # numbers out of the same JSON and pass them off as scores.
        return scores

    # Last resort: bare floats in order. Only accept a full-length match, so a
    # stray number in prose cannot be mistaken for candidate 1's score.
    import re
    floats = re.findall(r'\b(?:0(?:\.\d+)?|1(?:\.0+)?)\b', raw)
    if len(floats) >= n:
        for i in range(n):
            scores[i] = clamp(floats[i])
    return scores


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


class ApproachFitReranker:
    """Stage-3 judge. ``score()`` -> per candidate ``{samples, mean, std, n, ...}``.

    The tau + vote gate lives in the retriever, not here: this class only
    produces samples, so tau can be re-swept over cached samples for free.
    """

    def __init__(self, model: str = config.JUDGE_MODEL,
                 cache_path: Optional[str] = None,
                 n_samples: int = config.RERANK_SAMPLES_N,
                 temperature: float = config.RERANK_TEMPERATURE,
                 max_tokens: int = config.JUDGE_MAX_TOKENS,
                 usage=None,
                 dry_run: bool = False):
        self.model = model
        self.n_samples = int(n_samples)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.dry_run = dry_run
        self._usage = usage
        self._client = None
        self.cache_path = cache_path
        self.n_infra_failures = 0
        self.n_calls = 0
        self.n_cache_hits = 0

        # Same guard as everywhere else in this harness: fabricated scores must
        # never land in the cache a real run will read back as measured.
        if dry_run and cache_path and 'DRYRUN' not in os.path.basename(cache_path):
            raise ValueError(
                f'dry_run=True with a real cache path ({cache_path}). Fabricated '
                'fit scores would be indistinguishable from judged ones on the '
                'next real run. Use a DRYRUN-prefixed path.')

        self._fresh = os.environ.get('BBEH_RERANK_FRESH', '') not in ('', '0', 'false', 'False')
        self._cache = {}
        if self.cache_path and not self._fresh:
            self._load_cache()

    # ─── cache ───────────────────────────────────────────────────────
    def _load_cache(self):
        if not os.path.exists(self.cache_path):
            return
        n = 0
        with open(self.cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key, samples = rec.get('key'), rec.get('samples')
                if key and isinstance(samples, list) and samples:
                    self._cache[key] = [float(x) for x in samples]   # last wins
                    n += 1
        logging.info('rerank cache: %d entries (%d unique) from %s',
                     n, len(self._cache), self.cache_path)

    def _append_cache(self, key: str, samples: List[float], meta: dict):
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path) or '.', exist_ok=True)
        rec = {'key': key, 'samples': samples}
        rec.update(meta or {})
        with open(self.cache_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    def _cache_key(self, query_sig: str, cand_sig: str) -> str:
        h = hashlib.sha256()
        for part in (_PROMPT_HASH, config._slug(self.model), query_sig, cand_sig):
            h.update(part.encode('utf-8'))
            h.update(b'\x1f')
        return h.hexdigest()[:32]

    # ─── client ──────────────────────────────────────────────────────
    def _get_client(self):
        if self._client is None:
            from bbeh.api_client import build_client
            self._client = build_client(self.model, temperature=self.temperature,
                                        max_tokens=self.max_tokens, usage=self._usage)
        return self._client

    def _one_sample(self, prompt: str, n_cand: int, salt: str
                    ) -> tuple:
        """One judge call -> (scores, ok). ``scores`` may contain ``None``."""
        if self.dry_run:
            # Routed through the real parser rather than returning floats
            # directly, so a dry run exercises parse_fit_scores too.
            from bbeh.api_client import dry_run_judge
            res = dry_run_judge(n_cand, salt=salt)
        else:
            self.n_calls += 1
            res = self._get_client().generate_detailed(
                prompt, max_tokens=self.max_tokens, temperature=self.temperature)
        if not res.ok:
            self.n_infra_failures += 1
            logging.warning('judge call failed (%s) — sample dropped, not cached',
                            res.error)
            return [None] * n_cand, False
        return parse_fit_scores(res.text, n_cand), True

    # ─── scoring ─────────────────────────────────────────────────────
    def score(self, query_text: str, query_task: str,
              candidates: List[dict]) -> List[dict]:
        """N judge samples per candidate, cached where possible."""
        if not candidates:
            return []
        n_cand = len(candidates)
        excerpt = query_excerpt(query_text)
        query_sig = f'{query_task}\x1f{excerpt}'
        keys = [self._cache_key(query_sig, candidate_signature(c)) for c in candidates]

        cached = [None if self._fresh else self._cache.get(k) for k in keys]
        need = [i for i, s in enumerate(cached)
                if (not s) or len(s) < self.n_samples]
        self.n_cache_hits += n_cand - len(need)

        if need:
            prompt = _RERANK_PROMPT.format(
                task=query_task or '(unknown)',
                query=excerpt,
                candidates='\n'.join(format_candidate(i + 1, c)
                                     for i, c in enumerate(candidates)),
            )
            # One call scores every candidate; N calls give the vote its samples.
            matrix = [self._one_sample(prompt, n_cand, f'{query_sig}|{s}')[0]
                      for s in range(self.n_samples)]
            for i in need:
                got = [matrix[s][i] for s in range(self.n_samples)]
                ok = [x for x in got if x is not None]
                cached[i] = ok
                # Only a complete sample set is cacheable. A partial set would
                # freeze a network blip into a permanent verdict.
                if len(ok) == self.n_samples:
                    self._cache[keys[i]] = ok
                    self._append_cache(keys[i], ok, {
                        'task': query_task,
                        'candidate': candidate_signature(candidates[i])[:200],
                    })

        results = []
        for s in cached:
            s = s or []
            results.append({
                'samples': s,
                'mean': _mean(s),
                'std': _std(s),
                'n': len(s),
                'n_requested': self.n_samples,
                'degraded': len(s) < self.n_samples,
            })
        return results

    def stats(self) -> dict:
        return {'model': self.model, 'calls': self.n_calls,
                'cache_hits': self.n_cache_hits,
                'infra_failures': self.n_infra_failures,
                'cache_entries': len(self._cache),
                'prompt_hash': _PROMPT_HASH}


# ═════════════════════════════════════════════════════════════════════
#  Selftest — parsing is where judges break, so pin it down
# ═════════════════════════════════════════════════════════════════════

def selftest() -> None:
    cases = [
        ('[{"idx":1,"fit":0.9},{"idx":2,"fit":0.0}]', 2, [0.9, 0.0]),
        ('```json\n[{"idx": 2, "fit": 1.0}, {"idx": 1, "fit": 0.4}]\n```', 2, [0.4, 1.0]),
        ('[{"fit":0.7},{"fit":0.2}]', 2, [0.7, 0.2]),          # no idx, use order
        ('[{"idx":1,"fit":1.7},{"idx":2,"fit":-3}]', 2, [1.0, 0.0]),   # clamped
        ('[{"idx":1,"fit":"high"}]', 1, [None]),               # unparseable -> None
        ('', 2, [None, None]),
        ('total garbage', 2, [None, None]),
        ('[{"idx":1,"fit":0.5}]', 2, [0.5, None]),             # short reply
        ('[{"idx":9,"fit":0.5},{"idx":1,"fit":0.3}]', 2, [0.3, None]),  # bad idx dropped
    ]
    for raw, n, want in cases:
        got = parse_fit_scores(raw, n)
        assert got == want, f'parse {raw!r} -> {got} != {want}'

    # A candidate must hash identically regardless of shortlist position, and
    # differently when its content differs.
    a = {'task': 't', 'state': 's', 'action': 'a', 'next_state': 'n',
         'abstract_pattern': {'pattern_type': 'p', 'abstract_state': 'A',
                              'abstract_action': 'B', 'abstract_next_state': 'C'}}
    b = dict(a, action='a2')
    assert candidate_signature(a) == candidate_signature(dict(a))
    assert candidate_signature(a) != candidate_signature(b)

    # The concrete step must appear in the rendered candidate — the whole
    # "trust the concrete" rule depends on it being visible.
    rendered = format_candidate(1, a)
    assert 's -> a -> n' in rendered and 'A -> B -> C' in rendered

    # Excerpting must keep the tail: BBEH puts the actual question last.
    long_q = 'HEAD' + 'x' * 50_000 + 'What is the answer?'
    ex = query_excerpt(long_q)
    assert ex.startswith('HEAD') and ex.endswith('What is the answer?')
    assert len(ex) < 3200

    # Dry run must refuse a real cache path.
    try:
        ApproachFitReranker(cache_path='/tmp/rerank_cache.jsonl', dry_run=True)
    except ValueError:
        pass
    else:
        raise AssertionError('dry_run accepted a real cache path')

    print(f'reranker selftest OK ({len(cases)} parse cases)  '
          f'prompt_hash={_PROMPT_HASH}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    selftest()
