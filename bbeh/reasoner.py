"""
bbeh/reasoner.py — the two solve arms, sharing one code path on purpose.

Claim 2 is a *paired* comparison: same student, same items, same prompt, the
only difference being an injected precedent block. That makes the arms' shared
code path the load-bearing part of the experiment. So both arms are the same
function with a different precedent list:

    no_memory  ->  solve(item, precedents=[])
    memory     ->  solve(item, precedents=retrieved)

There is deliberately no separate "no-memory prompt builder" to drift out of
sync with the memory one; ``prompts.build_solve_prompt`` handles both and
``prompts.selftest_arm_parity`` asserts that the memory prompt is the bare
prompt plus a prefix. A consequence worth stating: when the Stage-3 gate rejects
everything, the memory arm issues a *byte-identical* request to the no_memory
arm. That is correct — those items should score identically, and if they don't,
the difference is sampling noise, which is itself a useful measurement.

Three outcomes are recorded separately, never conflated:

    correct / incorrect   the model answered; the scorer ruled
    truncated             hit max_tokens before emitting the answer line
    infra_error           no usable response at all

Only the first two belong in an accuracy denominator. Folding infra errors in as
wrong answers is how an unstable afternoon on a proxy endpoint turns into a
publishable-looking regression.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List

from bbeh import config, official_eval, prompts


@dataclass
class SolveResult:
    id: str
    task: str
    arm: str
    correct: bool = False
    prediction: str = ''
    reference: str = ''
    response: str = ''
    outcome: str = 'ok'            # ok | truncated | infra_error
    error: str = ''
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ''
    latency_s: float = 0.0
    # memory arm only
    n_injected: int = 0
    injected: List[dict] = field(default_factory=list)
    retrieval_error: str = ''

    @property
    def scorable(self) -> bool:
        """Whether this attempt belongs in an accuracy denominator."""
        return self.outcome != 'infra_error'

    def to_record(self) -> dict:
        d = dict(self.__dict__)
        d['scorable'] = self.scorable
        return d


def _looks_truncated(text: str, finish_reason: str) -> bool:
    """Did the response stop before committing to an answer?

    Two independent signals, because neither alone is reliable: the API's
    ``finish_reason`` is not always populated by proxies, and a model can also
    ramble to the token ceiling while emitting an answer line earlier. Requiring
    the answer line to be *absent* keeps a long-but-complete response scorable.
    """
    if not text:
        return False
    has_answer = any(p in text for p in (
        'The final answer is', 'The answer is', 'final answer is:'))
    return (finish_reason or '').lower() in ('length', 'max_tokens') and not has_answer


class StandardReasoner:
    """The bare baseline: question in, answer out. Claim 2's control arm."""

    arm = 'no_memory'

    def __init__(self, client, max_tokens: int = config.SOLVE_MAX_TOKENS,
                 temperature: float = config.SOLVE_TEMPERATURE):
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature

    def build_prompt(self, item: dict) -> tuple:
        return prompts.build_solve_prompt(item, ()), []

    def solve(self, item: dict) -> SolveResult:
        t0 = time.perf_counter()
        prompt, injected = self.build_prompt(item)
        res = self.client.generate_detailed(
            prompt, max_tokens=self.max_tokens, temperature=self.temperature)

        out = SolveResult(id=item['id'], task=item['task'], arm=self.arm,
                          reference=item.get('target', ''),
                          prompt_tokens=res.prompt_tokens,
                          completion_tokens=res.completion_tokens,
                          finish_reason=getattr(res, 'finish_reason', '') or '',
                          latency_s=round(time.perf_counter() - t0, 2),
                          n_injected=len(injected),
                          injected=injected)
        if not res.ok or not (res.text or '').strip():
            out.outcome = 'infra_error'
            out.error = res.error or 'empty response body'
            return out

        out.response = res.text
        correct, pred, ref = official_eval.score_with_detail(res.text, item['target'])
        out.correct, out.prediction, out.reference = correct, pred, ref
        if not correct and _looks_truncated(res.text, out.finish_reason):
            # Still counted as an attempt and still scored False — the model did
            # fail to produce an answer. Flagged so that a wave of truncations
            # shows up as "raise SOLVE_MAX_TOKENS" instead of "memory hurt".
            out.outcome = 'truncated'
        return out


class MemoryAugmentedReasoner(StandardReasoner):
    """The treatment arm: identical, plus a gated precedent block."""

    arm = 'memory'

    def __init__(self, client, retriever, query_embeddings, reranker=None,
                 top_k: int = config.TOP_K,
                 max_tokens: int = config.SOLVE_MAX_TOKENS,
                 temperature: float = config.SOLVE_TEMPERATURE,
                 **retrieval_kwargs):
        super().__init__(client, max_tokens=max_tokens, temperature=temperature)
        self.retriever = retriever
        self.reranker = reranker
        self.query_embeddings = query_embeddings
        self.top_k = top_k
        self.retrieval_kwargs = retrieval_kwargs

    def build_prompt(self, item: dict) -> tuple:
        try:
            chunks = self.retriever.retrieve_three_stage(
                item['input'], self.query_embeddings[item['id']],
                query_task=item['task'], reranker=self.reranker,
                top_k_chunks=self.top_k, **self.retrieval_kwargs)
        except Exception as exc:
            # Retrieval must never take down a solve. Falling back to the bare
            # prompt keeps the item scorable; the failure is recorded so it does
            # not masquerade as "the gate rejected everything".
            logging.warning('retrieval failed for %s (%s) — solving bare',
                            item['id'], exc)
            self._last_retrieval_error = f'{type(exc).__name__}: {exc}'
            return prompts.build_solve_prompt(item, ()), []
        self._last_retrieval_error = ''
        return prompts.build_solve_prompt(item, chunks), chunks

    def solve(self, item: dict) -> SolveResult:
        self._last_retrieval_error = ''
        out = super().solve(item)
        out.retrieval_error = getattr(self, '_last_retrieval_error', '')
        # Keep the audit trail, not the payload: the chunk text is already in
        # the memory version, and duplicating it per item would balloon
        # results.jsonl for no analytical gain.
        out.injected = [{
            'chunk_id': c.get('chunk_id'),
            'item_id': c.get('item_id'),
            'task': c.get('task'),
            'same_task': c.get('task') == item['task'],
            'pattern_type': c.get('pattern_type', ''),
            'retrieval_layer': c.get('retrieval_layer', ''),
            'similarity': round(float(c.get('similarity', 0.0)), 4),
            'tag_bonus': round(float(c.get('tag_bonus', 0.0)), 4),
            'fit_mean': round(float(c.get('fit_mean') or 0.0), 4),
            'fit_votes': c.get('fit_votes'),
            'fit_degraded': c.get('fit_degraded', False),
        } for c in out.injected]
        out.n_injected = len(out.injected)
        return out


def build_reasoner(arm: str, client, **kwargs):
    if arm == 'no_memory':
        return StandardReasoner(client,
                                max_tokens=kwargs.get('max_tokens', config.SOLVE_MAX_TOKENS),
                                temperature=kwargs.get('temperature', config.SOLVE_TEMPERATURE))
    if arm == 'memory':
        return MemoryAugmentedReasoner(client, **kwargs)
    raise ValueError(f'unknown arm {arm!r} (expected no_memory | memory)')
