"""
bbeh/api_client.py — OpenAI-compatible generation client.

Descended from ``adaptive_memory/api_client.py``, hardened with the failure
modes that cost us real debugging time on the PuzzleWorld runs:

  * **Empty 200 responses.** The proxy sometimes returns HTTP 200 with an empty
    body. The old client passed ``''`` through, which scored 0.0 and looked
    exactly like "the model was wrong". Here an empty body raises and is
    retried, and if it never resolves the caller gets ``error`` set — so the
    runner can label it ``infra_error`` instead of a wrong answer.
  * **No retries on timeout.** A single transient timeout permanently poisoned
    a cached result. Now: generous timeout plus exponential backoff.
  * **Truncation.** If the model hits the token ceiling mid-reasoning, the
    "The final answer is:" line never appears and the item scores 0. We surface
    ``finish_reason`` so truncation is distinguishable from a wrong answer.

Also provides a dry-run generator so the whole pipeline's plumbing (caching,
resume, aggregation, analysis) can be exercised with zero API spend.
"""

import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from bbeh import config


@dataclass
class GenResult:
    """Outcome of one generation request."""
    text: Optional[str] = None          # None => never got a usable body
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 0
    finish_reason: str = ''
    error: str = ''                     # non-empty => infra failure, not a wrong answer

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.error

    @property
    def truncated(self) -> bool:
        return self.finish_reason in ('length', 'max_tokens')


class TokenUsage:
    """Thread-safe token/call accumulator."""

    def __init__(self):
        self._lock = threading.Lock()
        self.by_model: Dict[str, Dict[str, int]] = {}

    def add(self, model: str, prompt_tokens: int, completion_tokens: int,
            calls: int = 1, errors: int = 0):
        with self._lock:
            slot = self.by_model.setdefault(
                model, {'prompt_tokens': 0, 'completion_tokens': 0,
                        'calls': 0, 'errors': 0})
            slot['prompt_tokens'] += int(prompt_tokens or 0)
            slot['completion_tokens'] += int(completion_tokens or 0)
            slot['calls'] += int(calls)
            slot['errors'] += int(errors)

    def snapshot(self) -> dict:
        with self._lock:
            out = {m: dict(v) for m, v in self.by_model.items()}
        total = {'prompt_tokens': 0, 'completion_tokens': 0, 'calls': 0, 'errors': 0}
        for v in out.values():
            for k in total:
                total[k] += v[k]
        return {'by_model': out, 'total': total}

    def write(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.snapshot(), f, indent=2, ensure_ascii=False)


AUTH_SCHEMES = ('bearer', 'x-api-key')


def _auth_headers(scheme: str, key: str) -> tuple:
    """Return ``(client_headers, per_request_headers)`` putting ``key`` on the wire.

    Two layers, and the split is not cosmetic. The OpenAI SDK always emits
    ``Authorization: Bearer <key>`` and offers no setting to turn it off, so a
    gateway that authenticates on ``x-api-key`` and validates Authorization must
    have the Bearer header actively removed — assigning ``None`` or ``''`` sends
    an empty header instead of removing it, so the SDK's ``Omit`` sentinel is the
    only mechanism.

    The sentinel has to travel as a **per-request** header. Client-level
    ``default_headers`` are merged into the outgoing headers before
    ``_validate_headers`` runs, so an Omit there does delete Authorization — and
    then validation sees no Authorization header *and* no Omit among the
    per-request headers, decides no credential was resolvable, and raises
    ``TypeError: Could not resolve authentication method``. The request never
    leaves the process. Its own message says the header must be "explicitly
    omitted", meaning at the call site.

    If a future SDK drops ``Omit`` we send both headers and warn, rather than
    failing outright: both-headers works on many gateways, and a warning beats an
    import error two hours into a paid run.
    """
    if scheme == 'bearer':
        return None, None                # the SDK's own default is correct
    if scheme != 'x-api-key':
        raise ValueError(
            f'unknown api_auth_scheme {scheme!r}; expected one of {AUTH_SCHEMES}')
    try:
        from openai._types import Omit
    except Exception:                                        # noqa: BLE001
        logging.warning(
            'openai._types.Omit is unavailable, so Authorization: Bearer cannot '
            'be suppressed; sending x-api-key alongside it. If the gateway '
            'rejects the request, this is the first thing to check.')
        return {'x-api-key': key}, None
    return {'x-api-key': key}, {'Authorization': Omit()}


class GenClient:
    """Generation client with retry, empty-body detection, and usage tracking."""

    def __init__(self, model: str,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 protocol: Optional[str] = None,
                 auth_scheme: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 1024,
                 timeout: float = config.API_TIMEOUT,
                 max_retries: int = config.API_MAX_RETRIES,
                 retry_base_delay: float = config.API_RETRY_BASE_DELAY,
                 usage: Optional[TokenUsage] = None):
        self.model = model
        self.protocol = protocol or config.API['api_protocol']
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.usage = usage

        from openai import OpenAI
        key = api_key or config.API['api_key']
        self.client = OpenAI(
            base_url=base_url or config.API['base_url'],
            api_key=key,
        )

    @property
    def _extra(self) -> dict:
        """Per-request kwargs carrying the auth headers, if any are needed."""
        return {}

    # ─── single request ──────────────────────────────────────────────
    def _request(self, prompt: str, max_tokens: int, temperature: float,
                 system: Optional[str]) -> GenResult:
        """One HTTP round trip. Raises on empty body so the caller retries."""
        if self.protocol == 'responses':
            payload = [{'role': 'user', 'content': prompt}]
            if system:
                payload.insert(0, {'role': 'system', 'content': system})
            resp = self.client.responses.create(
                model=self.model, input=payload, max_output_tokens=max_tokens,
                **self._extra,
            )
            text = resp.output_text or ''
            usage = getattr(resp, 'usage', None)
            pt = getattr(usage, 'input_tokens', 0) or 0
            ct = getattr(usage, 'output_tokens', 0) or 0
            finish = ''
        elif self.protocol == 'completions':
            resp = self.client.completions.create(
                model=self.model, prompt=prompt,
                max_tokens=max_tokens, temperature=temperature,
                **self._extra,
            )
            choice = resp.choices[0]
            text = choice.text or ''
            usage = getattr(resp, 'usage', None)
            pt = getattr(usage, 'prompt_tokens', 0) or 0
            ct = getattr(usage, 'completion_tokens', 0) or 0
            finish = getattr(choice, 'finish_reason', '') or ''
        else:  # 'chat'
            messages = []
            if system:
                messages.append({'role': 'system', 'content': system})
            messages.append({'role': 'user', 'content': prompt})
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
                **self._extra,
            )
            choice = resp.choices[0]
            text = (choice.message.content or '') if choice.message else ''
            usage = getattr(resp, 'usage', None)
            pt = getattr(usage, 'prompt_tokens', 0) or 0
            ct = getattr(usage, 'completion_tokens', 0) or 0
            finish = getattr(choice, 'finish_reason', '') or ''

        if not text.strip():
            # HTTP 200 with nothing in it. Retrying usually fixes it; passing it
            # through would masquerade as a wrong answer.
            raise ValueError(f'empty response body from API. finish_reason={finish}')

        return GenResult(text=text, prompt_tokens=pt, completion_tokens=ct,
                         finish_reason=finish)

    # ─── retrying wrapper ────────────────────────────────────────────
    def generate_detailed(self, prompt: str,
                          max_tokens: Optional[int] = None,
                          temperature: Optional[float] = None,
                          system: Optional[str] = None) -> GenResult:
        """Generate with backoff. Never raises — inspect ``.ok`` / ``.error``."""
        mt = max_tokens or self.max_tokens
        temp = self.temperature if temperature is None else temperature
        last_error = ''

        # max_retries is a count of *attempts*, so 0 would make the loop below
        # empty: no request is sent, and the caller gets back an error result
        # with a blank reason — a silent no-op that looks exactly like a failed
        # call. Nobody means "do not call the API" when they pass 0, so treat it
        # as one attempt rather than honouring a value that can only be a bug.
        attempts_allowed = max(1, self.max_retries)

        for attempt in range(1, attempts_allowed + 1):
            try:
                result = self._request(prompt, mt, temp, system)
                result.attempts = attempt
                if self.usage:
                    self.usage.add(self.model, result.prompt_tokens,
                                   result.completion_tokens)
                return result
            except Exception as e:                      # noqa: BLE001
                last_error = f'{type(e).__name__}: {e}'
                if attempt == attempts_allowed:
                    break
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                delay *= 0.75 + 0.5 * random.random()   # jitter, avoid lockstep
                logging.warning('%s call failed (%d/%d): %s — retry in %.1fs',
                                self.model, attempt, attempts_allowed,
                                last_error, delay)
                time.sleep(delay)

        logging.error('%s failed after %d attempt(s): %s', self.model,
                      attempts_allowed, last_error or '(no reason recorded)')
        if self.usage:
            self.usage.add(self.model, 0, 0, calls=1, errors=1)
        return GenResult(text=None, attempts=attempts_allowed,
                         error=last_error or 'no attempt was made')

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 system: Optional[str] = None) -> str:
        """Convenience wrapper: text on success, ``''`` on failure."""
        return self.generate_detailed(prompt, max_tokens, temperature, system).text or ''


# ═════════════════════════════════════════════════════════════════════
#  Dry run
# ═════════════════════════════════════════════════════════════════════

def dry_run_solve(item: dict, correct_rate: float = 0.35,
                  salt: str = '') -> GenResult:
    """Fabricate a solver response without touching the network.

    Deterministic per ``(item id, salt)``: emits the gold answer with roughly
    ``correct_rate`` probability, otherwise a plausible wrong one. Enough to
    exercise scoring, caching, resume, aggregation and the analysis tables —
    the numbers are meaningless, the plumbing is real.
    """
    rng = random.Random(f'{item["id"]}|{salt}')
    target = str(item.get('target', ''))
    if rng.random() < correct_rate:
        answer = target
    else:
        answer = target + '_wrong' if target else 'wrong'
    body = ('Step 1: restate the problem.\nStep 2: work through it.\n'
            '[dry-run: no model was called]\n')
    return GenResult(
        text=f'{body}The final answer is: {answer}',
        prompt_tokens=len(item.get('input', '')) // 4,
        completion_tokens=len(body) // 4,
        attempts=1, finish_reason='stop',
    )


def dry_run_teacher(item: dict, n_steps: int = 4) -> GenResult:
    """Fabricate a teacher CoT with well-formed structured steps."""
    steps = [
        {'state': f'given: input of task {item["task"]}; goal: solve step {i + 1}',
         'action': f'apply reasoning move {i + 1} (dry-run placeholder)',
         'next_state': f'intermediate result {i + 1} established'}
        for i in range(n_steps)
    ]
    payload = json.dumps({'steps': steps, 'answer': str(item.get('target', ''))},
                         ensure_ascii=False, indent=1)
    return GenResult(
        text=f'```json\n{payload}\n```\nThe final answer is: {item.get("target", "")}',
        prompt_tokens=len(item.get('input', '')) // 4,
        completion_tokens=len(payload) // 4,
        attempts=1, finish_reason='stop',
    )


def dry_run_abstract(pattern_types: List[str], n: int = 1,
                     salt: str = '') -> GenResult:
    """Fabricate an abstractor response for ``n`` triples."""
    rng = random.Random(f'abstract|{salt}')
    out = [{'abstract_state': 'a state with the entities removed',
            'abstract_action': 'the mechanism, entities stripped',
            'abstract_next_state': 'the resulting state',
            'pattern_type': rng.choice(pattern_types)} for _ in range(n)]
    payload = json.dumps(out, ensure_ascii=False)
    return GenResult(text=payload, prompt_tokens=200, completion_tokens=len(payload) // 4,
                     attempts=1, finish_reason='stop')


def dry_run_judge(n_candidates: int, salt: str = '') -> GenResult:
    """Fabricate a reranker response: a fit score per candidate."""
    rng = random.Random(f'judge|{salt}')
    arr = [{'idx': i + 1, 'fit': rng.choice([0.0, 0.2, 0.4, 0.7, 0.7, 1.0])}
           for i in range(n_candidates)]
    payload = json.dumps(arr)
    return GenResult(text=payload, prompt_tokens=400, completion_tokens=len(payload) // 4,
                     attempts=1, finish_reason='stop')


def build_client(model: str, temperature: float, max_tokens: int,
                 usage: Optional[TokenUsage] = None) -> GenClient:
    """Construct a client from :mod:`bbeh.config`."""
    return GenClient(model=model, temperature=temperature,
                     max_tokens=max_tokens, usage=usage,
                     auth_scheme=config.API.get('api_auth_scheme'))
