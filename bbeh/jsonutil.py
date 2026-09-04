"""
bbeh/jsonutil.py — pull JSON out of a chatty LLM response.

Every structured call in this harness (teacher CoT, abstraction, rerank) asks
for bare JSON and sometimes gets a markdown fence, a preamble, a trailing
"Hope this helps!", or a stray trailing comma. A parse failure here is
indistinguishable downstream from "the model refused", so the extractor is
deliberately forgiving about packaging and strict about content.

The scanner is brace-balancing that respects string literals and escapes — a
naive ``text[text.find('{'):text.rfind('}')+1]`` breaks on any JSON whose
string values contain braces, which happens constantly with BBEH inputs about
sets, dicts and code.
"""

import json
import re
from typing import Any, List, Optional

_FENCE_RE = re.compile(r'```(?:json|JSON)?\s*(.*?)```', re.DOTALL)


def _strip_fences(text: str) -> str:
    """Return fenced content if a fence exists, else the text unchanged."""
    blocks = _FENCE_RE.findall(text)
    if blocks:
        # Longest block: models sometimes emit a tiny illustrative fence first.
        return max(blocks, key=len).strip()
    return text.strip()


def _scan_balanced(text: str, start: int, open_ch: str, close_ch: str) -> Optional[str]:
    """Slice from ``start`` to the matching close, respecting string literals."""
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _repair(blob: str) -> str:
    """Fix the two malformations that actually occur in practice."""
    # Trailing commas before a close.
    blob = re.sub(r',(\s*[}\]])', r'\1', blob)
    # Literal newlines inside string values (models pretty-print CoT that way).
    out, in_str, escaped = [], False, False
    for ch in blob:
        if in_str:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_str = False
            elif ch == '\n':
                out.append('\\n')
                continue
            elif ch == '\t':
                out.append('\\t')
                continue
        elif ch == '"':
            in_str = True
        out.append(ch)
    return ''.join(out)


def extract_json(text: Optional[str], expect: str = 'object') -> Optional[Any]:
    """Best-effort parse of the first JSON value in ``text``.

    Args:
        text: raw model output, or None.
        expect: ``'object'``, ``'array'``, or ``'any'`` — which container to
            hunt for first. ``'any'`` takes whichever appears earlier.

    Returns:
        The parsed value, or ``None`` if nothing parseable was found. Callers
        must treat ``None`` as a retryable model failure, not as empty data.
    """
    if not text:
        return None
    body = _strip_fences(text)

    # Fast path: the whole thing is valid JSON.
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        pass

    pairs = {'object': [('{', '}')], 'array': [('[', ']')],
             'any': [('{', '}'), ('[', ']')]}[expect]
    if expect == 'any':
        firsts = [(body.find(o), o, c) for o, c in pairs]
        firsts = sorted((p for p in firsts if p[0] >= 0))
        pairs = [(o, c) for _, o, c in firsts]

    for open_ch, close_ch in pairs:
        pos = body.find(open_ch)
        while pos >= 0:
            blob = _scan_balanced(body, pos, open_ch, close_ch)
            if blob:
                for candidate in (blob, _repair(blob)):
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        continue
            pos = body.find(open_ch, pos + 1)
    return None


def extract_json_objects(text: Optional[str]) -> List[dict]:
    """Every top-level JSON object in ``text``, in order.

    For responses that emit one object per line instead of the requested array.
    """
    if not text:
        return []
    body = _strip_fences(text)
    out, pos = [], body.find('{')
    while pos >= 0:
        blob = _scan_balanced(body, pos, '{', '}')
        if not blob:
            break
        for candidate in (blob, _repair(blob)):
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
            break
        pos = body.find('{', pos + len(blob))
    return out


def _selftest() -> None:
    cases = [
        # plain
        ('{"a": 1}', 'object', {'a': 1}),
        # fenced
        ('Sure!\n```json\n{"a": 2}\n```\nHope that helps!', 'object', {'a': 2}),
        # fenced without a language tag
        ('```\n{"a": 3}\n```', 'object', {'a': 3}),
        # preamble + trailing prose
        ('Here you go: {"a": 4} — let me know.', 'object', {'a': 4}),
        # trailing comma
        ('{"a": 5,}', 'object', {'a': 5}),
        # braces INSIDE a string value: the naive rfind approach fails here
        ('{"action": "compute f(x) = {x | x > 2} then sort"}', 'object',
         {'action': 'compute f(x) = {x | x > 2} then sort'}),
        # array
        ('[{"idx": 1, "fit": 0.7}]', 'array', [{'idx': 1, 'fit': 0.7}]),
        # array requested, prose around it
        ('Scores:\n[{"idx":1,"fit":0.0}]\ndone', 'array', [{'idx': 1, 'fit': 0.0}]),
        # nested
        ('{"steps": [{"state": "s", "action": "a"}], "answer": "4"}', 'object',
         {'steps': [{'state': 's', 'action': 'a'}], 'answer': '4'}),
        # nothing there
        ('I cannot help with that.', 'object', None),
        (None, 'object', None),
        ('', 'object', None),
    ]
    for text, expect, want in cases:
        got = extract_json(text, expect)
        assert got == want, f'{text!r}: got {got!r}, want {want!r}'

    # A raw newline inside a string value is invalid JSON but very common.
    got = extract_json('{"action": "line one\nline two"}')
    assert got == {'action': 'line one\nline two'}, got

    # Two objects, one per line.
    objs = extract_json_objects('{"a":1}\n{"b":2}\n')
    assert objs == [{'a': 1}, {'b': 2}], objs

    # An object whose string contains a fake closing brace must not truncate.
    tricky = '{"s": "the set }{ is weird", "n": 7}'
    assert extract_json(tricky) == {'s': 'the set }{ is weird', 'n': 7}

    print(f'jsonutil selftest OK ({len(cases) + 3} cases)')


if __name__ == '__main__':
    _selftest()
