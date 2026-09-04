"""
Lightweight OpenAI-compatible API client with retry and token estimation.

Reuses the protocol logic from common/api_client.py but is self-contained
so the adaptive_memory module has no cross-dependency.
"""

import logging
import time
from typing import Optional


def _retry_with_backoff(fn, max_retries=3, base_delay=1.0):
    """Run fn() with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logging.warning('API call failed (attempt %d/%d): %s. Retrying in %.1fs...',
                          attempt + 1, max_retries, e, delay)
            time.sleep(delay)


class GenClient:
    """Wrapper for the generation model (OpenAI-compatible API)."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 protocol: str = 'chat', temperature: float = 0.0,
                 max_tokens: int = 256):
        self.model = model
        self.protocol = protocol
        self.temperature = temperature
        self.max_tokens = max_tokens

        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, prompt: str, max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """Generate a single completion."""
        mt = max_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        if self.protocol == 'chat':
            def _call():
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=mt,
                    temperature=temp,
                )
                return resp.choices[0].message.content or ''
        elif self.protocol == 'responses':
            def _call():
                resp = self.client.responses.create(
                    model=self.model,
                    input=[{'role': 'user', 'content': prompt}],
                    max_output_tokens=mt,
                )
                return resp.output_text or ''
        else:
            def _call():
                resp = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=mt,
                    temperature=temp,
                )
                return resp.choices[0].text or ''

        return _retry_with_backoff(_call)

    def estimate_tokens(self, text: str) -> int:
        """Rough token count: 1 token per 4 chars."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding('cl100k_base')
            return len(enc.encode(text))
        except (ImportError, Exception):
            return max(1, len(text) // 4)
