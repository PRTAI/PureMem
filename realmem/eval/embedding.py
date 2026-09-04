"""Shared sentence embedder.

Previously duplicated in build_memory.py and three_stage_retriever.py, and both
copies carried the same two defects:

1. ``model.get_embedding_dimension()`` — the SentenceTransformer API is
   ``get_sentence_embedding_dimension()``. The wrong name raises AttributeError.
2. ``except ImportError`` — which does not catch that AttributeError, so the
   moment sentence-transformers was actually installed the constructor crashed
   instead of falling back.

The net effect was that the code only ever ran on machines *without*
sentence-transformers, silently on the bag-of-words fallback.

The fallback is kept, because the offline selftests must run with no torch and
no network. But it is now *labelled*, and a bank records which backend built it
so a hash-encoded query can never be silently matched against ST-encoded
sessions — that mismatch produces plausible-looking nonsense rather than an
error, which is the worst failure mode available.
"""

import logging
import os
import zlib
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

BACKEND_ST = "sentence-transformers"
BACKEND_HASH = "hash-fallback"

_MODEL_CACHE: dict = {}
_FALLBACK_WARNED = False


def _load_st(model_name: str):
    from sentence_transformers import SentenceTransformer
    if model_name not in _MODEL_CACHE:
        logger.info("Loading SentenceTransformer: %s", model_name)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class Embedder:
    """Encode text to L2-normalized float32 vectors.

    Parameters
    ----------
    model_name : str
        SentenceTransformer model id.
    require_backend : str, optional
        If given, raise unless the resolved backend matches. Used when loading a
        prebuilt bank: silently mixing encoders is unrecoverable.
    allow_fallback : bool
        When False, an unavailable SentenceTransformer is an error rather than a
        downgrade to hashing.
    """

    def __init__(self, model_name: str, require_backend: Optional[str] = None,
                 allow_fallback: bool = True):
        self.model_name = model_name
        self.model = None
        try:
            self.model = _load_st(model_name)
            self.dim = int(self.model.get_sentence_embedding_dimension())
            self.backend = BACKEND_ST
        except Exception as exc:  # ImportError, AttributeError, network, OSError
            if not allow_fallback:
                raise RuntimeError(
                    f"cannot load '{model_name}' ({type(exc).__name__}: {exc})\n"
                    f"  Falling back to hash embeddings is disabled, because a bank "
                    f"built that way looks normal on disk and retrieves nonsense.\n"
                    f"  If this is a download failure behind a firewall, the endpoint "
                    f"must be set BEFORE python starts — huggingface_hub reads it into "
                    f"a module constant at import time, so setting os.environ inside "
                    f"the process is too late:\n"
                    f"      export HF_ENDPOINT=https://hf-mirror.com\n"
                    f"  Current HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '(unset)')}"
                ) from exc
            global _FALLBACK_WARNED
            if not _FALLBACK_WARNED:
                logger.warning(
                    "sentence-transformers unavailable (%s: %s) — using %s. "
                    "Vectors are NOT semantic; do not compare these numbers with "
                    "a real run.", type(exc).__name__, exc, BACKEND_HASH,
                )
                _FALLBACK_WARNED = True
            self.model = None
            self.dim = 384
            self.backend = BACKEND_HASH

        if require_backend and self.backend != require_backend:
            raise RuntimeError(
                f"embedding backend mismatch: bank was built with "
                f"'{require_backend}' but this process resolved to "
                f"'{self.backend}'. Encoding queries with a different model than "
                f"the sessions yields meaningless similarities. Install "
                f"sentence-transformers, or rebuild the bank."
            )

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.model is not None:
            vecs = self.model.encode(
                texts, batch_size=32, show_progress_bar=False,
                normalize_embeddings=normalize,
            )
            return np.asarray(vecs, dtype=np.float32)
        return self._hash_encode(texts, normalize)

    def _hash_encode(self, texts: List[str], normalize: bool) -> np.ndarray:
        """Deterministic bag-of-words hashing. No network, no torch."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in (text or "").lower().split():
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
            if normalize:
                norm = float(np.linalg.norm(out[i]))
                if norm > 0:
                    out[i] /= norm
        return out


def preflight(model_name: str = None) -> int:
    """Report whether real encoding works, before anything expensive runs.

    Run this first on a new machine:  python -m eval.embedding
    """
    import time

    model_name = model_name or os.environ.get("REALMEM_EMBEDDING_MODEL",
                                              "all-MiniLM-L6-v2")
    print("embedding preflight")
    print("-" * 58)
    print(f"  model         {model_name}")
    print(f"  HF_ENDPOINT   {os.environ.get('HF_ENDPOINT', '(unset -> huggingface.co)')}")
    print(f"  HF_HUB_OFFLINE {os.environ.get('HF_HUB_OFFLINE', '(unset)')}")

    t0 = time.time()
    try:
        emb = Embedder(model_name, allow_fallback=False)
    except RuntimeError as exc:
        print(f"\n  FAIL  {exc}")
        return 1

    vecs = emb.encode(["a sentence about travel plans",
                       "an unrelated sentence about fitness"])
    took = time.time() - t0
    sim = float(vecs[0] @ vecs[1])
    print(f"  backend       {emb.backend}")
    print(f"  dim           {emb.dim}")
    print(f"  load+encode   {took:.1f}s")
    print(f"  sanity        cos(travel, fitness) = {sim:.3f}")

    if emb.backend != BACKEND_ST:
        print("\n  FAIL  resolved to the hash fallback despite allow_fallback=False")
        return 1
    if not looks_like_st_vectors(vecs):
        print("\n  FAIL  vectors have no negative components — not a real encoder")
        return 1
    print("\n  OK    real semantic encoding is available")
    return 0


def looks_like_st_vectors(matrix: np.ndarray) -> bool:
    """Heuristic used by bank verification.

    Hash vectors are non-negative and sparse; MiniLM vectors are dense with
    roughly half the coordinates negative. Lets us detect a bank whose meta.json
    claims one backend while the .npy was written by the other.
    """
    if matrix.size == 0:
        return False
    return float((matrix < 0).mean()) > 0.2


if __name__ == "__main__":
    import sys
    sys.exit(preflight())
