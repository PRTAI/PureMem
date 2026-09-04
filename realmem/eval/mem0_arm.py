"""mem0 as an external baseline arm.

Kept in a separate module so that importing the core harness never pulls in
mem0. Not part of DEFAULT_ARMS: it needs its own LLM and embedding credentials
and costs real money per added session, so it is opt-in via
``--arms ...,mem0``.

Two departures from the previous implementation, both about not manufacturing a
result:

* A missing mem0 install used to make ``retrieve`` return ``[]`` for every
  query. That is indistinguishable from "mem0 retrieved nothing", i.e. an
  uninstalled dependency would have been written up as a baseline scoring zero.
  It now raises at construction.
* Session text is remembered only for sessions already admitted, so the arm
  obeys the same streaming contract as every other one.

**API COMPATIBILITY.** mem0's search signature changed between 1.x and 2.x:

    1.x   memory.search(query, user_id=..., limit=N)
    2.x   memory.search(query, top_k=N, filters={"user_id": ...})

Both dropped parameters land in 2.x's ``**kwargs`` and are silently ignored —
no error, but the user filter and the result cap both stop working, so the arm
would score against an unfiltered, arbitrarily-sized result set. The vendored
``eval_mem0.py`` is written against the 1.x form and has this bug on any modern
install. We therefore inspect the signature at construction and adapt.

HONESTY NOTE: the calling convention is verified against mem0ai 2.0.19, but no
end-to-end run has been performed (it needs live LLM + embedding credentials).
Treat the first real run as a smoke test: check `depth` and `empty_rate` in the
diagnostics before trusting any number it produces.
"""

import inspect
import logging
import os
import re
from typing import Any, Dict, List, Optional

from eval.config import (API_KEY, BASE_URL, MEM0_LLM_MODEL, RETRIEVE_K,
                         EMBEDDING_MODEL, MEM0_EMBED_PROVIDER, MEM0_STORAGE_ROOT)
from eval import schema

logger = logging.getLogger(__name__)

# all-MiniLM-L6-v2. Qdrant needs the dimension up front and will reject vectors
# that disagree with the collection it already created.
EMBEDDING_DIMS = 384

# Fragments requested per query = RETRIEVE_K * this. Must be generous: mem0
# returns memory points, and dedup to sessions shrinks the list a long way.
MEM0_FETCH_MULTIPLIER = int(os.environ.get("REALMEM_MEM0_FETCH_MULTIPLIER", "10"))


class Mem0Retriever:
    """Streaming mem0 baseline exposing the same interface as ThreeStageRetriever."""

    def __init__(self, retrieve_k: int = RETRIEVE_K, user_id: str = "realmem_eval",
                 llm_model: str = MEM0_LLM_MODEL,
                 embed_model: Optional[str] = None,
                 embed_provider: str = MEM0_EMBED_PROVIDER,
                 storage_dir: Optional[str] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        # mem0 ships anonymous PostHog telemetry, on by default. It is read at
        # import time (mem0/memory/telemetry.py: MEM0_TELEMETRY =
        # os.environ.get(...)), so this must happen before the import below —
        # setting it afterwards has no effect, the same trap as HF_ENDPOINT.
        #
        # Disabled here for three reasons: us.i.posthog.com is unreachable from
        # many networks and its 0.5s timeout floods the log with tracebacks on
        # every call; the retries slow down an already slow write path; and an
        # evaluation run should not be shipping metadata to a third party.
        os.environ.setdefault("MEM0_TELEMETRY", "False")

        try:
            from mem0 import Memory
            from mem0.configs.base import MemoryConfig
        except ImportError as exc:
            raise RuntimeError(
                "the 'mem0' arm was requested but mem0 is not installed. "
                "Install it (pip install mem0ai) or drop mem0 from --arms. "
                "Refusing to run, because an arm that silently retrieves nothing "
                "would be recorded as a baseline that scores zero."
            ) from exc

        self.retrieve_k = retrieve_k
        self.user_id = user_id
        # shallow_results counts queries that came back with fewer sessions than
        # the metric asks for — the arm's own depth-ceiling alarm.
        self.stats = {"queries": 0, "add_failures": 0, "search_failures": 0,
                      "shallow_results": 0, "fragments_seen": 0, "sessions_kept": 0}
        self._session_text: Dict[str, str] = {}
        self.n_sessions = 0

        key = api_key or API_KEY
        url = base_url or BASE_URL

        # ── Embedder ──
        # Default to a LOCAL sentence-transformers encoder rather than an API
        # one. Two reasons, one practical and one methodological:
        #
        #   practical      the configured gateway may not proxy embeddings at
        #                  all. Measured on one such gateway: text-embedding-3-small
        #                  and text-embedding-ada-002 both return HTTP 503
        #                  "no available channel", so every mem0 add() fails
        #                  after retries with a bare "Connection error".
        #   methodological using the same encoder as our own arms
        #                  (all-MiniLM-L6-v2) means the comparison isolates the
        #                  memory architecture instead of confounding it with
        #                  embedding quality.
        #
        # The paper used text-embedding-3-small; set embed_provider='openai' to
        # match it, but only if the gateway actually serves embeddings.
        if embed_provider == "huggingface":
            model = embed_model or EMBEDDING_MODEL
            embedder = {"provider": "huggingface",
                        "config": {"model": model, "embedding_dims": EMBEDDING_DIMS}}
            dims = EMBEDDING_DIMS
        elif embed_provider == "openai":
            model = embed_model or "text-embedding-3-small"
            embedder = {"provider": "openai",
                        "config": {"model": model, "api_key": key,
                                   "openai_base_url": url}}
            dims = 1536
        else:
            raise ValueError(f"embed_provider must be 'huggingface' or 'openai', "
                             f"got {embed_provider!r}")

        # ── Vector store ──
        # mem0 defaults to a local Qdrant at the fixed path /tmp/qdrant with the
        # fixed collection name 'mem0'. Both are shared state, and both bite:
        # a second process raises "Storage folder /tmp/qdrant is already accessed
        # by another instance", so personas cannot run in parallel; and a reused
        # collection lets one persona's memories leak into the next one's
        # retrieval. Give every persona its own directory and collection.
        store_path = storage_dir or os.path.join(
            MEM0_STORAGE_ROOT, re.sub(r"[^A-Za-z0-9_.-]", "_", user_id))
        os.makedirs(store_path, exist_ok=True)

        self._memory = Memory(MemoryConfig(
            llm={"provider": "openai", "config": {
                "model": llm_model, "api_key": key, "openai_base_url": url}},
            embedder=embedder,
            vector_store={"provider": "qdrant", "config": {
                "path": store_path,
                "collection_name": f"mem0_{re.sub(r'[^A-Za-z0-9_]', '_', user_id)}",
                "embedding_model_dims": dims,
            }},
        ))
        logger.info("mem0 llm=%s, embedder=%s/%s (dims=%d), store=%s",
                    llm_model, embed_provider, model, dims, store_path)

        # Resolve the calling convention once, rather than passing arguments
        # that a newer mem0 would swallow into **kwargs without applying.
        params = inspect.signature(self._memory.search).parameters
        self._search_topk_kw = "top_k" if "top_k" in params else "limit"
        self._search_uses_filters = "filters" in params and "user_id" not in params
        # mem0 2.x defaults to threshold=0.1, silently dropping weak matches.
        # That is sensible for an assistant and wrong for a Recall@k benchmark:
        # it would shorten the ranked list without saying so, capping recall@20
        # the same way POOL_SIZE=15 once did. Ask for everything and let the
        # metric do the cutting.
        self._search_has_threshold = "threshold" in params
        logger.info("mem0 search convention: %s=<n>, %s%s",
                    self._search_topk_kw,
                    'filters={"user_id": ...}' if self._search_uses_filters
                    else "user_id=<id>",
                    ", threshold=0.0" if self._search_has_threshold else "")

    def reset(self):
        self._session_text.clear()
        self.n_sessions = 0
        try:
            self._memory.delete_all(user_id=self.user_id)
        except Exception:
            try:
                self._memory.reset()
            except Exception as exc:
                # Leftover memories from a previous persona would leak across
                # personas, which is worse than failing here.
                raise RuntimeError(
                    f"could not clear mem0 state between personas: {exc}") from exc

    def attach_bank(self, bank_dir: str):
        """mem0 owns its own store; the precomputed vectors do not apply."""
        return

    def add_session(self, session: dict):
        sid = session.get("session_identifier", "")
        text = schema.session_text(session)
        if not text.strip():
            return
        self._session_text[sid] = text
        self.n_sessions += 1
        try:
            self._memory.add(text, user_id=self.user_id,
                             metadata={"chunk_id": sid, "session_identifier": sid})
        except Exception as exc:
            self.stats["add_failures"] += 1
            logger.warning("mem0 add failed for %s: %s", sid, exc)

    def retrieve(self, query: str, query_topic: Optional[str] = None,
                 keywords: Optional[str] = None) -> List[Dict]:
        self.stats["queries"] += 1
        if not self._session_text:
            return []

        text = f"{query}\n{keywords}" if keywords else query
        # Ask for far more FRAGMENTS than the SESSIONS we need. mem0 returns
        # extracted memory points, several of which routinely belong to the same
        # session, so the fragment count collapses on dedup. Requesting 2x left
        # 44% of queries below 20 sessions (mean depth 15.6), structurally
        # capping recall@20 exactly the way POOL_SIZE=15 once capped ours.
        # Over-fetching costs nothing: search is a vector lookup, not an LLM call.
        kwargs: Dict[str, Any] = {self._search_topk_kw: self.retrieve_k * MEM0_FETCH_MULTIPLIER}
        if self._search_uses_filters:
            kwargs["filters"] = {"user_id": self.user_id}
        else:
            kwargs["user_id"] = self.user_id
        if self._search_has_threshold:
            kwargs["threshold"] = 0.0
        try:
            results = self._memory.search(query=text, **kwargs)
        except Exception as exc:
            self.stats["search_failures"] += 1
            logger.warning("mem0 search failed: %s", exc)
            return []

        entries = results.get("results", []) if isinstance(results, dict) else (results or [])

        # mem0 returns extracted memory fragments; the metric's unit is a
        # session. Collapse to sessions, keeping each one's best score and
        # earliest position.
        best: Dict[str, dict] = {}
        for rank, entry in enumerate(entries, 1):
            meta = entry.get("metadata") or {}
            sid = meta.get("chunk_id") or meta.get("session_identifier")
            if not sid or sid not in self._session_text:
                # Unresolvable or not-yet-admitted: dropping it keeps the
                # streaming contract and matches how unknown ids are handled
                # by the official scorer.
                continue
            score = float(entry.get("score") or 0.0)
            prev = best.get(sid)
            if prev is None or score > prev["score"]:
                best[sid] = {"score": score, "rank": min(rank, prev["rank"]) if prev else rank}
            elif rank < prev["rank"]:
                prev["rank"] = rank

        ordered = sorted(best.items(), key=lambda kv: (-kv[1]["score"], kv[1]["rank"]))
        self.stats["fragments_seen"] += len(entries)
        self.stats["sessions_kept"] += len(ordered)
        if len(ordered) < self.retrieve_k and len(self._session_text) >= self.retrieve_k:
            # Fewer sessions than the metric wants, while history had enough to
            # supply them: the fragment budget, not the corpus, is the limit.
            self.stats["shallow_results"] += 1
        return [
            {
                "res_type": "chunk",
                "chunk_id": sid,
                "content": self._session_text[sid],
                "score": round(info["score"], 4),
                "rank": i,
                "stage2_rank": info["rank"],
                "fit_votes": None,
                "fit_mean": None,
                "stage3_passed": None,
                "same_topic": (query_topic is not None
                               and schema.parse_topic(sid) == query_topic),
                "has_abstract": None,
            }
            for i, (sid, info) in enumerate(ordered[: self.retrieve_k], 1)
        ]
