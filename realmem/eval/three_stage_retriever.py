"""Three-stage retrieval, ported to RealMemBench.

  Stage 1  soft tag reweighting  — additive bonuses, never a filter
  Stage 2  dual-pool recall      — abstract + concrete, blended
  Stage 3  LLM approach-fit rerank + majority-vote gate

Two things differ from the BBEH/PuzzleWorld original, both forced by the
benchmark rather than chosen:

**Streaming.** The published protocol retrieves against history and only then
admits the current session. Gold references point backwards 100% of the time
(0 of 2319 point forward), so a retriever holding the whole corpus is not merely
cheating, it is *worse*: the future sessions it can see are never gold and
crowd out real answers. ``add_session`` is therefore the only way in, and
``attach_bank`` supplies precomputed vectors without making anything visible.

**Gate mode.** Upstream, Stage 3 returns [] when no candidate clears tau, and
the solver degrades to no-memory — correct when injecting a bad precedent
actively hurts. RealMem scores Recall@k, where an empty list is simply zero. So
``gate_mode`` selects:

  'rerank'  candidates that fail the vote are demoted, not dropped; the list
            stays RETRIEVE_K long and Recall@k remains comparable to baseline.
  'gate'    upstream semantics, preserved exactly, as a separate arm.
"""

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

from eval.config import (
    EMBEDDING_MODEL, STAGE3_MODEL, API_KEY, BASE_URL, API_TIMEOUT, BANK_FORMAT,
    RETRIEVE_K, POOL_SIZE, TOP_N_SESSIONS, MAX_METRIC_K,
    TAU, VOTE_THRESHOLD, STAGE3_SAMPLES, STAGE3_TEMPERATURE,
    STAGE3_SAMPLE_WORKERS, RERANK_CANDIDATES_M, GATE_MODE,
    STAGE3_ABSTRACT_CHARS, STAGE3_RAW_FALLBACK_CHARS, RERANK_CACHE_NAME,
    CONCRETE_W, ABSTRACT_W,
    STAGE1_ABSTRACT_WEIGHT, STAGE1_TOPIC_WEIGHT,
    STAGE1_RECENCY_WEIGHT, STAGE1_RECENCY_SCALE, TOPIC_SOURCE,
)
from eval.embedding import Embedder
from eval import schema

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v3-realmem-approach-fit"

# Stage-3 sample caches, keyed by cache file path. Every retriever pointed at the
# same file shares one dict, so an arm that judges a candidate list makes that
# result visible to the others immediately rather than at next construction.
_CACHE_REGISTRY: Dict[str, Dict[str, List[List[float]]]] = {}


def _shared_cache(path: str) -> Dict[str, List[List[float]]]:
    key = os.path.abspath(path)
    if key not in _CACHE_REGISTRY:
        cache: Dict[str, List[List[float]]] = {}
        if os.path.exists(key):
            try:
                with open(key, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rec = json.loads(line)
                            cache[rec["key"]] = rec["samples"]
                logger.info("Loaded %d cached Stage-3 results from %s", len(cache), key)
            except Exception as exc:
                logger.warning("Could not read rerank cache %s: %s", key, exc)
        _CACHE_REGISTRY[key] = cache
    return _CACHE_REGISTRY[key]


def reset_cache_registry():
    """Drop in-memory caches. For tests; on disk the files are untouched."""
    _CACHE_REGISTRY.clear()


class _GrowableMatrix:
    """Append-only row buffer with amortized growth.

    add_session is called once per session, so np.vstack per call would make
    bank assembly quadratic in corpus size.
    """

    def __init__(self, dim: int, capacity: int = 64):
        self._buf = np.zeros((capacity, dim), dtype=np.float32)
        self._n = 0
        self.dim = dim

    def append(self, row: np.ndarray):
        if self._n == self._buf.shape[0]:
            self._buf = np.vstack([self._buf, np.zeros_like(self._buf)])
        self._buf[self._n] = row
        self._n += 1

    @property
    def view(self) -> np.ndarray:
        return self._buf[: self._n]

    def __len__(self) -> int:
        return self._n


class ThreeStageRetriever:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        retrieve_k: int = RETRIEVE_K,
        pool_size: int = POOL_SIZE,
        top_n_sessions: int = TOP_N_SESSIONS,
        enable_stage1: bool = True,
        enable_stage3: bool = True,
        gate_mode: str = GATE_MODE,
        stage3_model: str = STAGE3_MODEL,
        stage3_samples: int = STAGE3_SAMPLES,
        rerank_candidates: int = RERANK_CANDIDATES_M,
        tau: float = TAU,
        vote_threshold: int = VOTE_THRESHOLD,
        topic_source: str = TOPIC_SOURCE,
        llm_client: Any = None,
        cache_path: Optional[str] = None,
        sample_workers: int = STAGE3_SAMPLE_WORKERS,
        enable_dual_pool: bool = True,
    ):
        if gate_mode not in ("rerank", "gate"):
            raise ValueError(f"gate_mode must be 'rerank' or 'gate', got {gate_mode!r}")

        self.embedder = embedder or Embedder(EMBEDDING_MODEL)
        self.retrieve_k = retrieve_k
        self.pool_size = max(pool_size, retrieve_k)
        self.top_n_sessions = top_n_sessions
        self.enable_stage1 = enable_stage1
        self.enable_stage3 = enable_stage3
        # False collapses Stage 2 to a single concrete pool (session text only),
        # which is what "plain cosine retrieval" actually means. The default
        # blend is itself a stage, so a control that keeps it cannot measure it.
        self.enable_dual_pool = enable_dual_pool
        self.gate_mode = gate_mode
        self.stage3_model = stage3_model
        self.stage3_samples = stage3_samples
        self.rerank_candidates = rerank_candidates
        self.tau = tau
        self.vote_threshold = vote_threshold
        self.topic_source = topic_source
        self.llm_client = llm_client
        self.sample_workers = sample_workers

        # Shared per cache file, NOT per instance. run_eval builds every arm up
        # front, so a per-instance snapshot taken at construction time is empty
        # for all of them; the rerank arm would then fill the file while the
        # gated arm — holding its own stale copy — re-paid for identical
        # judgements. Both arms score the same Stage-2 candidates, so that was
        # exactly double the Stage-3 bill and double the wall clock.
        self._cache_path = cache_path
        self._cache = _shared_cache(cache_path) if cache_path else {}

        self.stats = {
            "queries": 0, "gated_empty": 0, "stage3_calls": 0,
            "stage3_cache_hits": 0, "stage3_errors": 0,
        }
        self.reset()

    # ── State ──

    def reset(self):
        """Drop every admitted session. Called between personas."""
        dim = self.embedder.dim
        self._sessions: List[Dict] = []
        self._sess_emb = _GrowableMatrix(dim)
        self._topics: List[Optional[str]] = []
        self._has_abs: List[bool] = []
        self._sid_to_idx: Dict[str, int] = {}

        self._abstracts: List[Dict] = []
        self._abs_emb = _GrowableMatrix(dim)
        self._abs_owner: List[int] = []
        self._abs_by_sid: Dict[str, List[Dict]] = {}

        self._topic_arr: Optional[np.ndarray] = None
        self._has_abs_arr: Optional[np.ndarray] = None

    # ── Bank as an embedding cache ──

    def attach_bank(self, bank_dir: str):
        """Load precomputed vectors. Does NOT make any session retrievable.

        Sessions become visible only through add_session; this just spares us
        re-encoding text we already embedded offline.
        """
        def read_jsonl(name):
            path = os.path.join(bank_dir, name)
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

        with open(os.path.join(bank_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        found_format = meta.get("bank_format", 1)
        if found_format != BANK_FORMAT:
            raise RuntimeError(
                f"{bank_dir}: bank_format {found_format}, expected {BANK_FORMAT}. "
                f"Rebuild with: python -m eval.build_memory --all-personas "
                f"--require-st --force")

        # A hash-encoded query against ST-encoded sessions has the right shape
        # and produces confident nonsense. Nothing downstream could detect it,
        # so refuse here.
        bank_backend = meta.get("embedding_backend")
        if bank_backend and bank_backend != self.embedder.backend:
            raise RuntimeError(
                f"{bank_dir}: bank was built with '{bank_backend}' but queries "
                f"would be encoded with '{self.embedder.backend}'. Mixing encoders "
                f"yields meaningless similarities with no visible error. Install "
                f"sentence-transformers, or rebuild the bank with the current "
                f"backend (build_memory.py --force)."
            )

        sessions = read_jsonl("sessions.jsonl")
        abstracts = read_jsonl("abstracts.jsonl")
        sess_emb = np.load(os.path.join(bank_dir, "session_embeddings.npy"))
        abs_emb = np.load(os.path.join(bank_dir, "abstract_embeddings.npy"))

        if sess_emb.shape[0] != len(sessions) or abs_emb.shape[0] != len(abstracts):
            raise AssertionError(
                f"{bank_dir}: embedding rows do not match jsonl lines; refusing to "
                f"load a misaligned bank")
        if sess_emb.size and sess_emb.shape[1] != self.embedder.dim:
            raise AssertionError(
                f"{bank_dir}: bank dim {sess_emb.shape[1]} != embedder dim "
                f"{self.embedder.dim}")

        self._cached_session_vec = {
            s["session_identifier"]: sess_emb[i] for i, s in enumerate(sessions)
        }
        self._cached_abstract_vec = {}
        for i, a in enumerate(abstracts):
            self._cached_abstract_vec[(a["chunk_id"], a.get("abstract_id", i))] = abs_emb[i]
        logger.info("Attached embedding cache from %s (%d sessions, %d abstracts)",
                    bank_dir, len(sessions), len(abstracts))

    def _session_vector(self, sid: str, text: str) -> np.ndarray:
        cache = getattr(self, "_cached_session_vec", None)
        if cache and sid in cache:
            return cache[sid]
        from eval.config import EMBED_HEAD_CHARS, EMBED_TAIL_CHARS
        return self.embedder.encode(
            [schema.head_tail(text, EMBED_HEAD_CHARS, EMBED_TAIL_CHARS)])[0]

    def _abstract_vector(self, sid: str, abstract_id: Any, text: str) -> np.ndarray:
        cache = getattr(self, "_cached_abstract_vec", None)
        if cache:
            hit = cache.get((sid, abstract_id))
            if hit is not None:
                return hit
        return self.embedder.encode([text])[0]

    # ── Admission ──

    def add_session(self, session: dict):
        """Admit one session. Everything already admitted becomes retrievable."""
        sid = session.get("session_identifier") or ""
        text = schema.session_text(session)
        if not text.strip():
            return

        extracted = session.get("extracted_memory", []) or []
        idx = len(self._sessions)

        self._sessions.append({
            "chunk_id": sid,
            "session_identifier": sid,
            "session_uuid": session.get("session_uuid", ""),
            "current_time": session.get("current_time", ""),
            "content": text,
            "num_turns": len(session.get("dialogue_turns", []) or []),
        })
        self._sess_emb.append(self._session_vector(sid, text))
        self._topics.append(schema.parse_topic(sid))
        self._has_abs.append(bool(extracted))
        self._sid_to_idx[sid] = idx

        for j, mem in enumerate(extracted):
            content = mem.get("content", "")
            if not content:
                continue
            mem_type = mem.get("type", "General")
            rec = {
                "chunk_id": sid,
                "abstract_type": mem_type,
                "content": f"[{mem_type}] {content}",
                "raw_content": content,
            }
            self._abstracts.append(rec)
            self._abs_emb.append(self._abstract_vector(sid, j, rec["content"]))
            self._abs_owner.append(idx)
            self._abs_by_sid.setdefault(sid, []).append(rec)

        # Vectorized tag arrays are rebuilt lazily on next retrieve.
        self._topic_arr = None
        self._has_abs_arr = None

    @property
    def n_sessions(self) -> int:
        return len(self._sessions)

    # ── Stage 1 ──

    def _stage1_bonus(self, query_topic: Optional[str]) -> np.ndarray:
        """Additive per-session bonus. Never zeroes a candidate out."""
        n = len(self._sessions)
        bonus = np.zeros(n, dtype=np.float32)
        if not self.enable_stage1 or n == 0:
            return bonus

        if self._has_abs_arr is None:
            self._has_abs_arr = np.array(self._has_abs, dtype=bool)
            self._topic_arr = np.array(
                [t if t is not None else "" for t in self._topics], dtype=object)

        # Content-side: a session that yielded no structured memory is never gold
        # (0 of 2319 references). This is what actually removes the 48% filler.
        bonus += STAGE1_ABSTRACT_WEIGHT * self._has_abs_arr.astype(np.float32)

        # Metadata-side: 80% of gold shares the query's topic. Ablatable.
        if query_topic and self.topic_source == "identifier":
            bonus += STAGE1_TOPIC_WEIGHT * (self._topic_arr == query_topic).astype(np.float32)

        # Smooth decay over session distance. Weight 0 by default: the median
        # query-to-gold gap is 10-22 sessions, so recency is a weak signal here.
        if STAGE1_RECENCY_WEIGHT:
            dist = (n - 1) - np.arange(n, dtype=np.float32)
            bonus += STAGE1_RECENCY_WEIGHT * np.exp(-dist / max(STAGE1_RECENCY_SCALE, 1e-6))

        return bonus

    # ── Stage 2 ──

    def _stage2_pool(self, query: str, query_topic: Optional[str]) -> List[Dict]:
        n = len(self._sessions)
        if n == 0:
            return []

        q = self.embedder.encode([query])[0]
        concrete = self._sess_emb.view @ q

        # Per-session max over its abstracts, without a Python loop over pairs.
        abstract = np.zeros(n, dtype=np.float32)
        if self.enable_dual_pool and len(self._abs_emb):
            a_sims = self._abs_emb.view @ q
            np.maximum.at(abstract, np.asarray(self._abs_owner, dtype=np.intp), a_sims)

        abstract_w = ABSTRACT_W if self.enable_dual_pool else 0.0
        score = (CONCRETE_W * concrete + abstract_w * abstract
                 + self._stage1_bonus(query_topic))

        take = min(self.pool_size, self.top_n_sessions, n)
        top = np.argpartition(-score, take - 1)[:take] if take < n else np.arange(n)
        top = top[np.argsort(-score[top])]

        pool = []
        for rank, i in enumerate(top):
            rec = self._sessions[i]
            pool.append({
                "chunk_id": rec["chunk_id"],
                "content": rec["content"],
                "current_time": rec.get("current_time", ""),
                "concrete_sim": float(concrete[i]),
                "abstract_sim": float(abstract[i]),
                "stage1_bonus": float(score[i] - CONCRETE_W * concrete[i]
                                      - ABSTRACT_W * abstract[i]),
                "stage2_score": float(score[i]),
                "stage2_rank": rank + 1,
                "same_topic": bool(query_topic and self._topics[i] == query_topic),
                "has_abstract": bool(self._has_abs[i]),
            })
        return pool

    # ── Stage 3 ──

    def _candidate_excerpt(self, cand: Dict) -> str:
        """Abstracts first: a session's raw opening is mostly pleasantries, while
        extracted_memory is the structured payload."""
        abstracts = self._abs_by_sid.get(cand["chunk_id"], [])
        if abstracts:
            joined = " ".join(a["content"] for a in abstracts)
            return joined[:STAGE3_ABSTRACT_CHARS].replace("\n", " ")
        return cand["content"][:STAGE3_RAW_FALLBACK_CHARS].replace("\n", " ")

    def _stage3_prompt(self, query: str, candidates: List[Dict]) -> str:
        lines = []
        for i, cand in enumerate(candidates, 1):
            lines.append(f"[{i}] {self._candidate_excerpt(cand)}")
        return (
            "You are ranking past conversation sessions by how useful each one is "
            "for answering the user's current question.\n\n"
            f"Question: {query[:1000]}\n\n"
            "Candidate sessions:\n" + "\n\n".join(lines) + "\n\n"
            "Score each candidate 0.0-1.0 for how directly it supports answering "
            "the question.\n"
            "Rubric: 1.0=contains the specific fact/preference/commitment needed, "
            "0.7=same project and topically on point, 0.4=loosely related, "
            "0.2=weak, 0.0=irrelevant.\n"
            'Return ONLY a JSON array: [{"idx": 1, "fit": 0.8}, ...] with one '
            "entry per candidate.\n"
        )

    def _cache_key(self, query: str, candidates: List[Dict]) -> str:
        sig = "|".join(c["chunk_id"] for c in candidates)
        raw = f"{PROMPT_VERSION}\x00{self.stage3_model}\x00{query}\x00{sig}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _append_cache(self, key: str, samples: List[List[float]]):
        self._cache[key] = samples
        if not self._cache_path:
            return
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        with open(self._cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "samples": samples}) + "\n")

    def _one_sample(self, prompt: str, n_candidates: int) -> tuple:
        """One judge draw -> (fits, n_errors). Returns -1.0 for candidates the
        judge did not score, which stays distinct from a scored 0.0 — otherwise
        a parse failure reads as a rejection."""
        fits = [None] * n_candidates
        errors = 0
        try:
            res = self.llm_client.chat.completions.create(
                model=self.stage3_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=STAGE3_TEMPERATURE,
                timeout=API_TIMEOUT,
            )
            text = (res.choices[0].message.content or "").strip()
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                for item in json.loads(match.group(0)):
                    idx = int(item.get("idx", 0)) - 1
                    if 0 <= idx < n_candidates:
                        fits[idx] = float(item.get("fit", 0.0))
        except Exception as exc:
            errors = 1
            logger.warning("Stage 3 sample failed: %s: %s", type(exc).__name__, exc)
        return [-1.0 if f is None else f for f in fits], errors

    def _stage3_samples(self, query: str, candidates: List[Dict]) -> List[List[float]]:
        """N independent fit vectors. Cached on (prompt, model, query, candidates)
        and deliberately NOT on tau/vote_threshold, so those can be retuned for
        free.

        The N draws are independent by construction, so they run concurrently.
        They were serial originally, which at a measured ~6s per call made the
        full sweep several hours of pure waiting. Sampling concurrency does not
        touch the streaming contract: the retrieval loop over sessions stays
        strictly sequential, only the repeated judging of one already-fixed
        candidate list overlaps.
        """
        key = self._cache_key(query, candidates)
        if key in self._cache:
            self.stats["stage3_cache_hits"] += 1
            return self._cache[key]

        if self.llm_client is None:
            return []

        prompt = self._stage3_prompt(query, candidates)
        n = len(candidates)
        workers = min(self.stage3_samples, max(1, self.sample_workers))

        if workers > 1 and self.stage3_samples > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:
                drawn = list(pool.map(lambda _: self._one_sample(prompt, n),
                                      range(self.stage3_samples)))
        else:
            drawn = [self._one_sample(prompt, n) for _ in range(self.stage3_samples)]

        # Counters updated here, on one thread, rather than inside the workers.
        self.stats["stage3_calls"] += self.stage3_samples
        self.stats["stage3_errors"] += sum(e for _f, e in drawn)
        samples = [f for f, _e in drawn]

        self._append_cache(key, samples)
        return samples

    def _stage3_rerank(self, query: str, pool: List[Dict]) -> List[Dict]:
        head = pool[: self.rerank_candidates]
        tail = pool[self.rerank_candidates:]

        samples = self._stage3_samples(query, head)
        if not samples:
            # No judge available. In gate mode that must not silently become
            # "everything passed"; degrade to the honest empty answer.
            if self.gate_mode == "gate":
                return []
            return pool

        votes = [0] * len(head)
        fit_sum = [0.0] * len(head)
        fit_n = [0] * len(head)
        for sample in samples:
            for i, fit in enumerate(sample):
                if fit is None or fit < 0:
                    continue
                fit_sum[i] += fit
                fit_n[i] += 1
                if fit >= self.tau:
                    votes[i] += 1

        passed, failed = [], []
        for i, cand in enumerate(head):
            cand = dict(cand)
            cand["fit_votes"] = votes[i]
            cand["fit_mean"] = round(fit_sum[i] / fit_n[i], 4) if fit_n[i] else None
            cand["stage3_passed"] = votes[i] >= self.vote_threshold
            (passed if cand["stage3_passed"] else failed).append(cand)

        passed.sort(key=lambda c: (c["fit_votes"], c["fit_mean"] or 0.0,
                                   c["stage2_score"]), reverse=True)

        if self.gate_mode == "gate":
            # Upstream semantics: refused candidates are dropped outright, and an
            # empty survivor set means the query gets no memory at all.
            return passed

        failed.sort(key=lambda c: c["stage2_score"], reverse=True)
        return passed + failed + tail

    # ── Public API ──

    def retrieve(self, query: str, query_topic: Optional[str] = None,
                 keywords: Optional[str] = None) -> List[Dict]:
        """Rank admitted sessions for one query.

        Returns items in the format the official metric scripts expect:
        ``res_type='chunk'`` with ``chunk_id`` a session_identifier.
        """
        self.stats["queries"] += 1
        search_text = f"{query}\n{keywords}" if keywords else query

        pool = self._stage2_pool(search_text, query_topic)
        if not pool:
            return []

        if self.enable_stage3:
            ranked = self._stage3_rerank(search_text, pool)
        else:
            ranked = pool

        if not ranked:
            self.stats["gated_empty"] += 1
            return []

        out = []
        for rank, item in enumerate(ranked[: self.retrieve_k], 1):
            out.append({
                "res_type": "chunk",
                "chunk_id": item["chunk_id"],
                "content": item["content"],
                "score": round(float(item["stage2_score"]), 4),
                "rank": rank,
                "stage2_rank": item.get("stage2_rank"),
                # How much of the score came from tags rather than similarity.
                # Exposed so an ablation can be verified rather than inferred.
                "stage1_bonus": round(float(item.get("stage1_bonus", 0.0)), 4),
                # Kept so a single-pool control can be reconstructed offline
                # instead of re-running retrieval.
                "concrete_sim": round(float(item.get("concrete_sim", 0.0)), 4),
                "abstract_sim": round(float(item.get("abstract_sim", 0.0)), 4),
                "fit_votes": item.get("fit_votes"),
                "fit_mean": item.get("fit_mean"),
                "stage3_passed": item.get("stage3_passed"),
                "same_topic": item.get("same_topic"),
                "has_abstract": item.get("has_abstract"),
            })
        return out


class SimpleEmbeddingRetriever(ThreeStageRetriever):
    """Cosine-only control: no Stage-1 bonus, no Stage-3 judge.

    Shares the admission path with the three-stage arms so that every arm sees
    exactly the same sessions in exactly the same order — the comparison is only
    meaningful if the sole difference is the ranking function.
    """

    def __init__(self, embedder=None, retrieve_k: int = RETRIEVE_K, **kwargs):
        kwargs.pop("enable_stage1", None)
        kwargs.pop("enable_stage3", None)
        super().__init__(embedder=embedder, retrieve_k=retrieve_k,
                         enable_stage1=False, enable_stage3=False, **kwargs)


class NoMemoryRetriever:
    """Baseline: never retrieves. Still consumes sessions so the loop is uniform."""

    def __init__(self, **_kwargs):
        self.stats = {"queries": 0}
        self.n_sessions = 0

    def reset(self):
        self.n_sessions = 0

    def attach_bank(self, bank_dir: str):
        pass

    def add_session(self, session: dict):
        self.n_sessions += 1

    def retrieve(self, query: str, query_topic=None, keywords=None) -> List[Dict]:
        self.stats["queries"] += 1
        return []
