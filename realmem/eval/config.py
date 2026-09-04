"""Global configuration for the RealMemBench adaptive-memory harness.

Every knob here has an environment override so nothing below needs this file
edited. Naming follows ``REALMEM_<NAME>``.

The vendored benchmark scripts (``compute_auto_metrics_for_realmem.py``,
``compute_llm_metrics_for_realmem.py``, ``run_generation.py``) are treated as
strictly read-only: they are the reference implementation we check ourselves
against, so changing them would destroy the only independent oracle we have.
"""

import os

# Bank layout version. Bumped when a bank gains fields the retriever relies on.
# Format 1 (pre-2026-08-26) lacked the Stage-1 tags and, more dangerously, the
# embedding_backend provenance — without which a hash-encoded query could be
# matched against sentence-transformers vectors with no error at all. A bank
# without this key is refused rather than half-trusted.
BANK_FORMAT = 2

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_BANK_DIR = os.path.join(PROJECT_ROOT, "memory_banks")
RETRIEVAL_RESULT_DIR = os.path.join(EVAL_DIR, "retrieval_result")

# ── Models ──
EMBEDDING_MODEL = os.environ.get("REALMEM_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
STAGE3_MODEL = os.environ.get("REALMEM_STAGE3_MODEL", "gemini-3.5-flash")
GENERATION_MODEL = os.environ.get("REALMEM_GENERATION_MODEL", "gemini-3.5-flash")
JUDGE_MODEL = os.environ.get("REALMEM_JUDGE_MODEL", "gemini-3.5-flash")
KEYWORD_MODEL = os.environ.get("REALMEM_KEYWORD_MODEL", "gemini-3.5-flash")

# ── API ──
# No default key on purpose: an empty value fails loudly on the first request
# instead of silently authenticating as whoever committed one.
#   export REALMEM_API_KEY=sk-...      (bash)
#   $env:REALMEM_API_KEY = 'sk-...'    (PowerShell)
API_KEY = os.environ.get("REALMEM_API_KEY", os.environ.get("BBEH_API_KEY", ""))
BASE_URL = os.environ.get("REALMEM_BASE_URL", os.environ.get("BBEH_BASE_URL",
    "https://api.openai.com/v1"))

API_TIMEOUT = int(os.environ.get("REALMEM_API_TIMEOUT", "180"))
API_MAX_RETRIES = int(os.environ.get("REALMEM_API_MAX_RETRIES", "3"))

# ═════════════════════════════════════════════════════════════════════
#  Retrieval depth
# ═════════════════════════════════════════════════════════════════════
#
# The metric suite reports Recall@5/@10/@20 and NDCG@5/@10/@20, so a run that
# returns fewer than 20 ranked items has a *structural* ceiling on the @20
# columns. The previous configuration had POOL_SIZE=15 < RETRIEVE_K=20 and
# measured "recall@20" over lists that were never longer than 15 — the @20
# numbers were really @15 or shallower.
#
# POOL_SIZE and RERANK_CANDIDATES_M are deliberately decoupled: recall depth is
# cheap (cosine over a few hundred sessions), but every extra candidate handed
# to Stage 3 costs tokens on every sample of every query. So recall wide, rerank
# narrow, and let the un-reranked tail keep its Stage-2 order.

RETRIEVE_K = int(os.environ.get("REALMEM_RETRIEVE_K", "20"))
POOL_SIZE = int(os.environ.get("REALMEM_POOL_SIZE", "40"))
TOP_N_SESSIONS = int(os.environ.get("REALMEM_TOP_N_SESSIONS", "40"))

# Largest k any metric asks for. Used to assert the depth invariant.
MAX_METRIC_K = 20

# ═════════════════════════════════════════════════════════════════════
#  Stage 1 — soft tag weighting
# ═════════════════════════════════════════════════════════════════════
#
# PuzzleWorld weighted modality/skill overlap; BBEH weighted task/pattern_type.
# The RealMem analogues, measured over all 10 personas / 2319 gold references:
#
#   has_abstract  — 48% of sessions ("Enhanced:S1xxxx") are filler with an EMPTY
#                   extracted_memory list, and gold points at such a session
#                   0 times out of 2319. This is a CONTENT-side signal (did this
#                   session yield any structured memory at all?), not a naming
#                   artefact, which is why it is the default and carries the
#                   largest weight. Measured agreement with the identifier
#                   prefix is exact: 985 filler sessions, 0 of which have an
#                   abstract; 6 project sessions without one, none ever gold.
#
#   topic         — 80% of gold references sit in the same topic as the query
#                   (per-persona 0.65-0.95). Derived from the session_identifier
#                   prefix, i.e. benchmark metadata. Kept behind TOPIC_SOURCE so
#                   the contribution can be ablated away; see HARNESS.md.
#
#   recency       — the old config gave a bonus inside a 7-DAY window, but the
#                   median query-to-gold distance is 10-22 sessions (max 253),
#                   so that window systematically rewarded the wrong sessions.
#                   Replaced by a smooth decay over session distance, weight 0
#                   by default because the measured signal is weak.
#
# All three are ADDITIVE and never filter. That is the point of Stage 1: a
# cross-topic precedent stays in the pool and it is Stage 3's job to rule on it.

STAGE1_ABSTRACT_WEIGHT = float(os.environ.get("REALMEM_STAGE1_ABSTRACT_WEIGHT", "0.15"))
STAGE1_TOPIC_WEIGHT = float(os.environ.get("REALMEM_STAGE1_TOPIC_WEIGHT", "0.10"))
STAGE1_RECENCY_WEIGHT = float(os.environ.get("REALMEM_STAGE1_RECENCY_WEIGHT", "0.0"))
STAGE1_RECENCY_SCALE = float(os.environ.get("REALMEM_STAGE1_RECENCY_SCALE", "30.0"))

# 'identifier' parses the topic out of session_identifier; 'none' disables the
# topic bonus entirely (ablation arm).
TOPIC_SOURCE = os.environ.get("REALMEM_TOPIC_SOURCE", "identifier")

# The observed memory types are Dynamic / Static / Schedule. The previous config
# keyed on ("Dynamic", "Preference", "Fact") — the latter two do not occur in
# the data at all, and Dynamic covers 80%+ of items, so the bonus was a near
# constant offset that ranked nothing. Left here as an explicit, empty default.
MEMORY_TYPE_BONUSES = {}

# ═════════════════════════════════════════════════════════════════════
#  Stage 2 — dual-pool content recall
# ═════════════════════════════════════════════════════════════════════

CONCRETE_W = float(os.environ.get("REALMEM_CONCRETE_W", "0.5"))
ABSTRACT_W = float(os.environ.get("REALMEM_ABSTRACT_W", "0.3"))

# ═════════════════════════════════════════════════════════════════════
#  Stage 3 — LLM approach-fit rerank + vote gate
# ═════════════════════════════════════════════════════════════════════
#
# gate_mode decides what happens to a candidate the judge refuses:
#
#   'rerank' — it is demoted below every candidate that passed, but kept. The
#              returned list stays RETRIEVE_K long, so Recall@k stays
#              comparable with the embedding baseline.
#   'gate'   — BBEH/PuzzleWorld semantics: candidates that fail the vote are
#              dropped, and if none pass the query gets an EMPTY list and
#              degrades to no-memory. Faithful, but on a Recall@k benchmark an
#              empty list scores zero, which is why it is a separate arm rather
#              than the default.

GATE_MODE = os.environ.get("REALMEM_GATE_MODE", "rerank")

RERANK_CANDIDATES_M = int(os.environ.get("REALMEM_RERANK_CANDIDATES_M", "25"))
TAU = float(os.environ.get("REALMEM_TAU", "0.7"))
VOTE_THRESHOLD = int(os.environ.get("REALMEM_VOTE_THRESHOLD", "2"))
STAGE3_SAMPLES = int(os.environ.get("REALMEM_STAGE3_SAMPLES", "3"))
STAGE3_TEMPERATURE = float(os.environ.get("REALMEM_STAGE3_TEMPERATURE", "1.0"))

# The N draws for one query are independent, so they run concurrently. Measured
# judge latency is ~6s per call at 25 candidates; serially that made the full
# sweep hours of pure waiting. This does NOT relax the streaming contract — the
# loop over sessions stays sequential, only the repeated judging of one fixed
# candidate list overlaps. Set to 1 to force the old serial behaviour.
STAGE3_SAMPLE_WORKERS = int(os.environ.get("REALMEM_STAGE3_SAMPLE_WORKERS", "3"))

# Per-candidate excerpt handed to the judge. RealMem sessions run to ~4k chars
# and open with pleasantries, so the first 600 characters of raw dialogue are
# mostly greeting. The abstracts (extracted_memory) are denser, so they lead and
# raw text is only a fallback for sessions that have none.
STAGE3_ABSTRACT_CHARS = int(os.environ.get("REALMEM_STAGE3_ABSTRACT_CHARS", "400"))
STAGE3_RAW_FALLBACK_CHARS = int(os.environ.get("REALMEM_STAGE3_RAW_FALLBACK_CHARS", "300"))

RERANK_CACHE_NAME = "rerank_cache.jsonl"

# ═════════════════════════════════════════════════════════════════════
#  Query keyword expansion (README pipeline step)
# ═════════════════════════════════════════════════════════════════════
# The published pipeline calls generate_query_llm(question) before retrieval.
# Off by default so the headline arms cost nothing extra; flip to compare.

USE_KEYWORDS = os.environ.get("REALMEM_USE_KEYWORDS", "0") == "1"
KEYWORD_MAX_TOKENS = int(os.environ.get("REALMEM_KEYWORD_MAX_TOKENS", "64"))

# ═════════════════════════════════════════════════════════════════════
#  Embedding
# ═════════════════════════════════════════════════════════════════════
# MiniLM truncates at 256 word-pieces, so embedding a whole session wastes the
# window on the opening turns. Head+tail keeps the framing and the payload.

EMBED_HEAD_CHARS = int(os.environ.get("REALMEM_EMBED_HEAD_CHARS", "900"))
EMBED_TAIL_CHARS = int(os.environ.get("REALMEM_EMBED_TAIL_CHARS", "900"))

# ═════════════════════════════════════════════════════════════════════
#  Generation / judging
# ═════════════════════════════════════════════════════════════════════

GEN_TOP_K = int(os.environ.get("REALMEM_GEN_TOP_K", "5"))
GEN_TEMPERATURE = float(os.environ.get("REALMEM_GEN_TEMPERATURE", "0.7"))
JUDGE_TEMPERATURE = float(os.environ.get("REALMEM_JUDGE_TEMPERATURE", "0.0"))
MAX_WORKERS = int(os.environ.get("REALMEM_MAX_WORKERS", "8"))

# Content stored per ranked item on disk. Full text is kept once per session in
# the bank, so repeating it 20x per query per arm only inflates the artefacts
# (previously 10-27 MB per persona per arm).
RESULT_CONTENT_CHARS = int(os.environ.get("REALMEM_RESULT_CONTENT_CHARS", "1200"))

# ═════════════════════════════════════════════════════════════════════
#  Arms
# ═════════════════════════════════════════════════════════════════════

DEFAULT_ARMS = ("no_memory", "simple_embedding", "three_stage_rerank", "three_stage_gated")

# ═════════════════════════════════════════════════════════════════════
#  mem0 external baseline (opt-in via --arms ...,mem0)
# ═════════════════════════════════════════════════════════════════════
#
# 'huggingface' runs sentence-transformers locally. It is the default because
# an LLM gateway need not proxy embeddings at all — some gateways answer
# text-embedding-3-small with HTTP 503 "no available channel", which surfaces
# inside mem0 as a bare "Connection error" after retries. Running the encoder
# locally also matches our own arms, isolating the memory architecture from
# embedding quality. Set to 'openai' to follow the paper's text-embedding-3-small,
# but verify the gateway serves /embeddings first.
MEM0_EMBED_PROVIDER = os.environ.get("REALMEM_MEM0_EMBED_PROVIDER", "huggingface")

# mem0's memory-extraction LLM, deliberately SEPARATE from GENERATION_MODEL.
# Sharing one variable would mean that raising mem0's extraction quality also
# silently changes the answer-generation model for every other arm, making the
# QA numbers incomparable — and the person who set it would have no reason to
# suspect that. Defaults to GENERATION_MODEL so behaviour is unchanged unless
# explicitly overridden:
#     REALMEM_MEM0_LLM_MODEL=gemini-3.1-pro-preview python -m eval.run_eval ...
MEM0_LLM_MODEL = os.environ.get("REALMEM_MEM0_LLM_MODEL", GENERATION_MODEL)

# mem0's default Qdrant path (/tmp/qdrant) and collection ('mem0') are global.
# Two processes collide; two personas contaminate each other. One directory and
# one collection per persona.
MEM0_STORAGE_ROOT = os.environ.get(
    "REALMEM_MEM0_STORAGE", os.path.join(PROJECT_ROOT, "mem0_store"))


def persona_dataset_path(name: str) -> str:
    return os.path.join(DATASET_DIR, f"{name}_dialogues_256k.json")


def persona_memory_bank(name: str) -> str:
    return os.path.join(MEMORY_BANK_DIR, name)


def persona_retrieval_dir(name: str) -> str:
    return os.path.join(RETRIEVAL_RESULT_DIR, name)


def list_personas(dataset_dir: str = None) -> list:
    """Every persona with a dialogue file, sorted."""
    d = dataset_dir or DATASET_DIR
    return sorted(
        f[: -len("_dialogues_256k.json")]
        for f in os.listdir(d)
        if f.endswith("_dialogues_256k.json")
    )
