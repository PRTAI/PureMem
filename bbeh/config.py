"""
bbeh/config.py — configuration for the BBEH adaptive-memory harness.

Self-contained on purpose: this module does NOT import the repo-root
``config.py``, because ``read_config()`` there mkdir's the PuzzleWorld data
tree as an import side effect. The BBEH harness must be runnable in a tree
where ``PuzzleWorld/`` does not exist (it currently doesn't).

Every path below lives under ``bbeh/``. The vendored benchmark
(``bbeh-main/``) is treated as strictly read-only.

Environment overrides:
    BBEH_API_KEY (required), BBEH_BASE_URL, BBEH_API_PROTOCOL,
    BBEH_API_AUTH_SCHEME, BBEH_CANDIDATE_URLS,
    BBEH_STUDENT_MODEL, BBEH_TEACHER_MODEL, BBEH_JUDGE_MODEL
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

# ═════════════════════════════════════════════════════════════════════
#  API
# ═════════════════════════════════════════════════════════════════════

API = {
    # Any OpenAI-compatible gateway.
    'base_url': os.environ.get('BBEH_BASE_URL', 'https://api.openai.com/v1'),
    # No default on purpose: an empty key fails loudly on the first request
    # instead of silently authenticating as whoever committed one.
    #   export BBEH_API_KEY=sk-...       (bash)
    #   $env:BBEH_API_KEY = 'sk-...'     (PowerShell)
    'api_key': os.environ.get('BBEH_API_KEY', ''),
    'api_protocol': os.environ.get('BBEH_API_PROTOCOL', 'chat'),
    # How the credential is put on the wire:
    #   'bearer'    -> Authorization: Bearer <key>   (OpenAI convention)
    #   'x-api-key' -> x-api-key: <key>              (Anthropic convention)
    # Gateways differ, and one that expects x-api-key answers a Bearer request
    # with a 403 that reads exactly like a bad key — which is why this is a
    # setting rather than an assumption baked into the client.
    #
    # 'bearer' is the right default for an OpenAI-compatible gateway, because the
    # scheme travels with the path: /v1/chat/completions authenticates with
    # Authorization: Bearer, while x-api-key belongs to Anthropic's /v1/messages.
    # Changing one without the other cannot work.
    #
    # Measured on a previous gateway rather than assumed: bearer returned 403
    # "access denied" — the credential was read and
    # refused — while x-api-key returned 401 "no token provided", meaning that
    # header was never looked at. Recorded because the 401/403 pair is the only
    # thing that separates a wrong scheme from a wrong key, and the reasoning is
    # about the route, not that particular host.
    'api_auth_scheme': os.environ.get('BBEH_API_AUTH_SCHEME', 'bearer'),
}

# ─── Model roles ─────────────────────────────────────────────────────
# STUDENT  — the "base model" in both claims. It is what we probe for
#            difficulty (ZPD), what we evaluate with no_memory, and what we
#            evaluate with memory. Claim 2 is about this model.
# TEACHER  — generates the chain-of-thought that becomes memory. BBEH ships
#            no reference solutions, so memory content has to come from
#            somewhere; every CoT is verified against the gold target before
#            being admitted.
#            The default is the SAME checkpoint as the student. That is a
#            deliberate choice, not an oversight: it removes stronger-model
#            distillation as an explanation for any gain, so whatever remains
#            is attributable to curation and gating. The cost is that the
#            teacher cannot solve every source problem, which is why trace
#            verification rejects a large share of attempts. Point
#            BBEH_TEACHER_MODEL at a stronger model to study the other regime,
#            and say which one you ran.
# JUDGE    — the Stage-3 approach-fit reranker. Cheap on purpose.
STUDENT_MODEL = os.environ.get('BBEH_STUDENT_MODEL', 'gemini-3.5-flash')
TEACHER_MODEL = os.environ.get('BBEH_TEACHER_MODEL', 'gemini-3.5-flash')
JUDGE_MODEL = os.environ.get('BBEH_JUDGE_MODEL', 'gemini-3.5-flash')

# Request shaping. The empty-body / timeout lesson from the PuzzleWorld runs is
# baked into api_client.py; these are its knobs.
API_TIMEOUT = 300           # BBEH inputs run to 32k chars; long reasoning needs room
API_MAX_RETRIES = 4
API_RETRY_BASE_DELAY = 4.0  # exponential: 4, 8, 16, 32 s

# Token budgets. BBEH tasks genuinely need long reasoning; starving the solver
# truncates the "The final answer is:" line and scores a false 0.
SOLVE_MAX_TOKENS = 16384
TEACHER_MAX_TOKENS = 6144
ABSTRACT_MAX_TOKENS = 4096
JUDGE_MAX_TOKENS = 512

# Sampling temperatures.
SOLVE_TEMPERATURE = 0.0     # deterministic-ish for the graded runs
PROBE_TEMPERATURE = 1.0     # the probe NEEDS variance to estimate a pass rate
TEACHER_TEMPERATURE = 1.0   # rejection sampling needs variance across attempts
ABSTRACT_TEMPERATURE = 0.0

# ═════════════════════════════════════════════════════════════════════
#  Paths
# ═════════════════════════════════════════════════════════════════════

# Vendored benchmark — READ ONLY. Never write inside this tree.
BBEH_VENDOR_DIR = os.path.join(REPO_ROOT, 'bbeh-main', 'bbeh')
BBEH_TASKS_DIR = os.path.join(BBEH_VENDOR_DIR, 'benchmark_tasks')

SPLITS_DIR = os.path.join(BASE_DIR, 'splits')
TRAIN_JSONL = os.path.join(SPLITS_DIR, 'train.jsonl')
TEST_JSONL = os.path.join(SPLITS_DIR, 'test.jsonl')
SPLIT_META_JSON = os.path.join(SPLITS_DIR, 'split_meta.json')

# Intermediate artifacts (expensive to produce, cached aggressively).
WORK_DIR = os.path.join(BASE_DIR, 'work')

MEMORY_BANKS_DIR = os.path.join(BASE_DIR, 'memory_banks')
MEMORY_VERSIONS_DIR = os.path.join(MEMORY_BANKS_DIR, 'versions')

RUNS_DIR = os.path.join(BASE_DIR, 'runs')

# Filenames inside a memory version directory.
MEMORY_JSONL_NAME = 'memory.jsonl'
EMBEDDINGS_NPY_NAME = 'embeddings.npy'
QUESTION_EMBEDDINGS_NPY_NAME = 'question_embeddings.npy'
ABSTRACT_MEMORY_JSONL_NAME = 'abstract_memory.jsonl'
ABSTRACT_EMBEDDINGS_NPY_NAME = 'abstract_embeddings.npy'
# Row i of question_embeddings.npy belongs to demo i; demos.jsonl records which
# train item that is. Without it, source_idx is an index into nothing.
DEMOS_JSONL_NAME = 'demos.jsonl'
META_JSON_NAME = 'meta.json'
RERANK_CACHE_NAME = 'rerank_cache.jsonl'

VERSION_FILES = (
    MEMORY_JSONL_NAME,
    EMBEDDINGS_NPY_NAME,
    QUESTION_EMBEDDINGS_NPY_NAME,
    ABSTRACT_MEMORY_JSONL_NAME,
    ABSTRACT_EMBEDDINGS_NPY_NAME,
    DEMOS_JSONL_NAME,
    META_JSON_NAME,
)


def probe_path(model: str) -> str:
    """Per-item difficulty file for a given student model (aggregated)."""
    return os.path.join(WORK_DIR, f'difficulty_{_slug(model)}.jsonl')


def probe_samples_path(model: str) -> str:
    """Raw per-sample probe log — the actual cache.

    Kept separate from :func:`probe_path` because the cache unit is one
    *attempt*, not one item: that makes raising k incremental (k=5 -> k=10
    reuses the first five attempts) and makes a crash mid-item cost one
    attempt instead of five.
    """
    return os.path.join(WORK_DIR, f'probe_samples_{_slug(model)}.jsonl')


def cot_bank_path(model: str) -> str:
    """Verified teacher chain-of-thought bank for a given teacher model."""
    return os.path.join(WORK_DIR, f'cot_bank_{_slug(model)}.jsonl')


def version_dir(version_id: str) -> str:
    return os.path.join(MEMORY_VERSIONS_DIR, version_id)


def run_dir(arm_label: str, model: str) -> str:
    return os.path.join(RUNS_DIR, f'{arm_label}_{_slug(model)}')


def _slug(name: str) -> str:
    """Filesystem-safe model name."""
    return ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in str(name))


def ensure_dirs():
    """Create the writable directories. Never touches the vendored tree."""
    for path in (SPLITS_DIR, WORK_DIR, MEMORY_BANKS_DIR,
                 MEMORY_VERSIONS_DIR, RUNS_DIR):
        os.makedirs(path, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
#  Split
# ═════════════════════════════════════════════════════════════════════

SPLIT_SEED = 42
TRAIN_PER_TASK = 100
TEST_PER_TASK = 100

# ═════════════════════════════════════════════════════════════════════
#  Embedding
# ═════════════════════════════════════════════════════════════════════

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIM = 384

# BBEH inputs run up to 32k chars but MiniLM truncates at 256 word-pieces.
# Embedding a 32k-char blob wastes the signal on the preamble, so we embed a
# head+tail window: the head carries the task framing, the tail carries the
# actual question/instruction, which is where the discriminative content is.
EMBED_HEAD_CHARS = 900
EMBED_TAIL_CHARS = 900

# ═════════════════════════════════════════════════════════════════════
#  Three-stage retrieval
# ═════════════════════════════════════════════════════════════════════

TOP_N_DEMOS = 8       # Layer 1 recall width (demo-level)
TOP_K = 2             # max precedents injected into the solver prompt
TOP_N_ABSTRACTS = 2   # Layer 0 abstract-pattern hits

# ─── Stage 1: soft tag weighting ─────────────────────────────────────
# PuzzleWorld weighted modality/skills overlap. BBEH has neither, so the
# native analogues are:
#   task         — the 23 BBEH task families (coarse but very informative)
#   pattern_type — the reasoning-mechanism label our abstractor assigns
# Kept SOFT (additive) rather than a hard within-task filter for two reasons:
#   1. a hard filter would silently turn this into a per-task few-shot
#      retriever, which is a much weaker claim;
#   2. cross-task mechanism transfer is the interesting result, and only the
#      Stage-3 judge is qualified to rule on it.
STAGE1_TASK_WEIGHT = 0.15      # bonus when candidate comes from the same task
STAGE1_PATTERN_WEIGHT = 0.10   # bonus when pattern_type matches the query's

# ─── Stage 2: content recall ─────────────────────────────────────────
RERANK_CANDIDATES_M = 5        # shortlist length handed to the judge

# ─── Stage 3: LLM approach-fit rerank + majority-vote gate ───────────
RERANK_FIT_TAU = 0.7
RERANK_SAMPLES_N = 5
RERANK_VOTE_THRESHOLD = 2
RERANK_TEMPERATURE = 1.0

# ═════════════════════════════════════════════════════════════════════
#  ZPD (zone of proximal development)
# ═════════════════════════════════════════════════════════════════════

# Difficulty probe: k stochastic student attempts per train item.
PROBE_K = 5

# An item is in the ZPD when the student solves it *sometimes* — neither
# mastered (nothing left to teach) nor out of reach (a teacher CoT it cannot
# assimilate). Band is on pass_rate = n_correct / k, inclusive.
ZPD_LOW = 0.2
ZPD_HIGH = 0.8

# Strict-interior variant used by `--selector zpd --zpd-strict`: 0 < p < 1.
