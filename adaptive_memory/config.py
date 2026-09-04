"""
Adaptive Memory configuration.

All paths, API settings, and hyperparameters live here.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── API ─────────────────────────────────────────────────────────────
# Read defaults from the project's root config if available
# (so we share the same base_url / api_key)
import sys
sys.path.insert(0, os.path.join(BASE_DIR, '..'))
from config import read_config as _read_root_config
_root = _read_root_config()

API = {
    'base_url': _root['api']['base_url'],
    'api_key':  _root['api']['api_key'],
    # Generation model (OpenAI-compatible)
    'gen_model': _root['api'].get('default_model', 'claude-haiku-4-5-20251001'),
    'gen_temperature': 0.0,
    'max_tokens': 512,
    'api_protocol': _root['api'].get('api_protocol', 'chat'),
}

# ─── Data paths ──────────────────────────────────────────────────────
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'gsm8k')

MEMORY_DIR = os.path.join(BASE_DIR, 'memory')
os.makedirs(MEMORY_DIR, exist_ok=True)

MEMORY_JSONL         = os.path.join(MEMORY_DIR, 'memory.jsonl')
EMBEDDINGS_NPY       = os.path.join(MEMORY_DIR, 'embeddings.npy')       # chunk embeddings
QUESTION_EMBEDDINGS_NPY = os.path.join(MEMORY_DIR, 'question_embeddings.npy')  # demo question embeddings

LOGS_DIR   = os.path.join(BASE_DIR, 'logs')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Memory versions (v8.6) ─────────────────────────────────────────
MEMORY_VERSIONS_DIR = os.path.join(BASE_DIR, 'memory_versions')
ABSTRACT_MEMORY_JSONL = os.path.join(MEMORY_VERSIONS_DIR, 'v0000', 'abstract_memory.jsonl')

# ─── Embedding ───────────────────────────────────────────────────────
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

# ─── Retrieval ───────────────────────────────────────────────────────
TOP_N_DEMOS = 8  # Layer 1: top-N demos retrieved by question similarity
TOP_K = 2        # Layer 2: top-K chunks retrieved by state similarity
SIMILARITY_THRESHOLD = 0.0  # No threshold (retriever.py imports this)
TOP_N_ABSTRACTS = 2  # Layer 0: top-N abstract patterns retrieved by state similarity

# ─── Three-stage retrieval (tag pre-filter → content recall → LLM rerank) ──
# Stage 1: soft tag pre-filter by modality/skills overlap (weights, not a hard
#          cut). Each overlapping tag nudges a candidate's blended score up.
STAGE1_MODALITY_WEIGHT = 0.10  # bonus per overlapping modality tag
STAGE1_SKILL_WEIGHT    = 0.20  # bonus per overlapping skill tag (finer-grained)
# Stage 2: content-similarity recall. Take top-M candidates into the reranker.
RERANK_CANDIDATES_M = 5
# Stage 3: LLM reranker (haiku) scores each candidate's *approach fit* to the
#          query puzzle in [0,1]. Only candidates scoring >= tau are injected.
RERANK_FIT_TAU = 0.7
# If every candidate scores below tau, inject nothing (degrade to no-memory)
# rather than feeding a misfit precedent that actively derails the solver.

# ─── Stage-3 sampling + majority-vote gate (reproducible) ──────────────
# The reranker is stochastic, so we sample it N times at a nonzero temperature
# and gate by MAJORITY VOTE: inject a candidate only if >= RERANK_VOTE_THRESHOLD
# of its N samples score >= RERANK_FIT_TAU. Voting is robust to a single sample
# drifting across the tau boundary (a bare mean gate is not).
RERANK_SAMPLES_N       = 5    # judge samples per candidate (the "votes")
RERANK_VOTE_THRESHOLD  = 3    # >= this many samples must clear tau to inject
RERANK_TEMPERATURE     = 1.0  # sampling temp (0 would make the N samples identical)
# The N sampled scores are cached on disk (in the memory-version dir) keyed by
# (prompt-template, query, candidate) — NOT by tau/threshold — so reruns read
# back identical samples and reach identical gate decisions. Force fresh
# sampling by deleting the cache file or setting ADAPTIVE_RERANK_FRESH=1.
RERANK_CACHE_FILENAME  = 'rerank_cache.jsonl'

# ─── Reasoning ───────────────────────────────────────────────────────
GEN_MAX_TOKENS = 512  # Max tokens for single-call reasoning generation

# ─── Prompt templates ────────────────────────────────────────────────

# Template: Offline triple extraction (prep_memory.py)
TPL_EXTRACT_TRIPLES = """You are a math problem-solving process decomposer. Below is a math problem and its full chain-of-thought. Break the reasoning into a sequence of (state, action, next_state) triples, meaning "state before this step → action of this step → state after this step".

Requirements:
- state: compact description of the current known quantities (variable_name = value/meaning) and the current goal. Do NOT repeat the original question text. Do NOT accumulate all intermediate results from previous steps.
- action: the mathematical operation of this step, must include a concrete equation (e.g. 16 - 3 - 4 = 9).
- next_state: compact state after this step, same format as state.
- Each triple corresponds to exactly ONE reasoning step. Do NOT merge multiple steps into one triple.

Output format: a JSON array, e.g.:
[
  {{"state": "total eggs = 16; eggs eaten = 3; eggs baked = 4; goal: eggs left", "action": "16 - 3 - 4 = 9", "next_state": "eggs left = 9"}},
  {{"state": "eggs left = 9; price per egg = $2; goal: total revenue", "action": "9 * 2 = 18", "next_state": "total revenue = $18"}}
]

Problem: {question}
Chain-of-thought: {cot}"""

# Template 2: Initial state extraction (online, test question)
TPL_INITIAL_STATE = """You are a math problem state parser. Convert the following math word problem into a compact initial state for step-by-step solving.

Output JSON ONLY (no other text):
{{"state": "compact description listing all known quantities with their values/meanings, and the goal"}}

Requirements:
- Do NOT output the original question text.
- Do NOT give the solution, only the initial state.
- Example: question "Janet's ducks lay 16 eggs. She eats 3 and bakes 4. How much does she make if she sells the rest at $2 each?"
  output: {{"state": "total eggs = 16; eggs eaten = 3; eggs baked = 4; price per egg = $2; goal: total revenue from remaining eggs"}}

Problem: {question}"""

# Template 3: Single-step reasoning (online, per step)
TPL_STEP_REASONING = """You are a math-solving agent. We are solving a math problem step by step, one reasoning action at a time. Given the current state and two relevant precedents from memory, decide the next single step.

Current state:
{state}

Relevant precedents from memory (similar state → successful action → resulting state):
Precedent 1:
  State then: {prev_state_1}
  Action then: {prev_action_1}
  Result state: {prev_next_state_1}
Precedent 2:
  State then: {prev_state_2}
  Action then: {prev_action_2}
  Result state: {prev_next_state_2}

Requirements:
1. Output exactly ONE step. Do NOT give the full solution.
2. Refer to the mathematical operations in similar precedents, but substitute values to match the current state's actual numbers.
3. If the current state already leads directly to the final answer, set status to "done" and write the answer-computing equation in action.
4. Output strict JSON only, no other text:
{{"action": "the mathematical operation for this step, with equation", "new_state": "compact state after this step", "status": "continue or done"}}"""

# Template 4: Single-call with memory (no step-by-step loop)
TPL_SINGLE_CALL = """You are solving a math problem. Below are two relevant precedents from memory that might help, followed by the problem to solve.

IMPORTANT: The precedents show ONLY the reasoning approach. Do NOT use any numbers or values from the precedents. Only use the numbers given in the actual problem below.

{precedents}

Problem: {question}

Solve this problem step by step. Show your reasoning clearly. At the end, output your final answer in the format:
#### <answer>

Example: #### 42
"""

# Template 5: Abstract pattern extraction (v8.6)
TPL_EXTRACT_ABSTRACTS = """You are a math reasoning pattern abstractor. Below are concrete (state, action, next_state) triples from math problem solutions. For each triple, extract the ABSTRACT reasoning pattern — the general mathematical operation or reasoning template, stripping out all specific numbers and entity names.

Requirements:
- Replace all specific numbers with variables (X, Y, A, B, R, etc.)
- Replace entity-specific terms with role labels (e.g., "eggs" → "initial_quantity", "price per egg" → "unit_price")
- Name the pattern_type: one of "subtraction_chain", "rate_multiplication", "proportional_allocation", "multi_step_addition", "multi_step_multiplication", "percentage_calculation", "ratio_comparison", "comparison_difference", "conversion_unit", "distribution_remainder", "other"
- The abstract pattern must be readable and useful without any concrete values
- abstract_action should describe the operation type AND give the abstract formula

Input concrete triples:
{concrete_triples}

Output a JSON array. For each triple output:
{{"abstract_state": "...", "abstract_action": "...", "abstract_next_state": "...", "pattern_type": "..."}}
"""

# Template 6: Abstract-aware precedent context (v8.6)
TPL_ABSTRACT_RETRIEVAL_CONTEXT = """You are solving a math problem. The reasoning precedents below show the ABSTRACT reasoning pattern AND a concrete example.

IMPORTANT: Do NOT use any numbers or values from the precedents. Only use the numbers given in the actual problem below.

Pattern type: {pattern_type}
Abstract: {abstract_state} → {abstract_action} → {abstract_next_state}
Concrete example: {concrete_state} → {concrete_action} → {concrete_next_state}"""

# Template 7: Memory evolution from failure cases (v8.6)
TPL_EVOLVE_MEMORY = """You are improving a math reasoning memory system. The system failed on these questions because the retrieved memory patterns were not helpful.

Failed cases:
{failed_cases}

Current abstract patterns in memory:
{current_patterns}

TASK: Propose new abstract patterns or refinements that would help solve similar questions in the future. Focus on the REASONING PATTERN, not specific numbers.

For each new pattern, output:
- abstract_state: what the initial state looks like (with variables, not numbers)
- abstract_action: the operation type and abstract formula
- abstract_next_state: the resulting state
- pattern_type: category name
- applicability: when this pattern should be triggered (description of question types)

Output a JSON array of new/updated abstract patterns.
"""
