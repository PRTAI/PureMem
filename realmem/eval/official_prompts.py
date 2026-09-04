"""The vendored scoring/generation prompts, loaded from their source files.

The judge prompt IS the metric. A paraphrase changes the scale, and scores from
a paraphrased rubric cannot sit next to published numbers. The previous harness
had rewritten both prompts down to a few lines, dropping the explicit "do not
reward an answer for merely sounding reasonable" instruction and the three-step
Mem_recall definition.

The obvious fix — keep a hand-copied duplicate — fails quietly: the first
attempt at exactly that was already six characters off (trailing whitespace),
which no reviewer would ever catch. So nothing is copied. The strings and the
two generation helpers are extracted from

    eval/compute_llm_metrics_for_realmem.py     QA_eval_prompt, Mem_eval_prompt
    eval/run_generation.py                      construct_evidence_text,
                                                construct_answer_prompt

at import time, by parsing those files rather than importing them: both pull in
openai/tqdm at module scope, and the offline selftests must run without either.
Extracting the two functions' AST and exec'ing just those nodes gets the real
implementations with none of the dependencies.

Consequence worth stating: if the vendored files change, this module changes
with them. That is the intent — drift becomes impossible rather than merely
detectable.
"""

import ast
import os
from typing import Dict, List

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_METRICS_SRC = os.path.join(_EVAL_DIR, "compute_llm_metrics_for_realmem.py")
GENERATION_SRC = os.path.join(_EVAL_DIR, "run_generation.py")


def _parse(path: str) -> ast.Module:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"vendored evaluation script missing: {path}. It is the reference "
            f"definition of the metric and must not be deleted or moved.")
    with open(path, "r", encoding="utf-8") as f:
        return ast.parse(f.read(), filename=path)


def _string_constants(path: str) -> Dict[str, str]:
    out = {}
    for node in _parse(path).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value.value
    return out


def _extract_functions(path: str, names: List[str]) -> dict:
    """Exec only the named top-level functions, skipping module-level imports."""
    tree = _parse(path)
    wanted = [n for n in tree.body
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    missing = set(names) - {n.name for n in wanted}
    if missing:
        raise ImportError(f"{os.path.basename(path)} no longer defines {sorted(missing)}")
    namespace: dict = {}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), path, "exec"), namespace)
    return namespace


_consts = _string_constants(LLM_METRICS_SRC)
try:
    QA_eval_prompt: str = _consts["QA_eval_prompt"]
    Mem_eval_prompt: str = _consts["Mem_eval_prompt"]
except KeyError as exc:
    raise ImportError(
        f"{os.path.basename(LLM_METRICS_SRC)} no longer defines {exc}; the judge "
        f"prompts cannot be recovered.") from exc

_gen = _extract_functions(GENERATION_SRC,
                          ["construct_evidence_text", "construct_answer_prompt"])
construct_evidence_text = _gen["construct_evidence_text"]
construct_answer_prompt = _gen["construct_answer_prompt"]


def build_qa_judge_prompt(question: str, gt_memory: str, gt_answer: str,
                          candidate: str) -> str:
    """The concatenation used by compute_llm_metrics_for_realmem.evaluate_single_item."""
    return f"""{QA_eval_prompt}

### Input Data
1. Query: {question}
2. User-related Memory: {gt_memory}
3. Reference Answer: {gt_answer}
4. Candidate Answer: {candidate}
"""


def build_mem_judge_prompt(question: str, groundtruth_memory: str,
                           retrieved_memory: str) -> str:
    return Mem_eval_prompt.format(question=question,
                                  groundtruth_memory=groundtruth_memory,
                                  retrieved_memory=retrieved_memory)


def describe() -> dict:
    """Provenance, so a run log records which prompt text produced its scores."""
    return {
        "qa_prompt_chars": len(QA_eval_prompt),
        "mem_prompt_chars": len(Mem_eval_prompt),
        "qa_prompt_source": os.path.basename(LLM_METRICS_SRC),
        "generation_source": os.path.basename(GENERATION_SRC),
    }
