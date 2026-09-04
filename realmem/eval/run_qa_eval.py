"""Answer generation + LLM-judge scoring for RealMemBench retrieval arms.

Mirrors the vendored two-step pipeline:

    run_generation.py                  retrieved memories -> answer
    compute_llm_metrics_for_realmem.py answer + gold       -> QA / Mem scores

Prompts and evidence formatting are the official ones, loaded from those files
by eval/official_prompts.py rather than paraphrased, so the 0-3 QA scale and the
Mem_recall definition match the published rubric exactly.

Artefacts are written in the official layout
(``{question: {question, generated_answer, evidence_used, ranked_items}}``), so
the vendored scripts can be pointed at them directly as an independent check.

Reading the output: ``average_mem_recall`` is computed only over queries where
evidence was actually retrieved AND gold memory is annotated — the official
script skips the memory prompt otherwise. So the no_memory arm has no
mem_recall at all rather than a zero, and the per-metric ``n`` differs between
arms. The summary prints every ``n`` for that reason; comparing a mean without
it will read a missing measurement as a tie.
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import (
    API_KEY, BASE_URL, API_TIMEOUT, GENERATION_MODEL, JUDGE_MODEL,
    GEN_TOP_K, GEN_TEMPERATURE, JUDGE_TEMPERATURE, MAX_WORKERS,
    DEFAULT_ARMS, RETRIEVAL_RESULT_DIR,
    persona_dataset_path, persona_retrieval_dir, list_personas,
)
from eval.official_prompts import (
    construct_evidence_text, construct_answer_prompt,
    build_qa_judge_prompt, build_mem_judge_prompt, describe as prompt_provenance,
)
from eval import schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_json(text: str):
    """Official parsing order: fenced block, then any braces, then the raw body."""
    if not text:
        return None
    try:
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        return json.loads(text)
    except Exception:
        return None


def _chat(client, model: str, prompt: str, temperature: float) -> Optional[str]:
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        timeout=API_TIMEOUT,
    )
    return res.choices[0].message.content


# ── Generation ──

def run_generation(retrieval_results: dict, client, model: str, top_k: int,
                   max_workers: int) -> dict:
    def gen_one(qid: str, item: dict) -> dict:
        question = item.get("question", qid)
        evidence = construct_evidence_text(item.get("ranked_items", []), "chunk", top_k)
        answer, error = "", None
        try:
            answer = _chat(client, model, construct_answer_prompt(question, evidence),
                           GEN_TEMPERATURE) or ""
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("Generation failed for %.60s: %s", qid, error)
        return {
            "id": item.get("id", qid),
            "question": question,
            "generated_answer": answer,
            "evidence_used": evidence,
            "ranked_items": item.get("ranked_items", []),
            "gen_error": error,
        }

    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(gen_one, qid, item): qid
                   for qid, item in retrieval_results.items()}
        for fut in as_completed(futures):
            qid = futures[fut]
            try:
                out[qid] = fut.result()
            except Exception as exc:
                logger.warning("Generation thread died for %.60s: %s", qid, exc)
    return out


# ── Judging ──

def run_judge(gen_results: dict, gt_map: Dict[str, dict], client, model: str,
              top_k: int, max_workers: int) -> tuple:
    def judge_one(qid: str, item: dict) -> dict:
        gt = gt_map.get(item.get("question", qid))
        if not gt:
            return {}
        gt_memory, gt_answer = gt["memory"], gt["answer"]
        answer = item.get("generated_answer", "")
        res = {}

        if answer:
            parsed = extract_json(_chat(
                client, model,
                build_qa_judge_prompt(item["question"], gt_memory, gt_answer, answer),
                JUDGE_TEMPERATURE))
            if parsed and isinstance(parsed.get("score"), (int, float)):
                res["qa_score"] = parsed["score"]
                res["qa_reason"] = parsed.get("reason", "")

        # Official gating: no evidence or no annotated memory -> no Mem metrics.
        evidence = construct_evidence_text(item.get("ranked_items", []), "chunk", top_k)
        if evidence and gt_memory and gt_memory != schema.NO_MEMORY_ANNOTATION:
            parsed = extract_json(_chat(
                client, model,
                build_mem_judge_prompt(item["question"], gt_memory, evidence),
                JUDGE_TEMPERATURE))
            if parsed:
                if isinstance(parsed.get("Mem_recall"), (int, float)):
                    res["mem_recall"] = float(parsed["Mem_recall"])
                if isinstance(parsed.get("Mem_helpful_score"), (int, float)):
                    res["mem_helpful"] = parsed["Mem_helpful_score"]
        return res

    detailed = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(judge_one, qid, item): qid
                   for qid, item in gen_results.items()}
        for fut in as_completed(futures):
            qid = futures[fut]
            try:
                r = fut.result()
                if r:
                    detailed[qid] = r
            except Exception as exc:
                logger.warning("Judge thread died for %.60s: %s", qid, exc)

    qa = [r["qa_score"] for r in detailed.values() if "qa_score" in r]
    rec = [r["mem_recall"] for r in detailed.values() if "mem_recall" in r]
    helpful = [r["mem_helpful"] for r in detailed.values() if "mem_helpful" in r]

    summary = {
        "n_generated": len(gen_results),
        "n_gen_failed": sum(1 for v in gen_results.values() if v.get("gen_error")),
        "n_qa_scored": len(qa),
        "n_mem_scored": len(rec),
    }
    if qa:
        dist = Counter(int(s) for s in qa)
        summary["average_qa_score"] = round(float(np.mean(qa)), 4)
        summary["qa_score_distribution"] = {i: dist.get(i, 0) for i in range(4)}
        summary["qa_hallucination_rate"] = round(dist.get(0, 0) / len(qa), 4)
        summary["qa_perfect_rate"] = round(dist.get(3, 0) / len(qa), 4)
    if rec:
        summary["average_mem_recall"] = round(float(np.mean(rec)), 4)
    if helpful:
        summary["average_mem_helpful_score"] = round(float(np.mean(helpful)), 4)
    return summary, detailed


# ── Driver ──

def run_arm(persona: str, arm: str, results_dir: str, gt_map: dict, client,
            gen_model: str, judge_model: str, top_k: int, max_workers: int,
            resume: bool) -> Optional[dict]:
    retrieval_file = os.path.join(results_dir, f"{arm}_retrieval_results.json")
    if not os.path.exists(retrieval_file):
        alt = os.path.join(results_dir, f"DRYRUN-{arm}_retrieval_results.json")
        if not os.path.exists(alt):
            logger.warning("  %s: no retrieval results", arm)
            return None
        retrieval_file = alt

    gen_file = os.path.join(results_dir, f"{arm}_generation_results.json")
    metrics_file = os.path.join(results_dir, f"{arm}_llm_metrics.json")

    if resume and os.path.exists(metrics_file):
        logger.info("  %s: metrics exist, skipping", arm)
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)["summary"]

    with open(retrieval_file, "r", encoding="utf-8") as f:
        retrieval = json.load(f)

    if resume and os.path.exists(gen_file):
        logger.info("  %s: reusing cached generations", arm)
        with open(gen_file, "r", encoding="utf-8") as f:
            gen = json.load(f)
    else:
        logger.info("  %s: generating %d answers (%s)", arm, len(retrieval), gen_model)
        gen = run_generation(retrieval, client, gen_model, top_k, max_workers)
        with open(gen_file, "w", encoding="utf-8") as f:
            json.dump(gen, f, ensure_ascii=False, indent=2)

    logger.info("  %s: judging (%s)", arm, judge_model)
    summary, detailed = run_judge(gen, gt_map, client, judge_model, top_k, max_workers)

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "prompts": prompt_provenance(),
                   "detailed_results": detailed}, f, ensure_ascii=False, indent=2)
    logger.info("  %s: %s", arm, {k: v for k, v in summary.items()
                                  if k != "qa_score_distribution"})
    return summary


def main():
    p = argparse.ArgumentParser(description="Generate answers and score them")
    p.add_argument("--personas", default=None)
    p.add_argument("--all-personas", action="store_true")
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    p.add_argument("--gen-model", default=GENERATION_MODEL)
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--top-k", type=int, default=GEN_TOP_K)
    p.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    if args.all_personas:
        personas = list_personas()
    elif args.personas:
        personas = [x.strip() for x in args.personas.split(",") if x.strip()]
    else:
        p.error("Specify --personas or --all-personas")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    logger.info("Prompt provenance: %s", prompt_provenance())

    all_results = {}
    for persona in personas:
        logger.info("=" * 68)
        logger.info("Persona: %s", persona)
        logger.info("=" * 68)
        with open(persona_dataset_path(persona), "r", encoding="utf-8") as f:
            data = json.load(f)
        gt_map = schema.extract_qa_gold(data)
        results_dir = persona_retrieval_dir(persona)

        per_arm = {}
        for arm in arms:
            summary = run_arm(persona, arm, results_dir, gt_map, client,
                              args.gen_model, args.judge_model, args.top_k,
                              args.max_workers, resume=not args.no_resume)
            if summary:
                per_arm[arm] = summary
        all_results[persona] = per_arm

    _print_summary(all_results, personas, arms)

    out = os.path.join(RETRIEVAL_RESULT_DIR, "qa_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"per_persona": all_results, "prompts": prompt_provenance()},
                  f, indent=2, ensure_ascii=False)
    logger.info("QA summary written to %s", out)


def _print_summary(all_results: dict, personas: List[str], arms: List[str]):
    metrics = ["average_qa_score", "qa_hallucination_rate", "qa_perfect_rate",
               "average_mem_recall", "average_mem_helpful_score"]
    print("\n" + "=" * 108)
    print("LLM JUDGE  (macro-average over personas)")
    print("=" * 108)
    header = f"{'arm':<21}"
    for m in metrics:
        header += f"{m.replace('average_', ''):>19}"
    print(header + f"{'n_qa':>8}{'n_mem':>8}")
    print("-" * 108)

    for arm in arms:
        vals = {m: [] for m in metrics}
        n_qa = n_mem = 0
        for persona in personas:
            s = all_results.get(persona, {}).get(arm)
            if not s:
                continue
            n_qa += s.get("n_qa_scored", 0)
            n_mem += s.get("n_mem_scored", 0)
            for m in metrics:
                if m in s:
                    vals[m].append(s[m])
        row = f"{arm:<21}"
        for m in metrics:
            row += f"{np.mean(vals[m]):>19.4f}" if vals[m] else f"{'n/a':>19}"
        print(row + f"{n_qa:>8}{n_mem:>8}")

    print("\nn/a in mem_recall is a *skipped measurement*, not a zero: the official "
          "\nrubric omits the memory prompt when no evidence was retrieved.")


if __name__ == "__main__":
    main()
