"""Multi-arm retrieval runner for RealMemBench.

Implements the evaluation protocol from the published README verbatim:

    for session in dialogue_sessions:
        for turn in session['turns']:
            if turn['is_query']:
                keywords = generate_query_llm(question)
                memories = retrieve_memories(question, keywords, k=10)
                answer   = generate_answer(question, memories)
        memory_system.add_session_content(session)

The ordering is the whole point. Gold references point backwards 100% of the
time (0 of 2319 point forward), so a retriever that holds the entire corpus is
scored on a pool half of which cannot possibly be correct. The previous
implementation loaded every session up front and then iterated — every number it
produced was measured under that defect.

Every arm walks one shared loop, sees the same sessions in the same order, and
answers the same query set. Differences between arms are therefore attributable
to the ranking function and nothing else.
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import (
    API_KEY, BASE_URL, API_TIMEOUT, EMBEDDING_MODEL, KEYWORD_MODEL,
    KEYWORD_MAX_TOKENS, USE_KEYWORDS, RETRIEVE_K, MAX_METRIC_K,
    RESULT_CONTENT_CHARS, DEFAULT_ARMS, RERANK_CACHE_NAME,
    persona_dataset_path, persona_memory_bank, persona_retrieval_dir,
    list_personas,
)
from eval.embedding import Embedder
from eval.build_memory import build_bank, verify_bank
from eval.three_stage_retriever import (
    ThreeStageRetriever, SimpleEmbeddingRetriever, NoMemoryRetriever,
)
from eval import schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Arms ──

def make_arm(name: str, embedder: Embedder, llm_client, bank_dir: str,
             retrieve_k: int, topic_source: str, persona: str = ""):
    """Every three-stage arm shares one cache file: the cache key includes the
    candidate list and prompt version, so a gated run reuses a rerank run's
    samples instead of paying twice for identical judgements."""
    cache = os.path.join(bank_dir, RERANK_CACHE_NAME)

    if name == "no_memory":
        return NoMemoryRetriever()
    if name == "simple_embedding":
        return SimpleEmbeddingRetriever(
            embedder=embedder, retrieve_k=retrieve_k, topic_source=topic_source)
    if name == "three_stage_rerank":
        return ThreeStageRetriever(
            embedder=embedder, retrieve_k=retrieve_k, gate_mode="rerank",
            llm_client=llm_client, cache_path=cache, topic_source=topic_source)
    if name == "three_stage_gated":
        return ThreeStageRetriever(
            embedder=embedder, retrieve_k=retrieve_k, gate_mode="gate",
            llm_client=llm_client, cache_path=cache, topic_source=topic_source)
    if name == "stage2_only":
        # Ablation: Stage 1 + 2 without the judge.
        return ThreeStageRetriever(
            embedder=embedder, retrieve_k=retrieve_k, enable_stage3=False,
            llm_client=None, topic_source=topic_source)
    if name == "concrete_only":
        # The true plain-cosine control: one pool, session text only.
        # simple_embedding is NOT this — it keeps the Stage-2 dual-pool blend,
        # so comparing against it measures Stage 1 and Stage 3 but silently
        # credits Stage 2 to the baseline.
        return SimpleEmbeddingRetriever(
            embedder=embedder, retrieve_k=retrieve_k, topic_source=topic_source,
            enable_dual_pool=False)
    if name == "mem0":
        # Imported lazily: the core harness must not depend on mem0.
        # user_id carries the persona so mem0's per-persona store and collection
        # stay separate — its defaults are global and would cross-contaminate.
        from eval.mem0_arm import Mem0Retriever
        return Mem0Retriever(retrieve_k=retrieve_k, user_id=persona or "realmem_eval")
    raise ValueError(
        f"unknown arm {name!r}; known: no_memory, simple_embedding, "
        f"three_stage_rerank, three_stage_gated, stage2_only, concrete_only, mem0")


# Arms that reach the network on their own account. mem0 is here because it
# calls its own LLM/embedding endpoints on every add and search, so a dry run
# must not construct it.
ARMS_NEEDING_LLM = {"three_stage_rerank", "three_stage_gated", "mem0"}


# ── Keyword expansion (README step) ──

def generate_keywords(client, question: str, model: str) -> str:
    if client is None:
        return ""
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": ("Extract 3-6 search keywords for retrieving past "
                            "conversation sessions relevant to this message. "
                            "Reply with the keywords only, comma-separated.\n\n"
                            f"{question[:800]}"),
            }],
            temperature=0.0,
            max_tokens=KEYWORD_MAX_TOKENS,
            timeout=API_TIMEOUT,
        )
        return (res.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Keyword generation failed: %s", exc)
        return ""


# ── Runner ──

def run_persona(
    persona: str,
    dataset_path: str,
    bank_dir: str,
    output_dir: str,
    arm_names: List[str],
    retrieve_k: int = RETRIEVE_K,
    limit: int = 0,
    use_keywords: bool = USE_KEYWORDS,
    topic_source: str = "identifier",
    dry_run: bool = False,
    resume: bool = True,
    prefix: str = "",
    llm_client: Optional[object] = None,
) -> Dict[str, dict]:

    pending = []
    for arm in arm_names:
        out = os.path.join(output_dir, f"{prefix}{arm}_retrieval_results.json")
        if resume and os.path.exists(out):
            logger.info("  %s: results exist, skipping (use --no-resume to redo)", arm)
            continue
        pending.append(arm)
    if not pending:
        return {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if dry_run and "mem0" in pending:
        # mem0 reaches the network inside add_session, which no flag of ours can
        # suppress, so it cannot participate in a zero-network run.
        logger.warning("  dry-run: skipping the mem0 arm (it calls its own "
                       "endpoints on every add/search)")
        pending = [a for a in pending if a != "mem0"]
        if not pending:
            return {}

    needs_llm = bool(set(pending) & ARMS_NEEDING_LLM) or use_keywords
    if dry_run:
        # A dry run that still calls the judge is not a dry run. Nothing below
        # may construct a client, and an injected one is ignored.
        llm_client = None
        if needs_llm:
            logger.info("  dry-run: Stage 3 judge disabled, no network calls will be made")
    elif needs_llm and llm_client is None:
        from openai import OpenAI
        llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    embedder = Embedder(EMBEDDING_MODEL)
    arms = {name: make_arm(name, embedder, llm_client, bank_dir, retrieve_k,
                           topic_source, persona)
            for name in pending}

    for arm in arms.values():
        arm.reset()
        try:
            arm.attach_bank(bank_dir)
        except FileNotFoundError:
            logger.warning("  no bank at %s; sessions will be encoded on the fly", bank_dir)

    results: Dict[str, Dict[str, dict]] = {name: {} for name in pending}
    n_queries = 0
    query_cap = limit if limit > 0 else float("inf")

    logger.info("  streaming %d sessions...", len(data.get("dialogues", []) or []))

    for s_idx, session in schema.iter_sessions(data):
        sid = session.get("session_identifier", "")
        query_topic = schema.parse_topic(sid)
        turns = session.get("dialogue_turns", []) or []

        # 1. Answer this session's queries against history only.
        for t_idx, turn in enumerate(turns):
            if not turn.get("is_query"):
                continue
            question = (turn.get("content") or "").strip()
            if not question or n_queries >= query_cap:
                continue

            # One keyword expansion shared by every arm: differing query text
            # would confound the arm comparison.
            keywords = generate_keywords(llm_client, question, KEYWORD_MODEL) \
                if use_keywords else ""

            for name, arm in arms.items():
                items = arm.retrieve(question, query_topic=query_topic,
                                     keywords=keywords or None)
                results[name][question] = {
                    "id": turn.get("query_id") or f"{sid}_Q{t_idx}",
                    "question": question,
                    "session_identifier": sid,
                    "session_idx": s_idx,
                    "keywords": keywords,
                    "ranked_items": [_trim(it) for it in items],
                }
            n_queries += 1

        # 2. Only now does this session become visible.
        for arm in arms.values():
            arm.add_session(session)

    os.makedirs(output_dir, exist_ok=True)
    _write_run_config(output_dir, persona, pending, bank_dir, retrieve_k,
                      use_keywords, topic_source, dry_run, prefix)

    summaries = {}
    for name in pending:
        out = os.path.join(output_dir, f"{prefix}{name}_retrieval_results.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results[name], f, ensure_ascii=False, indent=2)
        summaries[name] = _diagnostics(name, results[name], arms[name],
                                       warn_gate=not dry_run)
        logger.info("  %-20s %d queries -> %s", name, len(results[name]), out)

    return summaries


def _write_run_config(output_dir: str, persona: str, arms: List[str], bank_dir: str,
                      retrieve_k: int, use_keywords: bool, topic_source: str,
                      dry_run: bool, prefix: str):
    """Freeze the settings that produced these results, next to the results.

    Every knob here is env-overridable, so a results directory on its own does
    not say what produced it. Six months later, "why is this number different
    from the paper" is unanswerable without this file — and tuning happens on
    dev personas before the test set is run, so the two sets MUST be checkable
    for having used identical settings.
    """
    import eval.config as cfg

    snapshot = {
        k: v for k, v in vars(cfg).items()
        if k.isupper() and isinstance(v, (int, float, str, bool, tuple, dict))
        and not k.endswith(("_DIR", "_ROOT", "KEY", "URL"))
    }
    bank_meta = {}
    meta_path = os.path.join(bank_dir, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            bank_meta = {k: m.get(k) for k in
                         ("bank_format", "embedding_model", "embedding_backend",
                          "n_sessions", "n_abstracts", "n_filler_sessions", "built_at")}
        except Exception:
            pass

    payload = {
        "persona": persona,
        "arms": list(arms),
        "retrieve_k": retrieve_k,
        "use_keywords": use_keywords,
        "topic_source": topic_source,
        "dry_run": dry_run,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bank": bank_meta,
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in sorted(snapshot.items())},
    }
    path = os.path.join(output_dir, f"{prefix}run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _trim(item: dict) -> dict:
    """Truncate stored content. The bank holds full text once; repeating it 20x
    per query per arm previously produced 10-27 MB artefacts per persona."""
    out = dict(item)
    content = out.get("content") or ""
    if len(content) > RESULT_CONTENT_CHARS:
        out["content"] = content[:RESULT_CONTENT_CHARS]
        out["content_truncated"] = True
    return out


def _diagnostics(name: str, results: dict, arm, warn_gate: bool = True) -> dict:
    """The numbers that say whether an arm is even functioning.

    Depth matters most: with Recall@20 in the metric suite, an arm returning 15
    items has a structural ceiling on its @20 column, which reads as a quality
    difference when it is really a configuration bug.
    """
    lengths = [len(r["ranked_items"]) for r in results.values()]
    empty = sum(1 for n in lengths if n == 0)
    depth = Counter(lengths)
    same_topic = injected = passed = 0
    for r in results.values():
        for it in r["ranked_items"]:
            injected += 1
            if it.get("same_topic"):
                same_topic += 1
            if it.get("stage3_passed"):
                passed += 1

    diag = {
        "n_queries": len(results),
        "empty_results": empty,
        "empty_rate": round(empty / len(results), 4) if results else 0.0,
        "mean_depth": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        "min_depth": min(lengths) if lengths else 0,
        "depth_below_max_k": sum(1 for n in lengths if 0 < n < MAX_METRIC_K),
        "same_topic_share": round(same_topic / injected, 4) if injected else None,
        "stage3_pass_share": round(passed / injected, 4) if injected else None,
    }
    if hasattr(arm, "stats"):
        diag["stats"] = dict(arm.stats)

    if diag["depth_below_max_k"]:
        logger.warning(
            "  %s: %d/%d queries returned fewer than %d items — Recall@%d is "
            "capped for those.", name, diag["depth_below_max_k"], len(results),
            MAX_METRIC_K, MAX_METRIC_K)
    if warn_gate and name == "three_stage_gated" and diag["empty_rate"] > 0.85:
        logger.warning(
            "  %s: gate rejected everything on %.0f%% of queries — this is gate "
            "collapse, not a result.", name, 100 * diag["empty_rate"])
    return diag


def main():
    p = argparse.ArgumentParser(description="Run retrieval arms over RealMemBench")
    p.add_argument("--persona", default=None)
    p.add_argument("--personas", default=None, help="Comma-separated")
    p.add_argument("--all-personas", action="store_true")
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    p.add_argument("--bank-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--retrieve-k", type=int, default=RETRIEVE_K)
    p.add_argument("--limit", type=int, default=0, help="Max queries per persona")
    p.add_argument("--use-keywords", action="store_true",
                   help="Enable the README's generate_query_llm step")
    p.add_argument("--topic-source", choices=("identifier", "none"), default="identifier",
                   help="'none' ablates the Stage-1 topic bonus")
    p.add_argument("--dry-run", action="store_true",
                   help="No network; DRYRUN- prefixed outputs only")
    p.add_argument("--no-resume", action="store_true", help="Recompute existing results")
    p.add_argument("--build-missing-banks", action="store_true")
    args = p.parse_args()

    if args.all_personas:
        personas = list_personas()
    elif args.personas:
        personas = [x.strip() for x in args.personas.split(",") if x.strip()]
    elif args.persona:
        personas = [args.persona]
    else:
        p.error("Specify --persona, --personas or --all-personas")

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    prefix = "DRYRUN-" if args.dry_run else ""
    all_diags = {}
    # A persona we could not evaluate must not look like success. Batch runners
    # key off the exit code, and a silently skipped persona means the final
    # analysis is missing data that nobody notices.
    skipped = []

    for persona in personas:
        logger.info("=" * 68)
        logger.info("Persona: %s", persona)
        logger.info("=" * 68)

        dataset_path = persona_dataset_path(persona)
        bank_dir = args.bank_dir or persona_memory_bank(persona)
        if args.dry_run:
            bank_dir = os.path.join(os.path.dirname(bank_dir), "DRYRUN-" + persona)

        may_build = args.build_missing_banks or args.dry_run
        if not os.path.exists(os.path.join(bank_dir, "meta.json")):
            if may_build:
                build_bank(dataset_path, persona, bank_dir, dry_run=args.dry_run)
            else:
                logger.error("No bank at %s. Run build_memory.py first, or pass "
                             "--build-missing-banks.", bank_dir)
                skipped.append((persona, "no bank"))
                continue
        else:
            try:
                verify_bank(bank_dir)
            except AssertionError as exc:
                # A stale or corrupt bank must never be ranked with. Rebuilding
                # is safe when we were already allowed to build one.
                if may_build:
                    logger.warning("Rebuilding unusable bank — %s", exc)
                    build_bank(dataset_path, persona, bank_dir, dry_run=args.dry_run)
                else:
                    logger.error("%s", exc)
                    skipped.append((persona, "unusable bank"))
                    continue

        diags = run_persona(
            persona=persona,
            dataset_path=dataset_path,
            bank_dir=bank_dir,
            output_dir=args.output_dir or persona_retrieval_dir(persona),
            arm_names=arm_names,
            retrieve_k=args.retrieve_k,
            limit=args.limit,
            use_keywords=args.use_keywords,
            topic_source=args.topic_source,
            dry_run=args.dry_run,
            resume=not args.no_resume,
            prefix=prefix,
        )
        if diags:
            all_diags[persona] = diags

    if all_diags:
        print("\n" + "=" * 68)
        print("RETRIEVAL DIAGNOSTICS")
        print("=" * 68)
        print(f"{'persona':16s} {'arm':20s} {'n':>5s} {'depth':>7s} {'empty':>7s} "
              f"{'topic':>7s} {'pass':>7s}")
        print("-" * 68)
        for persona, diags in all_diags.items():
            for arm, d in diags.items():
                st = "" if d["same_topic_share"] is None else f"{d['same_topic_share']:.3f}"
                pa = "" if d["stage3_pass_share"] is None else f"{d['stage3_pass_share']:.3f}"
                print(f"{persona:16s} {arm:20s} {d['n_queries']:5d} "
                      f"{d['mean_depth']:7.1f} {d['empty_rate']:7.3f} {st:>7s} {pa:>7s}")

    if skipped:
        print()
        for persona, why in skipped:
            logger.error("SKIPPED %s (%s) — produced no results", persona, why)
        logger.error("%d of %d personas were skipped; exiting non-zero so a batch "
                     "run does not mistake this for success.", len(skipped), len(personas))
        sys.exit(1)


if __name__ == "__main__":
    main()


