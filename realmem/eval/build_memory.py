"""Build a per-persona memory bank from a RealMemBench dialogue file.

A bank is an *offline embedding cache*, not a retrievable set. Nothing here may
be handed to a retriever wholesale: the published evaluation protocol requires
that a query only ever sees sessions that precede it, so the retriever admits
sessions one at a time and looks their vectors up here to avoid re-encoding.
See run_eval.py for the streaming loop that enforces this.

Layout:
  sessions.jsonl           one record per session, in dataset order
  session_embeddings.npy   (N_sessions, D) L2-normalized, row i <-> line i
  abstracts.jsonl          one record per extracted_memory item
  abstract_embeddings.npy  (N_abstracts, D) L2-normalized, row i <-> line i
  queries.jsonl            one record per is_query turn, with gold
  meta.json                counts, embedding backend, build provenance
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import (
    EMBEDDING_MODEL, EMBED_HEAD_CHARS, EMBED_TAIL_CHARS, BANK_FORMAT,
    persona_dataset_path, persona_memory_bank, list_personas,
)
from eval.embedding import Embedder, BACKEND_ST, looks_like_st_vectors
from eval import schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_bank(dataset_path: str, persona_name: str, bank_dir: str,
               dry_run: bool = False, allow_fallback: bool = True) -> str:
    logger.info("Loading dialogue data from %s", dataset_path)
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dialogues = data.get("dialogues", []) or []
    logger.info("Found %d sessions for persona '%s'", len(dialogues), persona_name)

    embedder = Embedder(EMBEDDING_MODEL, allow_fallback=allow_fallback)

    # ── Sessions (concrete pool) ──
    sessions: List[Dict] = []
    for idx, dlg in schema.iter_sessions(data):
        sid = dlg.get("session_identifier") or f"session_{idx}"
        text = schema.session_text(dlg)
        if not text.strip():
            continue
        sessions.append({
            "chunk_id": sid,
            "session_identifier": sid,
            "session_uuid": dlg.get("session_uuid", ""),
            "current_time": dlg.get("current_time", ""),
            "content": text,
            "num_turns": len(dlg.get("dialogue_turns", []) or []),
            "source_idx": idx,
            # Stage-1 tags, resolved once here so retrieval never re-parses.
            "topic": schema.parse_topic(sid),
            "project": schema.parse_project(sid),
            "has_abstract": bool(dlg.get("extracted_memory")),
        })

    logger.info("Collected %d concrete session chunks", len(sessions))
    session_embs = embedder.encode(
        [schema.head_tail(s["content"], EMBED_HEAD_CHARS, EMBED_TAIL_CHARS) for s in sessions],
        normalize=True,
    )

    # ── Abstracts (abstract pool) ──
    # Indexed by session_identifier rather than rescanned per session: the old
    # implementation did a linear scan of `dialogues` for every session.
    by_sid = {d.get("session_identifier"): d for _, d in schema.iter_sessions(data)}

    abstracts: List[Dict] = []
    for s in sessions:
        orig = by_sid.get(s["session_identifier"])
        if not orig:
            continue
        for mem in orig.get("extracted_memory", []) or []:
            content = mem.get("content", "")
            if not content:
                continue
            mem_type = mem.get("type", "General")
            abstracts.append({
                "abstract_id": len(abstracts),
                "chunk_id": s["session_identifier"],
                "session_uuid": s["session_uuid"],
                "source_idx": s["source_idx"],
                "abstract_type": mem_type,
                "content": f"[{mem_type}] {content}",
                "raw_content": content,
                "index": mem.get("index", ""),
            })

    logger.info("Collected %d abstracts from extracted_memory", len(abstracts))
    abstract_embs = embedder.encode([a["content"] for a in abstracts], normalize=True)

    # ── Queries + gold ──
    gold_map = schema.extract_retrieval_gold(data)
    queries = []
    for s_idx, session, t_idx, turn, question in schema.iter_queries(data):
        queries.append({
            "question": question,
            "session_identifier": session.get("session_identifier", ""),
            "session_uuid": session.get("session_uuid", ""),
            "session_idx": s_idx,
            "turn_idx": t_idx,
            "query_id": turn.get("query_id", ""),
            "current_time": session.get("current_time", ""),
            "gold_session_identifiers": gold_map.get(question, []),
        })
    logger.info("Collected %d query turns (%d with gold)",
                len(queries), sum(1 for q in queries if q["gold_session_identifiers"]))

    # ── Write ──
    os.makedirs(bank_dir, exist_ok=True)
    _write_jsonl(os.path.join(bank_dir, "sessions.jsonl"), sessions)
    _write_jsonl(os.path.join(bank_dir, "abstracts.jsonl"), abstracts)
    _write_jsonl(os.path.join(bank_dir, "queries.jsonl"), queries)
    np.save(os.path.join(bank_dir, "session_embeddings.npy"), session_embs)
    np.save(os.path.join(bank_dir, "abstract_embeddings.npy"), abstract_embs)

    meta = {
        "bank_format": BANK_FORMAT,
        "persona": persona_name,
        "n_sessions": len(sessions),
        "n_abstracts": len(abstracts),
        "n_queries": len(queries),
        "n_filler_sessions": sum(1 for s in sessions if not s["has_abstract"]),
        "n_topics": len({s["topic"] for s in sessions if s["topic"]}),
        "embedding_dim": int(session_embs.shape[1]) if session_embs.size else embedder.dim,
        "embedding_model": embedder.model_name,
        "embedding_backend": embedder.backend,
        "dry_run": bool(dry_run),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(bank_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    verify_bank(bank_dir)
    logger.info("Memory bank built at %s", bank_dir)
    logger.info("  sessions=%d abstracts=%d queries=%d backend=%s",
                len(sessions), len(abstracts), len(queries), embedder.backend)
    return bank_dir


def _write_jsonl(path: str, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_bank(bank_dir: str) -> dict:
    """Assert the alignment invariants a retriever depends on.

    Row i of an .npy must correspond to line i of its .jsonl. A misalignment
    here is invisible at read time and shows up only as inexplicably bad
    retrieval, so it is checked at build and again at load.
    """
    with open(os.path.join(bank_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    found_format = meta.get("bank_format", 1)
    if found_format != BANK_FORMAT:
        raise AssertionError(
            f"{bank_dir}: bank_format {found_format}, expected {BANK_FORMAT}. "
            f"A format-1 bank records neither the Stage-1 tags nor which encoder "
            f"built it, and an unlabelled bank can be silently matched against "
            f"queries from a different encoder. Rebuild: "
            f"python -m eval.build_memory --all-personas --require-st --force")

    sessions = _read_jsonl(os.path.join(bank_dir, "sessions.jsonl"))
    abstracts = _read_jsonl(os.path.join(bank_dir, "abstracts.jsonl"))
    sess_embs = np.load(os.path.join(bank_dir, "session_embeddings.npy"))
    abs_embs = np.load(os.path.join(bank_dir, "abstract_embeddings.npy"))

    if sess_embs.shape[0] != len(sessions):
        raise AssertionError(
            f"{bank_dir}: session_embeddings has {sess_embs.shape[0]} rows but "
            f"sessions.jsonl has {len(sessions)} lines")
    if abs_embs.shape[0] != len(abstracts):
        raise AssertionError(
            f"{bank_dir}: abstract_embeddings has {abs_embs.shape[0]} rows but "
            f"abstracts.jsonl has {len(abstracts)} lines")

    for name, embs in (("session", sess_embs), ("abstract", abs_embs)):
        if embs.shape[0]:
            norms = np.linalg.norm(embs, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-4):
                raise AssertionError(f"{bank_dir}: {name}_embeddings not L2-normalized")

    valid = {s["session_identifier"] for s in sessions}
    for a in abstracts:
        if a["chunk_id"] not in valid:
            raise AssertionError(
                f"{bank_dir}: abstract {a['abstract_id']} references unknown "
                f"chunk_id '{a['chunk_id']}'")

    ids = [s["session_identifier"] for s in sessions]
    if len(set(ids)) != len(ids):
        raise AssertionError(
            f"{bank_dir}: session_identifier is not unique — it is the corpus id "
            f"for Recall/NDCG, so duplicates would silently merge gold labels")

    # meta.json can claim a backend the .npy does not support.
    claimed = meta.get("embedding_backend")
    if claimed == BACKEND_ST and sess_embs.size and not looks_like_st_vectors(sess_embs):
        raise AssertionError(
            f"{bank_dir}: meta.json claims backend '{claimed}' but the vectors "
            f"look like the hash fallback (no negative components)")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Build a RealMemBench memory bank")
    parser.add_argument("--persona", default=None)
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--bank-dir", default=None)
    parser.add_argument("--all-personas", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write to DRYRUN- prefixed bank dirs only")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing banks instead of building")
    # Default to failing. A bank silently built on bag-of-words hashing looks
    # entirely normal on disk and produces meaningless retrieval; building one
    # is never what you wanted from this CLI.
    parser.add_argument("--allow-hash-fallback", action="store_true",
                        help="Permit the non-semantic hash encoder if "
                             "sentence-transformers is unavailable (testing only)")
    parser.add_argument("--require-st", action="store_true",
                        help="Deprecated: this is now the default")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the bank already exists")
    args = parser.parse_args()

    if args.all_personas:
        personas = list_personas()
    elif args.persona:
        personas = [args.persona]
    elif args.dataset_file:
        personas = [os.path.basename(args.dataset_file).replace("_dialogues_256k.json", "")]
    else:
        parser.error("Specify --persona, --dataset-file or --all-personas")

    if args.verify:
        failures = 0
        for persona in personas:
            bank_dir = args.bank_dir or persona_memory_bank(persona)
            try:
                meta = verify_bank(bank_dir)
                logger.info("OK   %-16s sessions=%-4d abstracts=%-5d backend=%s",
                            persona, meta["n_sessions"], meta["n_abstracts"],
                            meta.get("embedding_backend"))
            except Exception as exc:
                failures += 1
                logger.error("FAIL %-16s %s", persona, exc)
        sys.exit(1 if failures else 0)

    for persona in personas:
        dataset_path = args.dataset_file or persona_dataset_path(persona)
        bank_dir = args.bank_dir or persona_memory_bank(persona)
        if args.dry_run:
            bank_dir = os.path.join(os.path.dirname(bank_dir), "DRYRUN-" + persona)

        if os.path.exists(os.path.join(bank_dir, "meta.json")) and not args.force:
            logger.info("Bank exists, skipping (use --force to rebuild): %s", bank_dir)
            continue

        try:
            build_bank(dataset_path, persona, bank_dir, dry_run=args.dry_run,
                       allow_fallback=args.allow_hash_fallback)
        except RuntimeError as exc:
            # Encoder unavailable. A stack trace buries the one line that says
            # what to do about it.
            logger.error("%s", exc)
            sys.exit(1)


if __name__ == "__main__":
    main()
