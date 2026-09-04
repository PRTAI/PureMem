"""End-to-end harness invariants on a fabricated persona. No network, no spend.

selftest_retrieval.py checks the retriever in isolation; this checks the loop
that drives it — the part where the temporal protocol actually lives, and where
arm comparability is either established or quietly lost.

What is asserted:

  * every arm answers exactly the same queries, in the same order, having seen
    exactly the same sessions — otherwise arm deltas are not attributable
  * results carry no session at or after the query's own                <- P0
  * on-disk format is what the vendored scorers expect
  * --dry-run constructs no client and issues no request, even for arms whose
    whole point is calling one
  * --limit and resume behave as documented
  * bank verification rejects a misaligned bank instead of ranking with it

Run:  python -m eval.selftest_run
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import schema
from eval.build_memory import build_bank, verify_bank
from eval.run_eval import run_persona, make_arm, ARMS_NEEDING_LLM
from eval.embedding import Embedder

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


# ── Fixture ──

TOPICS = ["Travel_Planning", "Fitness", "Knowledge_Learning"]


def make_persona(n_sessions=30, queries_every=4):
    """A miniature persona with the real corpus's structure: filler sessions with
    no extracted_memory, projects with them, and queries that reference the past."""
    dialogues = []
    for i in range(n_sessions):
        topic = TOPICS[i % 3] if i % 4 else None
        if topic:
            sid, mems = f"{topic}_1:S1_{i:02d}", [
                {"index": f"{topic}-DM-S1_{i:02d}-01", "type": "Dynamic",
                 "content": f"{topic} commitment recorded in session {i}"}]
        else:
            sid, mems = f"Enhanced:S1{i:04d}", []

        turns = [
            {"speaker": "User", "content": f"Let us work on {topic or 'small talk'} item {i}",
             "is_query": False},
            {"speaker": "Assistant", "content": f"Noted for session {i}", "is_query": False},
        ]
        # A query in later sessions, whose gold is an earlier session.
        # Offset by 2 so it never lands on a filler slot (i % 4 == 0), and so
        # gold (i - 4) is also a project session.
        if i >= queries_every and i % queries_every == 2 and topic:
            gold_idx = i - queries_every
            turns.append({"speaker": "User", "is_query": True,
                          "query_id": f"Q{i}",
                          "content": f"What did we decide about {topic} earlier, in round {gold_idx}?"})
            turns.append({"speaker": "Assistant",
                          "content": f"Earlier you decided on {topic} plan {gold_idx}",
                          "memory_session_uuids": [f"uuid-{gold_idx}"],
                          "memory_used": [{"session_uuid": f"uuid-{gold_idx}",
                                           "content": f"{TOPICS[gold_idx % 3]} commitment "
                                                      f"recorded in session {gold_idx}"}]})
        dialogues.append({
            "session_identifier": sid,
            "session_uuid": f"uuid-{i}",
            "current_time": f"2025-12-{(i % 28) + 1:02d} (Monday)",
            "extracted_memory": mems,
            "dialogue_turns": turns,
        })
    return {"_metadata": {"person_name": "Test_Persona", "total_sessions": n_sessions},
            "dialogues": dialogues}


class FakeClient:
    """Judge stand-in that accepts everything, and counts calls."""

    def __init__(self, fit=0.95):
        self.calls = 0
        outer = self

        class Completions:
            def create(self, model, messages, temperature=0.0, timeout=None, **kw):
                outer.calls += 1
                prompt = messages[0]["content"]
                fits, idx = [], 1
                while f"[{idx}]" in prompt:
                    fits.append({"idx": idx, "fit": fit})
                    idx += 1
                body = json.dumps(fits)

                class M:
                    content = body

                class C:
                    message = M()

                class R:
                    choices = [C()]

                return R()

        class Chat:
            completions = Completions()

        self.chat = Chat()


class ExplodingClient:
    """Any use at all is a failure."""

    def __init__(self):
        outer = self

        class Completions:
            def create(self, *a, **kw):
                raise AssertionError("network call issued during a dry run")

        class Chat:
            completions = Completions()

        self.chat = Chat()


def setup(tmp, n_sessions=30):
    data = make_persona(n_sessions)
    ds = os.path.join(tmp, "Test_Persona_dialogues_256k.json")
    with open(ds, "w", encoding="utf-8") as f:
        json.dump(data, f)
    bank = os.path.join(tmp, "bank")
    build_bank(ds, "Test_Persona", bank)
    return data, ds, bank


# ── Tests ──

def test_arms_are_comparable_and_causal():
    with tempfile.TemporaryDirectory() as tmp:
        data, ds, bank = setup(tmp)
        out = os.path.join(tmp, "results")
        arms = ["no_memory", "simple_embedding", "stage2_only", "three_stage_rerank"]

        run_persona("Test_Persona", ds, bank, out, arms,
                    retrieve_k=20, llm_client=FakeClient())

        loaded = {}
        for arm in arms:
            path = os.path.join(out, f"{arm}_retrieval_results.json")
            with open(path, "r", encoding="utf-8") as f:
                loaded[arm] = json.load(f)

        keysets = {arm: list(r.keys()) for arm, r in loaded.items()}
        first = keysets[arms[0]]
        check("every arm answered the same queries in the same order",
              all(keysets[a] == first for a in arms),
              {a: len(k) for a, k in keysets.items()})

        expected = sum(1 for _ in schema.iter_queries(data))
        check("query count matches the dataset", len(first) == expected,
              f"{len(first)} vs {expected}")

        # Temporal causality, measured on real output rather than in principle.
        order = {s["session_identifier"]: i for i, s in enumerate(data["dialogues"])}
        violations = []
        for arm, results in loaded.items():
            for q, rec in results.items():
                own = order[rec["session_identifier"]]
                for item in rec["ranked_items"]:
                    if order.get(item["chunk_id"], -1) >= own:
                        violations.append((arm, item["chunk_id"]))
        check("no arm retrieved its own or a later session", not violations,
              str(violations[:3]))

        check("no_memory returns nothing",
              all(r["ranked_items"] == [] for r in loaded["no_memory"].values()))
        check("memory arms return something",
              all(any(r["ranked_items"] for r in loaded[a].values())
                  for a in ("simple_embedding", "stage2_only", "three_stage_rerank")))


def test_output_format_matches_official_expectations():
    with tempfile.TemporaryDirectory() as tmp:
        _data, ds, bank = setup(tmp)
        out = os.path.join(tmp, "results")
        run_persona("Test_Persona", ds, bank, out, ["simple_embedding"], retrieve_k=20)
        with open(os.path.join(out, "simple_embedding_retrieval_results.json"),
                  "r", encoding="utf-8") as f:
            results = json.load(f)

        rec = next(iter(results.values()))
        check("top level is keyed by question text",
              next(iter(results.keys())) == rec["question"])
        check("each record carries question and ranked_items",
              "question" in rec and "ranked_items" in rec)
        item = rec["ranked_items"][0]
        check("ranked item uses res_type='chunk'", item["res_type"] == "chunk")
        check("chunk_id is a session_identifier", ":" in item["chunk_id"])
        for field in ("content", "score", "rank"):
            check(f"ranked item has '{field}'", field in item)


def test_dry_run_makes_no_network_calls():
    with tempfile.TemporaryDirectory() as tmp:
        _data, ds, bank = setup(tmp)
        out = os.path.join(tmp, "results")
        # Both a judge-dependent arm and keyword expansion requested, with a
        # client that raises if touched.
        run_persona("Test_Persona", ds, bank, out,
                    ["three_stage_rerank", "three_stage_gated"],
                    retrieve_k=20, dry_run=True, use_keywords=True,
                    llm_client=ExplodingClient(), prefix="DRYRUN-")

        check("dry run wrote only DRYRUN- prefixed files",
              all(f.startswith("DRYRUN-") for f in os.listdir(out)),
              str(os.listdir(out)))
        with open(os.path.join(out, "DRYRUN-three_stage_gated_retrieval_results.json"),
                  "r", encoding="utf-8") as f:
            gated = json.load(f)
        check("gate mode without a judge yields empty results, not fabricated ones",
              all(r["ranked_items"] == [] for r in gated.values()))


def test_shared_cache_within_one_run():
    """Both gate modes in ONE run_persona — the way it is actually invoked.

    This is the case an earlier version of this file missed by testing two
    separate run_persona calls. run_eval constructs every arm up front, so a
    cache snapshot taken per instance at construction time is empty for all of
    them; the gated arm then re-paid for judgements the rerank arm had already
    made and written. Same candidates, same prompt, double the bill and double
    the wall clock — which is exactly what a real 2-hour single-persona run
    turned out to be.
    """
    from eval.three_stage_retriever import reset_cache_registry
    with tempfile.TemporaryDirectory() as tmp:
        _data, ds, bank = setup(tmp)
        out = os.path.join(tmp, "results")

        reset_cache_registry()
        both = FakeClient()
        run_persona("Test_Persona", ds, bank, out,
                    ["three_stage_rerank", "three_stage_gated"],
                    retrieve_k=20, llm_client=both)
        calls_both = both.calls

        # Baseline: the same work with only the rerank arm.
        reset_cache_registry()
        shutil.rmtree(out, ignore_errors=True)
        os.remove(os.path.join(bank, "rerank_cache.jsonl"))
        solo = FakeClient()
        run_persona("Test_Persona", ds, bank, out, ["three_stage_rerank"],
                    retrieve_k=20, llm_client=solo)
        calls_solo = solo.calls

        check("judge was actually exercised", calls_solo > 0)
        check("adding the gated arm costs no extra judge calls",
              calls_both == calls_solo,
              f"two arms={calls_both} one arm={calls_solo} "
              f"(ratio {calls_both / max(calls_solo, 1):.1f}x)")


def test_cache_survives_across_runs():
    """A second invocation reuses what the first wrote to disk."""
    from eval.three_stage_retriever import reset_cache_registry
    with tempfile.TemporaryDirectory() as tmp:
        _data, ds, bank = setup(tmp)
        out = os.path.join(tmp, "results")
        client = FakeClient()
        run_persona("Test_Persona", ds, bank, out, ["three_stage_rerank"],
                    retrieve_k=20, llm_client=client)
        after_first = client.calls

        # Simulate a fresh process: drop in-memory state, keep the file.
        reset_cache_registry()
        shutil.rmtree(out, ignore_errors=True)
        run_persona("Test_Persona", ds, bank, out, ["three_stage_gated"],
                    retrieve_k=20, llm_client=client)
        check("a later run reloads the cache from disk",
              client.calls == after_first, f"{client.calls} vs {after_first}")


def test_limit_and_resume():
    with tempfile.TemporaryDirectory() as tmp:
        _data, ds, bank = setup(tmp)
        out = os.path.join(tmp, "results")
        run_persona("Test_Persona", ds, bank, out, ["simple_embedding"], limit=3)
        with open(os.path.join(out, "simple_embedding_retrieval_results.json"),
                  "r", encoding="utf-8") as f:
            check("--limit caps the query count", len(json.load(f)) == 3)

        diag = run_persona("Test_Persona", ds, bank, out, ["simple_embedding"],
                           resume=True)
        check("resume skips an arm that already has results", diag == {})

        run_persona("Test_Persona", ds, bank, out, ["simple_embedding"], resume=False)
        with open(os.path.join(out, "simple_embedding_retrieval_results.json"),
                  "r", encoding="utf-8") as f:
            check("--no-resume recomputes the full set", len(json.load(f)) > 3)


def test_bank_verification_catches_corruption():
    with tempfile.TemporaryDirectory() as tmp:
        _data, ds, bank = setup(tmp)
        check("a freshly built bank verifies", bool(verify_bank(bank)))

        emb_path = os.path.join(bank, "session_embeddings.npy")
        good = np.load(emb_path)
        np.save(emb_path, good[:-3])
        try:
            verify_bank(bank)
            check("misaligned embeddings are rejected", False, "verify_bank accepted it")
        except AssertionError:
            check("misaligned embeddings are rejected", True)
        np.save(emb_path, good)

        np.save(emb_path, good * 3.0)
        try:
            verify_bank(bank)
            check("unnormalized embeddings are rejected", False, "verify_bank accepted it")
        except AssertionError:
            check("unnormalized embeddings are rejected", True)


def test_bank_records_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        _data, ds, bank = setup(tmp)
        with open(os.path.join(bank, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        check("meta records the embedding backend", "embedding_backend" in meta)
        check("meta records dry_run honestly", meta["dry_run"] is False)
        check("meta counts filler sessions", meta["n_filler_sessions"] > 0)
        check("bank stores Stage-1 tags", _bank_has_tags(bank))


def _bank_has_tags(bank):
    with open(os.path.join(bank, "sessions.jsonl"), "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    return all("topic" in r and "has_abstract" in r for r in rows)


def test_backend_mismatch_is_refused():
    """A hash-encoded query against an ST-encoded bank has the right shape and
    produces confident nonsense — it must fail loudly instead."""
    with tempfile.TemporaryDirectory() as tmp:
        _data, ds, bank = setup(tmp, n_sessions=12)
        meta_path = os.path.join(bank, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        real_backend = meta["embedding_backend"]
        meta["embedding_backend"] = "some-other-encoder"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        from eval.three_stage_retriever import ThreeStageRetriever
        r = ThreeStageRetriever(embedder=Embedder("all-MiniLM-L6-v2"), enable_stage3=False)
        try:
            r.attach_bank(bank)
            check("bank built with a different encoder is refused", False,
                  "attach_bank accepted it")
        except RuntimeError:
            check("bank built with a different encoder is refused", True)

        meta["embedding_backend"] = real_backend
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        r.attach_bank(bank)
        check("matching backend attaches cleanly", True)


def test_stale_bank_format_is_refused():
    """A format-1 bank records neither the Stage-1 tags nor which encoder built
    it. The missing encoder field is the dangerous half: without it the backend
    check silently passes and a hash query gets ranked against ST vectors."""
    with tempfile.TemporaryDirectory() as tmp:
        _data, ds, bank = setup(tmp, n_sessions=12)
        meta_path = os.path.join(bank, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        # Simulate a bank produced by the old builder.
        del meta["bank_format"]
        del meta["embedding_backend"]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        try:
            verify_bank(bank)
            check("format-1 bank is refused by verify_bank", False, "it was accepted")
        except AssertionError:
            check("format-1 bank is refused by verify_bank", True)

        from eval.three_stage_retriever import ThreeStageRetriever
        r = ThreeStageRetriever(embedder=Embedder("all-MiniLM-L6-v2"), enable_stage3=False)
        try:
            r.attach_bank(bank)
            check("format-1 bank is refused by attach_bank", False, "it was accepted")
        except RuntimeError:
            check("format-1 bank is refused by attach_bank", True)


def test_analysis_runs_end_to_end():
    """The retrieval -> analysis chain, including the macro-average that used to
    report only the first persona."""
    with tempfile.TemporaryDirectory() as tmp:
        data, ds, bank = setup(tmp)
        results_root = os.path.join(tmp, "retrieval_result")
        out = os.path.join(results_root, "Test_Persona")
        arms = ["no_memory", "simple_embedding", "stage2_only"]
        run_persona("Test_Persona", ds, bank, out, arms, retrieve_k=20)

        # analyze() locates dialogue files by persona name inside dataset_dir.
        from eval.analyze import analyze
        agg = analyze(["Test_Persona"], arms, results_root, tmp,
                      baseline="simple_embedding")

        check("analysis produced an aggregate for each arm",
              all(a in agg for a in arms), str(list(agg)))
        check("no_memory scores zero recall",
              agg["no_memory"]["recall_any@10"]["mean"] == 0.0)
        check("a memory arm scores above zero",
              agg["simple_embedding"]["recall_any@10"]["mean"] > 0.0)

        summary_path = os.path.join(results_root, "analysis_summary.json")
        with open(summary_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        check("summary records paired comparisons", "paired_comparisons" in saved)
        check("summary records which arms ran", set(saved["arms"]) == set(arms))


def test_macro_average_uses_every_persona():
    """Direct regression on the bug where the aggregate table's comprehension
    ignored its loop variable and reported persona #1 for every column."""
    from eval.analyze import _aggregate
    per_persona = {
        "A": {"arm": {"auto": {"recall_any@5": 0.10, "recall_any@10": 0.20,
                               "ndcg@10": 0.30, "n_evaluated": 100}}},
        "B": {"arm": {"auto": {"recall_any@5": 0.90, "recall_any@10": 0.80,
                               "ndcg@10": 0.70, "n_evaluated": 100}}},
    }
    agg = _aggregate(per_persona, ["arm"])
    check("macro-average is the mean over personas, not the first one",
          abs(agg["arm"]["recall_any@5"]["mean"] - 0.5) < 1e-9,
          f"got {agg['arm']['recall_any@5']['mean']}")
    check("aggregate counts personas", agg["arm"]["recall_any@5"]["n_personas"] == 2)
    check("aggregate totals the query count", agg["arm"]["n_queries_total"] == 200)


def test_skipped_persona_exits_nonzero():
    """A persona that produced nothing must not report success.

    Batch runners key off the exit code. A silently skipped persona means the
    final analysis is quietly missing data — which is worse than a crash,
    because the tables still render.
    """
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        # A dataset with no bank, and building one is not permitted.
        data = make_persona(8)
        ds_dir = os.path.join(tmp, "dataset")
        os.makedirs(ds_dir)
        with open(os.path.join(ds_dir, "Ghost_dialogues_256k.json"), "w",
                  encoding="utf-8") as f:
            json.dump(data, f)

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-m", "eval.run_eval", "--persona", "Ghost",
             "--arms", "simple_embedding", "--bank-dir",
             os.path.join(tmp, "nonexistent-bank")],
            cwd=root, env=env, capture_output=True, encoding="utf-8",
            errors="replace")
        check("a persona with no bank exits non-zero", proc.returncode != 0,
              f"exit {proc.returncode}")
        check("the skip is reported explicitly",
              "SKIPPED" in (proc.stdout + proc.stderr),
              (proc.stdout + proc.stderr)[-200:])


def test_unknown_arm_is_rejected():
    try:
        make_arm("definitely_not_an_arm", Embedder("all-MiniLM-L6-v2"), None,
                 tempfile.gettempdir(), 20, "identifier")
        check("unknown arm names are rejected", False, "no error raised")
    except ValueError:
        check("unknown arm names are rejected", True)


def main():
    print("selftest_run")
    print("-" * 60)
    import logging
    logging.disable(logging.INFO)
    for fn in [
        test_arms_are_comparable_and_causal,
        test_output_format_matches_official_expectations,
        test_dry_run_makes_no_network_calls,
        test_shared_cache_within_one_run,
        test_cache_survives_across_runs,
        test_limit_and_resume,
        test_bank_verification_catches_corruption,
        test_bank_records_provenance,
        test_backend_mismatch_is_refused,
        test_stale_bank_format_is_refused,
        test_analysis_runs_end_to_end,
        test_macro_average_uses_every_persona,
        test_skipped_persona_exits_nonzero,
        test_unknown_arm_is_rejected,
    ]:
        print(f"\n{fn.__name__}:")
        try:
            fn()
        except Exception:
            import traceback
            traceback.print_exc()
            FAILURES.append(fn.__name__)

    print("\n" + "-" * 60)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("harness invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
