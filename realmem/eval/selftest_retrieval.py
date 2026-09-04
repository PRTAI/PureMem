"""Offline invariants for the retriever. No network, no torch, no spend.

Runs on fabricated sessions with the hash embedder, so it exercises ranking
logic rather than embedding quality. The point is the contracts:

  * a query never sees the current or any later session   <- the P0 regression
  * Stage 1 reweights and never removes
  * Stage 3 'rerank' keeps depth; Stage 3 'gate' may return nothing
  * the judge cache makes a rerun free and identical

Run:  python -m eval.selftest_retrieval
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.embedding import Embedder, BACKEND_HASH
from eval.three_stage_retriever import ThreeStageRetriever, SimpleEmbeddingRetriever
from eval import schema

FAILURES = []


def check(name: str, cond: bool, detail: str = ""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


# ── Fixtures ──

TOPICS = ["Travel_Planning", "Fitness", "Knowledge_Learning"]


def make_session(i: int, topic=None, n_mem: int = 2, text: str = None):
    """topic=None produces an 'Enhanced' filler session with no extracted_memory,
    matching the 48% of the real corpus that is never gold."""
    if topic:
        sid = f"{topic}_1:S1_{i:02d}"
        mems = [{"index": f"{topic}-DM-S1_{i:02d}-{j:02d}", "type": "Dynamic",
                 "content": f"{topic} memory point {j} for session {i}"}
                for j in range(n_mem)]
    else:
        sid = f"Enhanced:S1{i:04d}"
        mems = []
    body = text or f"discussion about {topic or 'unrelated filler chatter'} number {i}"
    return {
        "session_identifier": sid,
        "session_uuid": f"uuid-{i}",
        "current_time": f"2025-12-{(i % 28) + 1:02d} (Monday)",
        "extracted_memory": mems,
        "dialogue_turns": [
            {"speaker": "User", "content": body, "is_query": False},
            {"speaker": "Assistant", "content": f"reply about {body}", "is_query": False},
        ],
    }


def corpus(n: int = 24):
    out = []
    for i in range(n):
        topic = TOPICS[i % 3] if i % 4 else None   # every 4th is filler
        out.append(make_session(i, topic))
    return out


class FakeCompletions:
    def __init__(self, parent):
        self.parent = parent

    def create(self, model, messages, temperature=0.0, timeout=None, **kw):
        self.parent.calls += 1
        prompt = messages[0]["content"]
        n = prompt.count("\n[") + (1 if "\n[1] " in prompt or prompt.count("[1] ") else 0)
        n = max(n, 1)
        # Count candidates by scanning for the bracketed indices we emit.
        idx = 1
        fits = []
        while f"[{idx}]" in prompt:
            fits.append({"idx": idx, "fit": self.parent.fit_for(idx)})
            idx += 1
        body = json.dumps(fits)

        class M:
            content = body

        class C:
            message = M()

        class R:
            choices = [C()]

        return R()


class FakeClient:
    """Deterministic stand-in for the Stage-3 judge."""

    def __init__(self, fit=0.9):
        self.calls = 0
        self._fit = fit

        class Chat:
            pass

        self.chat = Chat()
        self.chat.completions = FakeCompletions(self)

    def fit_for(self, idx: int) -> float:
        return self._fit(idx) if callable(self._fit) else self._fit


def build(**kw):
    emb = Embedder("all-MiniLM-L6-v2")
    return ThreeStageRetriever(embedder=emb, **kw), emb


# ── Tests ──

def test_no_temporal_leakage():
    """The defect this whole rewrite exists for.

    Gold in RealMemBench points backwards 100% of the time, so a retriever that
    can see the current or a future session is both cheating and worse.
    """
    r, _ = build(enable_stage3=False, retrieve_k=20)
    sessions = corpus(24)
    admitted = set()
    violations = []

    for s in sessions:
        items = r.retrieve("tell me about my travel plans",
                           query_topic=schema.parse_topic(s["session_identifier"]))
        for it in items:
            if it["chunk_id"] not in admitted:
                violations.append((s["session_identifier"], it["chunk_id"]))
        r.add_session(s)
        admitted.add(s["session_identifier"])

    check("retrieval never returns an unadmitted session", not violations,
          f"{len(violations)} leaks, e.g. {violations[:2]}")

    check("first query returns nothing (empty history)",
          len(ThreeStageRetriever(embedder=Embedder('all-MiniLM-L6-v2'),
                                  enable_stage3=False).retrieve("anything")) == 0)


def test_reset_clears_state():
    r, _ = build(enable_stage3=False)
    for s in corpus(8):
        r.add_session(s)
    check("sessions admitted", r.n_sessions == 8)
    r.reset()
    check("reset clears admitted sessions", r.n_sessions == 0)
    check("retrieval after reset is empty", r.retrieve("q") == [])


def test_stage1_is_additive_not_a_filter():
    """Stage 1 reweights; it must never shrink the candidate set. A cross-topic
    precedent has to survive to Stage 3, which is the only stage entitled to
    rule it out."""
    sessions = corpus(24)

    with_s1, _ = build(enable_stage1=True, enable_stage3=False, retrieve_k=24, pool_size=40)
    without_s1, _ = build(enable_stage1=False, enable_stage3=False, retrieve_k=24, pool_size=40)
    for s in sessions:
        with_s1.add_session(s)
        without_s1.add_session(s)

    a = {i["chunk_id"] for i in with_s1.retrieve("travel plans", query_topic="Travel_Planning")}
    b = {i["chunk_id"] for i in without_s1.retrieve("travel plans", query_topic="Travel_Planning")}
    check("Stage 1 does not drop candidates", a == b,
          f"only in one side: {a ^ b}")

    ranked = with_s1.retrieve("travel plans", query_topic="Travel_Planning")

    # Mechanism: the bonus is exactly the sum of the configured weights for the
    # tags each candidate carries. Deterministic, encoder-independent.
    from eval.config import (STAGE1_ABSTRACT_WEIGHT, STAGE1_TOPIC_WEIGHT,
                             STAGE1_RECENCY_WEIGHT)
    if not STAGE1_RECENCY_WEIGHT:
        wrong = [(i["chunk_id"], i["stage1_bonus"]) for i in ranked
                 if abs(i["stage1_bonus"]
                        - (STAGE1_ABSTRACT_WEIGHT * bool(i["has_abstract"])
                           + STAGE1_TOPIC_WEIGHT * bool(i["same_topic"]))) > 1e-4]
        check("Stage 1 bonus is exactly the sum of its configured weights",
              not wrong, str(wrong[:3]))

    # Effect: both tagged groups should be over-represented at the head relative
    # to the full ranking. Stated as enrichment rather than an absolute count,
    # because how many make the cut depends on how far apart the encoder spaces
    # the cosine scores — which is exactly what broke the earlier version of the
    # topic-ablation test on real MiniLM.
    head = ranked[:8]

    def enrichment(field):
        return (sum(1 for i in head if i[field]) / len(head),
                sum(1 for i in ranked if i[field]) / len(ranked))

    h_abs, o_abs = enrichment("has_abstract")
    check("memory-bearing sessions are enriched at the head",
          h_abs >= o_abs, f"head={h_abs:.3f} overall={o_abs:.3f}")

    h_top, o_top = enrichment("same_topic")
    check("same-topic sessions are enriched at the head",
          h_top >= o_top, f"head={h_top:.3f} overall={o_top:.3f}")


def test_topic_source_none_ablates():
    """Assert the mechanism, not a particular ordering.

    Whether removing a 0.10 bonus actually reorders anything depends on how far
    apart the cosine scores happen to be, which depends on the encoder. An
    earlier version of this test asserted the order changed; that held under the
    hash fallback and failed under real MiniLM, where these templated fixture
    sentences cluster so tightly by topic that the bonus cannot cross the gap.
    Ordering was never the contract — the bonus being applied is.
    """
    from eval.config import STAGE1_TOPIC_WEIGHT
    sessions = corpus(24)

    off, _ = build(enable_stage3=False, topic_source="none", retrieve_k=24, pool_size=40)
    on, _ = build(enable_stage3=False, topic_source="identifier", retrieve_k=24, pool_size=40)
    for s in sessions:
        off.add_session(s)
        on.add_session(s)

    items_off = off.retrieve("travel plans", query_topic="Travel_Planning")
    items_on = on.retrieve("travel plans", query_topic="Travel_Planning")

    check("topic_source='none' keeps the same candidate set",
          {i["chunk_id"] for i in items_off} == {i["chunk_id"] for i in items_on})

    bonus_off = {i["chunk_id"]: i["stage1_bonus"] for i in items_off}
    bonus_on = {i["chunk_id"]: i["stage1_bonus"] for i in items_on}
    same_topic = [i["chunk_id"] for i in items_on if i["same_topic"]]
    check("fixture actually contains same-topic sessions", len(same_topic) > 0)

    check("topic bonus is applied only when topic_source='identifier'",
          all(abs((bonus_on[c] - bonus_off[c]) - STAGE1_TOPIC_WEIGHT) < 1e-4
              for c in same_topic),
          f"deltas: {[round(bonus_on[c] - bonus_off[c], 4) for c in same_topic[:3]]}")

    off_topic = [i["chunk_id"] for i in items_on if not i["same_topic"]]
    check("cross-topic candidates are unaffected by the ablation",
          all(abs(bonus_on[c] - bonus_off[c]) < 1e-4 for c in off_topic))

    # The bonus can only help same-topic sessions, so their share of the head of
    # the ranking cannot go down. Monotone, not strict — see the docstring.
    def share(items, k=8):
        head = items[:k]
        return sum(1 for i in head if i["same_topic"]) / max(len(head), 1)

    check("same-topic share in the head does not decrease",
          share(items_on) >= share(items_off) - 1e-9,
          f"on={share(items_on):.3f} off={share(items_off):.3f}")


def test_depth_reaches_metric_k():
    """recall@20 over a list of 15 is really recall@15. The old config had
    POOL_SIZE=15 < RETRIEVE_K=20 and shipped exactly that."""
    r, _ = build(enable_stage3=False, retrieve_k=20, pool_size=40, top_n_sessions=40)
    for s in corpus(60):
        r.add_session(s)
    items = r.retrieve("travel", query_topic="Travel_Planning")
    check("returns RETRIEVE_K items when the corpus allows", len(items) == 20,
          f"got {len(items)}")
    check("ranks are 1..k contiguous",
          [i["rank"] for i in items] == list(range(1, len(items) + 1)))
    check("chunk_ids are unique", len({i["chunk_id"] for i in items}) == len(items))


def test_stage3_rerank_preserves_depth():
    """Even when the judge refuses everything, 'rerank' must keep the list full."""
    client = FakeClient(fit=0.0)          # refuse every candidate
    r, _ = build(enable_stage3=True, gate_mode="rerank", retrieve_k=20,
                 pool_size=40, top_n_sessions=40, llm_client=client, stage3_samples=3)
    for s in corpus(60):
        r.add_session(s)
    items = r.retrieve("travel", query_topic="Travel_Planning")
    check("rerank mode keeps depth even when the judge refuses all",
          len(items) == 20, f"got {len(items)}")
    check("refused candidates are marked, not dropped",
          all(i["stage3_passed"] is False for i in items[:5]))


def test_stage3_gate_can_return_empty():
    """Upstream semantics preserved: refuse everything -> no memory at all."""
    client = FakeClient(fit=0.0)
    r, _ = build(enable_stage3=True, gate_mode="gate", retrieve_k=20,
                 pool_size=40, llm_client=client, stage3_samples=3)
    for s in corpus(30):
        r.add_session(s)
    check("gate mode returns empty when nothing clears tau",
          r.retrieve("travel", query_topic="Travel_Planning") == [])
    check("gated_empty counter incremented", r.stats["gated_empty"] == 1)

    accept = FakeClient(fit=0.95)
    r2, _ = build(enable_stage3=True, gate_mode="gate", retrieve_k=20,
                  pool_size=40, llm_client=accept, stage3_samples=3)
    for s in corpus(30):
        r2.add_session(s)
    items = r2.retrieve("travel", query_topic="Travel_Planning")
    check("gate mode returns survivors when the judge accepts", len(items) > 0)
    check("survivors carry vote counts",
          all(i["fit_votes"] == 3 for i in items), f"{[i['fit_votes'] for i in items[:3]]}")


def test_gate_without_judge_is_not_a_free_pass():
    """A missing judge must not silently mean 'everything approved' — that would
    turn an outage into a fabricated result."""
    r, _ = build(enable_stage3=True, gate_mode="gate", llm_client=None, retrieve_k=20)
    for s in corpus(20):
        r.add_session(s)
    check("gate mode with no judge returns empty rather than everything",
          r.retrieve("travel", query_topic="Travel_Planning") == [])


def test_concurrent_sampling_matches_serial():
    """The N judge draws run concurrently. That must change timing only.

    Verified against sample_workers=1, which is the pre-concurrency code path.
    """
    sessions = corpus(20)

    def run(workers, fit):
        client = FakeClient(fit=fit)
        r, _ = build(enable_stage3=True, gate_mode="rerank", retrieve_k=10,
                     pool_size=20, llm_client=client, stage3_samples=3,
                     sample_workers=workers)
        for s in sessions:
            r.add_session(s)
        items = r.retrieve("travel", query_topic="Travel_Planning")
        return items, client.calls, r.stats

    # Deterministic judge: concurrent and serial must agree exactly.
    par, par_calls, par_stats = run(3, 0.9)
    ser, ser_calls, ser_stats = run(1, 0.9)

    check("concurrent sampling issues the same number of calls",
          par_calls == ser_calls == 3, f"par={par_calls} ser={ser_calls}")
    check("concurrent sampling yields the same ranking",
          [i["chunk_id"] for i in par] == [i["chunk_id"] for i in ser])
    check("concurrent sampling yields the same vote counts",
          [i["fit_votes"] for i in par] == [i["fit_votes"] for i in ser],
          f"par={[i['fit_votes'] for i in par][:3]} ser={[i['fit_votes'] for i in ser][:3]}")
    check("call counter is accurate under concurrency",
          par_stats["stage3_calls"] == 3, str(par_stats))
    check("no phantom errors recorded", par_stats["stage3_errors"] == 0)


def test_concurrent_sampling_counts_errors():
    """A failing judge must be counted once per failed draw, not lost in a thread."""
    class Exploding(FakeClient):
        def __init__(self):
            super().__init__(fit=0.5)
            outer = self

            class Completions:
                def create(self, *a, **kw):
                    outer.calls += 1
                    raise RuntimeError("simulated judge outage")

            class Chat:
                completions = Completions()

            self.chat = Chat()

    client = Exploding()
    r, _ = build(enable_stage3=True, gate_mode="rerank", retrieve_k=10,
                 pool_size=20, llm_client=client, stage3_samples=3, sample_workers=3)
    for s in corpus(20):
        r.add_session(s)
    items = r.retrieve("travel", query_topic="Travel_Planning")

    check("every failed draw is counted", r.stats["stage3_errors"] == 3, str(r.stats))
    check("a total judge outage still returns candidates in rerank mode",
          len(items) == 10, f"got {len(items)}")
    check("no candidate is credited with a vote after an outage",
          all(i["fit_votes"] == 0 for i in items))


def test_rerank_cache_makes_rerun_free():
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "rerank_cache.jsonl")
        client = FakeClient(fit=0.9)
        r, emb = build(enable_stage3=True, gate_mode="rerank", retrieve_k=10,
                       pool_size=20, llm_client=client, stage3_samples=3,
                       cache_path=cache)
        for s in corpus(20):
            r.add_session(s)
        first = r.retrieve("travel", query_topic="Travel_Planning")
        calls_after_first = client.calls
        check("judge was called once per sample", calls_after_first == 3,
              f"got {calls_after_first}")

        r2 = ThreeStageRetriever(embedder=emb, enable_stage3=True, gate_mode="rerank",
                                 retrieve_k=10, pool_size=20, llm_client=client,
                                 stage3_samples=3, cache_path=cache)
        for s in corpus(20):
            r2.add_session(s)
        second = r2.retrieve("travel", query_topic="Travel_Planning")

        check("cached rerun issues no new judge calls", client.calls == calls_after_first,
              f"{client.calls} vs {calls_after_first}")
        check("cached rerun is identical",
              [i["chunk_id"] for i in first] == [i["chunk_id"] for i in second])


def test_simple_embedding_arm_is_plain_cosine():
    """The baseline must be a pure similarity ranking.

    An earlier version of this check was vacuous — `abs(x) >= 0` and
    `not X or True` are both tautologies, so it passed without testing anything.
    """
    from eval.config import CONCRETE_W, ABSTRACT_W
    sessions = corpus(30)

    base = SimpleEmbeddingRetriever(embedder=Embedder("all-MiniLM-L6-v2"), retrieve_k=20)
    full, _ = build(enable_stage3=False, retrieve_k=20, pool_size=40)
    for s in sessions:
        base.add_session(s)
        full.add_session(s)

    items = base.retrieve("travel", query_topic="Travel_Planning")
    check("baseline applies zero Stage-1 bonus",
          all(i["stage1_bonus"] == 0.0 for i in items),
          f"nonzero: {[i['stage1_bonus'] for i in items if i['stage1_bonus']][:3]}")

    # Score must be exactly the blended similarity, with nothing added.
    pool = base._stage2_pool("travel", "Travel_Planning")
    check("baseline score is exactly the similarity blend",
          all(abs(c["stage2_score"]
                  - (CONCRETE_W * c["concrete_sim"] + ABSTRACT_W * c["abstract_sim"])) < 1e-5
              for c in pool))

    three = full.retrieve("travel", query_topic="Travel_Planning")
    check("the three-stage arm, by contrast, does apply a bonus",
          any(i["stage1_bonus"] > 0 for i in three))

    check("baseline records no Stage-3 verdict",
          all(i["fit_votes"] is None and i["stage3_passed"] is None for i in items))


def test_hash_backend_is_labelled():
    emb = Embedder("all-MiniLM-L6-v2")
    if emb.backend == BACKEND_HASH:
        check("hash fallback is labelled as such", emb.backend == BACKEND_HASH)
    else:
        print("  skip  hash fallback labelling (sentence-transformers installed)")


def main():
    print("selftest_retrieval")
    print("-" * 60)
    for fn in [
        test_no_temporal_leakage,
        test_reset_clears_state,
        test_stage1_is_additive_not_a_filter,
        test_topic_source_none_ablates,
        test_depth_reaches_metric_k,
        test_stage3_rerank_preserves_depth,
        test_stage3_gate_can_return_empty,
        test_gate_without_judge_is_not_a_free_pass,
        test_concurrent_sampling_matches_serial,
        test_concurrent_sampling_counts_errors,
        test_rerank_cache_makes_rerun_free,
        test_simple_embedding_arm_is_plain_cosine,
        test_hash_backend_is_labelled,
    ]:
        print(f"\n{fn.__name__}:")
        try:
            fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            FAILURES.append(f"{fn.__name__} raised {type(exc).__name__}")

    print("\n" + "-" * 60)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("all retrieval invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
