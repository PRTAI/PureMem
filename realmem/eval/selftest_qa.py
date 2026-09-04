"""Offline cover for the generation + judging path. No network, no spend.

run_qa_eval.py needs an OpenAI client, so without this file its code path is
never executed until it runs against the real API — where a JSON-parsing or
denominator mistake costs money to discover and looks like a model result
rather than a bug.

A fake client stands in for the LLM. The point is not answer quality, it is:

  * evidence is built with the official formatter and top_k
  * the judge's JSON is parsed the way the official script parses it, including
    the malformed cases
  * the official skip rule holds: no evidence -> no Mem metrics, not a zero
  * every metric reports the denominator it was actually computed over
  * a failed generation is recorded, not silently scored

Run:  python -m eval.selftest_qa
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.run_qa_eval import extract_json, run_generation, run_judge, run_arm
from eval.official_prompts import construct_evidence_text
from eval import schema

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


class FakeClient:
    """Replies based on which prompt it sees. Counts calls, can inject failures."""

    def __init__(self, qa_score=2, mem_recall=0.5, mem_helpful=1,
                 malformed=False, explode_on=None):
        self.calls = {"gen": 0, "qa": 0, "mem": 0}
        self.prompts = []
        outer = self

        class Completions:
            def create(self, model, messages, temperature=0.0, timeout=None, **kw):
                prompt = messages[0]["content"]
                outer.prompts.append(prompt)
                if explode_on and explode_on in prompt:
                    raise RuntimeError("simulated API failure")

                if "Mem_recall" in prompt:
                    outer.calls["mem"] += 1
                    body = ("garbage, not json" if malformed else
                            "```json\n" + json.dumps({
                                "Mem_recall": mem_recall,
                                "Mem_helpful_score": mem_helpful,
                                "Mem_hits": ["a"], "Mem_helpful_reason": "r"}) + "\n```")
                elif "candidate answer" in prompt or "Candidate Answer" in prompt:
                    outer.calls["qa"] += 1
                    body = ("not json at all" if malformed else
                            json.dumps({"score": qa_score, "reason": "because"}))
                else:
                    outer.calls["gen"] += 1
                    body = "generated answer text"

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


def fake_retrieval(n=6, with_evidence=True):
    out = {}
    for i in range(n):
        q = f"question number {i}?"
        items = []
        if with_evidence:
            items = [{"res_type": "chunk", "chunk_id": f"Topic_1:S1_{j:02d}",
                      "content": f"memory content {i}-{j}", "rank": j + 1}
                     for j in range(8)]
        out[q] = {"id": f"Q{i}", "question": q, "ranked_items": items}
    return out


def fake_gt(n=6, annotated=True):
    return {f"question number {i}?": {
        "answer": f"reference answer {i}",
        "memory": (f"gold memory {i}" if annotated else schema.NO_MEMORY_ANNOTATION),
    } for i in range(n)}


# ── Tests ──

def test_json_extraction_matches_official_tolerance():
    cases = [
        ('```json\n{"score": 3}\n```', {"score": 3}),
        ('{"score": 2, "reason": "x"}', {"score": 2, "reason": "x"}),
        ('Here you go: {"score": 1} hope that helps', {"score": 1}),
        ('no json here', None),
        ('', None),
        ('{"broken": ', None),
    ]
    bad = [(t, extract_json(t), e) for t, e in cases if extract_json(t) != e]
    check("judge output parsing handles fenced/bare/malformed", not bad, str(bad[:2]))


def test_generation_uses_official_evidence_format():
    client = FakeClient()
    retrieval = fake_retrieval(3)
    gen = run_generation(retrieval, client, "fake-model", top_k=5, max_workers=2)

    check("one generation per query", len(gen) == 3)
    check("generation calls issued", client.calls["gen"] == 3)

    rec = next(iter(gen.values()))
    check("answer captured", rec["generated_answer"] == "generated answer text")
    check("no error recorded on success", rec["gen_error"] is None)
    check("evidence honours top_k=5", rec["evidence_used"].count("---- idx ") == 5,
          rec["evidence_used"][:80])
    check("evidence uses the official layout",
          rec["evidence_used"].startswith("---- idx 1 ----\n"))
    check("ranked_items are carried into the generation file",
          len(rec["ranked_items"]) == 8)

    prompt = next(p for p in client.prompts if "Memories:" in p)
    check("generation prompt is the official template",
          "personal AI assistant" in prompt and "Response:" in prompt)


def test_generation_failure_is_recorded_not_scored():
    client = FakeClient(explode_on="Memories:")
    gen = run_generation(fake_retrieval(3), client, "fake-model", 5, 2)
    check("failed generations still produce a record", len(gen) == 3)
    check("failure is recorded", all(r["gen_error"] for r in gen.values()))
    check("failed generation yields an empty answer",
          all(r["generated_answer"] == "" for r in gen.values()))

    summary, _ = run_judge(gen, fake_gt(3), client, "fake-model", 5, 2)
    check("summary counts generation failures", summary["n_gen_failed"] == 3)
    check("an empty answer is not given a QA score", summary["n_qa_scored"] == 0)


def test_judge_scores_and_denominators():
    client = FakeClient(qa_score=2, mem_recall=0.75, mem_helpful=2)
    gen = run_generation(fake_retrieval(4), client, "fake-model", 5, 2)
    summary, detailed = run_judge(gen, fake_gt(4), client, "fake-model", 5, 2)

    check("every answer got a QA score", summary["n_qa_scored"] == 4)
    check("every query got Mem metrics", summary["n_mem_scored"] == 4)
    check("QA mean is correct", summary["average_qa_score"] == 2.0)
    check("mem recall mean is correct", summary["average_mem_recall"] == 0.75)
    check("mem helpful mean is correct", summary["average_mem_helpful_score"] == 2.0)
    check("score distribution covers 0-3",
          set(summary["qa_score_distribution"]) == {0, 1, 2, 3})
    check("hallucination rate derived from score 0",
          summary["qa_hallucination_rate"] == 0.0)
    check("detail retained per query", len(detailed) == 4)


def test_no_evidence_means_no_mem_metric_not_zero():
    """The no_memory arm must not appear to 'tie' on mem_recall — the official
    rubric skips the memory prompt entirely when nothing was retrieved."""
    client = FakeClient()
    retrieval = fake_retrieval(4, with_evidence=False)
    gen = run_generation(retrieval, client, "fake-model", 5, 2)
    summary, _ = run_judge(gen, fake_gt(4), client, "fake-model", 5, 2)

    check("QA is still scored without evidence", summary["n_qa_scored"] == 4)
    check("Mem metrics are skipped, not zeroed", summary["n_mem_scored"] == 0)
    check("no average_mem_recall key at all", "average_mem_recall" not in summary)
    check("the mem judge was never called", client.calls["mem"] == 0)


def test_unannotated_gold_skips_mem_metric():
    client = FakeClient()
    gen = run_generation(fake_retrieval(4), client, "fake-model", 5, 2)
    summary, _ = run_judge(gen, fake_gt(4, annotated=False), client, "fake-model", 5, 2)
    check("placeholder gold memory skips the Mem prompt", summary["n_mem_scored"] == 0)
    check("QA unaffected by missing memory annotation", summary["n_qa_scored"] == 4)


def test_malformed_judge_output_is_dropped_not_counted():
    client = FakeClient(malformed=True)
    gen = run_generation(fake_retrieval(4), client, "fake-model", 5, 2)
    summary, _ = run_judge(gen, fake_gt(4), client, "fake-model", 5, 2)
    check("unparseable QA verdicts are not scored as 0", summary["n_qa_scored"] == 0)
    check("unparseable Mem verdicts are not scored as 0", summary["n_mem_scored"] == 0)
    check("no fabricated averages", "average_qa_score" not in summary)


def test_judge_prompts_are_the_official_ones():
    client = FakeClient()
    gen = run_generation(fake_retrieval(2), client, "fake-model", 5, 2)
    run_judge(gen, fake_gt(2), client, "fake-model", 5, 2)

    qa = next(p for p in client.prompts if "Candidate Answer" in p)
    check("QA prompt keeps the anti-plausibility rule", "sounds reasonable" in qa)
    check("QA prompt keeps the 0-3 scale",
          all(f"Score {i}" in qa for i in range(4)))
    check("QA prompt carries all four inputs",
          all(s in qa for s in ("1. Query:", "2. User-related Memory:",
                                "3. Reference Answer:", "4. Candidate Answer:")))

    mem = next(p for p in client.prompts if "Mem_recall" in p)
    check("Mem prompt keeps the three-step recall definition",
          all(s in mem for s in ("step1", "step2", "step3")))
    check("Mem prompt was filled in, not left as a template",
          "{question}" not in mem and "gold memory" in mem)


def test_run_arm_writes_artefacts_and_resumes():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "arm_retrieval_results.json"), "w",
                  encoding="utf-8") as f:
            json.dump(fake_retrieval(3), f)

        client = FakeClient()
        s1 = run_arm("P", "arm", tmp, fake_gt(3), client, "gm", "jm", 5, 2, resume=True)
        check("run_arm returns a summary", s1 and s1["n_qa_scored"] == 3)
        check("generation file written",
              os.path.exists(os.path.join(tmp, "arm_generation_results.json")))
        check("metrics file written",
              os.path.exists(os.path.join(tmp, "arm_llm_metrics.json")))

        with open(os.path.join(tmp, "arm_llm_metrics.json"), "r", encoding="utf-8") as f:
            saved = json.load(f)
        check("metrics file records prompt provenance", "prompts" in saved)
        check("metrics file keeps per-query detail", len(saved["detailed_results"]) == 3)

        before = dict(client.calls)
        run_arm("P", "arm", tmp, fake_gt(3), client, "gm", "jm", 5, 2, resume=True)
        check("resume re-reads metrics without calling the model",
              client.calls == before)

        check("missing retrieval file returns None",
              run_arm("P", "nope", tmp, fake_gt(3), client, "gm", "jm", 5, 2,
                      resume=True) is None)


def test_generation_file_is_readable_by_official_script():
    """The artefact layout the vendored compute_llm_metrics script expects."""
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "arm_retrieval_results.json"), "w",
                  encoding="utf-8") as f:
            json.dump(fake_retrieval(3), f)
        run_arm("P", "arm", tmp, fake_gt(3), FakeClient(), "gm", "jm", 5, 2, resume=False)

        with open(os.path.join(tmp, "arm_generation_results.json"), "r",
                  encoding="utf-8") as f:
            gen = json.load(f)
        rec = next(iter(gen.values()))
        for field in ("question", "generated_answer", "ranked_items", "evidence_used"):
            check(f"generation record has '{field}'", field in rec)
        check("keyed by question text", next(iter(gen)) == rec["question"])


def main():
    print("selftest_qa")
    print("-" * 60)
    import logging
    logging.disable(logging.WARNING)
    for fn in [
        test_json_extraction_matches_official_tolerance,
        test_generation_uses_official_evidence_format,
        test_generation_failure_is_recorded_not_scored,
        test_judge_scores_and_denominators,
        test_no_evidence_means_no_mem_metric_not_zero,
        test_unannotated_gold_skips_mem_metric,
        test_malformed_judge_output_is_dropped_not_counted,
        test_judge_prompts_are_the_official_ones,
        test_run_arm_writes_artefacts_and_resumes,
        test_generation_file_is_readable_by_official_script,
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
    print("generation and judging paths hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
