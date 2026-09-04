# PureMem

**Learner-relative memory curation for reliable retrieval-augmented reasoning.**

Retrieval-augmented systems usually ask *which memory is relevant to this query*.
PureMem separates that into three questions that are normally conflated:

| Question | Mechanism |
|---|---|
| What should the model remember? | ZPD source curation — keep source problems the frozen student solves *sometimes* |
| What memory is potentially relevant? | Three-stage retrieval over a precedent view and a schema view |
| What memory should be injected *now*? | An LLM applicability gate; inject nothing rather than something misfitting |

The third one is the part that most affects reliability. A precedent that is
topically similar but demonstrates the wrong mechanism is worse than no
precedent at all, because it hands the solver a confident wrong plan. So the
gate is deliberately conservative: when every candidate is refused, the prompt
falls back to a **byte-identical** copy of the no-memory baseline prompt
(asserted at startup by `prompts.selftest_arm_parity()`), which is what makes
the with/without-memory comparison a genuinely paired one.

This repository contains the harness for both evaluation tracks in the paper:
learner-relative curation on **BBEH**, and transfer of the retrieval
architecture alone to long-term conversational memory on **RealMemBench**.

---

## Repository layout

```
bbeh/                  BBEH track — ZPD curation, memory construction, 3-stage retrieval
  config.py            all knobs; every one has an environment override
  probe.py             difficulty probe: k stochastic student attempts per item
  selector.py          source selection (zpd / random_matched / stratified_matched / ...)
  teacher.py           rejection-sampled, answer-verified solution traces
  abstract.py          step triples -> abstract schemas + mechanism labels
  build_memory.py      assemble a memory version (no API calls)
  retriever.py         Stage I recall + Stage II structured matching
  reranker.py          Stage III LLM approach-fit judge, sampled and voted
  reasoner.py          the two solve arms, sharing one code path on purpose
  run.py               evaluate one arm
  analyze.py           equivalence / superiority tests across arms
  official_eval.py     BBEH's own scorer
  selftest_*.py        offline invariant checks (no API key needed)

realmem/               RealMemBench track — overlay for the upstream repo (see below)
  eval/
    three_stage_retriever.py   the same three stages over conversational sessions
    build_memory.py            session bank + abstracts
    metrics.py                 Recall@k / NDCG@k, re-implemented
    verify_against_official.py cross-checks metrics.py against the upstream scorer
    analyze.py, analyze_qa.py  aggregation and paired tests
    ablate_from_results.py     offline stage attribution (read its warnings)
  run_parallel.py              one process per persona; defines the dev/test split

figure_script/         regenerate the paper's figures
scripts/
  recompute_table2.py   counterfactual placebo table from run artifacts
```

## Install

```bash
pip install -r requirements.txt
```

Credentials come from the environment. **There is no default key**; an empty one
fails on the first request rather than silently authenticating as someone else.

```bash
export BBEH_API_KEY=sk-...                       # required
export BBEH_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible gateway
export BBEH_STUDENT_MODEL=gpt-4o-mini            # optional; see below
```

Before spending anything, run the offline selftests — they need only numpy and
no API access:

```bash
python -m bbeh.reranker              # judge-reply parser invariants
python -m bbeh.selftest_retrieval    # retrieval invariants
python -m bbeh.selftest_run          # arm-parity and resume invariants
cd realmem && python -m eval.selftest_retrieval
```

Then check connectivity, which costs a few tokens and tells you *which* of
key / auth-scheme / host-path is wrong instead of just failing:

```bash
python -m bbeh.pilot ping
python -m bbeh.pilot ping --sweep    # endpoint x auth-scheme matrix
```

## Getting the data

Neither benchmark is redistributed here; both are third-party and separately
licensed.

**BBEH** — clone [BIG-Bench Extra Hard](https://github.com/google-deepmind/bbeh)
into the repository root as `bbeh-main/`, so that
`bbeh-main/bbeh/benchmark_tasks/<task>/task.json` resolves. The harness treats
that tree as strictly read-only.

**RealMemBench** — the `realmem/` directory is an **overlay**, not a standalone
package. Clone the upstream RealMemBench repository, then copy `realmem/eval/*.py`
into its `eval/` and `realmem/run_parallel.py` into its root.

The overlay is necessary rather than incidental: `eval/official_prompts.py`
extracts the judge prompts and two generation helpers directly from the upstream
`compute_llm_metrics_for_realmem.py` and `run_generation.py` by parsing their
AST at import time. The judge prompt *is* the metric, so a hand-copied duplicate
would drift silently — an early attempt at exactly that was already six
characters off. Retrieval and metrics run standalone; the QA track needs the
upstream files present.

## BBEH track

```bash
# 1. deterministic, disjoint train/test split (seed 42; enforced by id AND input hash)
python -m bbeh.data build-splits
python -m bbeh.data verify                  # must print VERDICT: OK

# 2. difficulty probe — k stochastic attempts per train item, at temperature 1.0
#    (a deterministic decode cannot produce intermediate pass rates)
python -m bbeh.probe --k 5

# 3. answer-verified teacher traces for the ZPD band
python -m bbeh.teacher --select zpd --max-attempts 3

# 4. abstract each step into a schema + mechanism label
python -m bbeh.abstract

# 5. assemble memory versions (no API calls)
python -m bbeh.build_memory build --version-id zpd                --method zpd
python -m bbeh.build_memory build --version-id full               --method full
python -m bbeh.build_memory build --version-id random_matched     --method random_matched     --match-version zpd
python -m bbeh.build_memory build --version-id stratified_matched --method stratified_matched --match-version zpd
python -m bbeh.build_memory list

# 6. evaluate. --limit-per-task 30 is the paper's 690-instance subset;
#    omitting it evaluates all 2260 test items.
python -m bbeh.run --arm no_memory                          --limit-per-task 30
python -m bbeh.run --arm memory --memory-version zpd        --limit-per-task 30
python -m bbeh.run --arm memory --memory-version full       --limit-per-task 30

# gate and retrieval ablations
python -m bbeh.run --arm memory --memory-version zpd --limit-per-task 30 \
    --arm-label memory_zpd_nogate --no-reranker
python -m bbeh.run --arm memory --memory-version zpd --limit-per-task 30 \
    --arm-label memory_zpd_randret_gate --random-retrieval

# 7. aggregate
python -m bbeh.analyze
```

Two things about resuming that are easy to get wrong:

- `results.jsonl` **is** the cache. A record counts as done if it is scorable
  and non-empty — which includes **truncated** records. So re-running with a
  larger `--max-tokens` will *skip everything* and still rewrite `summary.json`
  with the new budget recorded against the old numbers. Pass
  `--no-skip-existing` to force a real re-solve (it deletes `results.jsonl` and
  `memory_injections.jsonl` first, so back them up).
- Infra errors are *not* cached as failures; they are retried on the next run
  and excluded from the accuracy denominator. A proxy timeout must never become
  a permanent zero.

## RealMemBench track

Run from the upstream repository root after applying the overlay:

```bash
python -m eval.build_memory --all-personas
python -m eval.run_eval --all-personas --arms no_memory,simple_embedding,three_stage_rerank,three_stage_gated
python -m eval.run_qa_eval --all-personas          # needs the upstream QA scripts
python -m eval.analyze --all-personas
python -m eval.verify_against_official             # metrics.py vs the upstream scorer
```

`run_parallel.py` fixes the evaluation split — two development personas
(`Lin_Wanyu`, `Adeleke_Okonjo`), chosen before any results were seen and used
for every configuration decision, and eight held-out test personas. Report
held-out numbers from the eight, and say so when a table spans all ten.

`ablate_from_results.py` reconstructs the intermediate `stage2_only` /
`concrete_only` rankings offline from recorded Stage-II ranks. Read its
docstring before citing the output: because the reconstruction and the full arm
are scored over the same returned depth, `Recall@k` for `k >= RETRIEVE_K` is
forced to coincide *by construction*, so that table cannot answer whether Stage
III adds new candidates. Run a real `stage2_only` arm for that.

## Default configuration

Every value below is an environment override away (`BBEH_*` / `REALMEM_*`); see
`bbeh/config.py` and `realmem/eval/config.py` for the reasoning behind each.

| | BBEH | RealMemBench |
|---|---|---|
| Encoder | `all-MiniLM-L6-v2`, 384-d, L2-normalized | same |
| Long-input handling | head+tail window, 900 chars each end | same |
| ZPD band | `m=5` attempts, keep `0.2 <= p <= 0.8` | not applicable |
| Teacher | ≤3 rejection-sampling attempts, answer-verified | not applicable |
| Stage I recall | 6 schemas; 8 source questions, then their steps | top-40 sessions |
| Stage II bonuses | task family `+0.15`, mechanism `+0.10` (additive) | abstract `+0.15`, topic `+0.10` |
| Stage II shortlist | `K'=5` | `K'=25` |
| Stage III | `N=5` samples, `tau=0.7`, `v=2` votes, temp 1.0 | `N=3`, `tau=0.7`, `v=2` |
| Injected | up to `K=2` precedents | top-5 sessions to the generator |

Stage II bonuses are additive and never filter. A hard same-task restriction
would quietly turn the system into a per-task few-shot retriever and make
cross-task mechanism transfer inexpressible — which is the effect worth
measuring, and the only thing Stage III is qualified to rule on.

A test query has no trace of its own and therefore no mechanism label, so the
expected mechanism set is taken from the empirical mechanism distribution of its
task family among the *training* precedents. This uses no test-side annotation.

Stage III scores an entire shortlist in one call, so the `N` samples are `N`
calls, not `N x K'`. Scores for different candidates within one sample are
therefore not independent — they come from the same generation. Samples are
cached keyed by (rubric hash, query excerpt, candidate identity) and
deliberately **not** by `tau` or `v`, so those thresholds can be re-swept over
fixed samples for free. Judge calls lost to infrastructure failures are dropped
rather than recorded as zero: scoring a timeout as "irrelevant" would freeze a
transient fault into a permanent veto, whereas an incomplete sample set simply
cannot reach `v` and degrades toward the no-memory fallback.

## Reproducing the figures

```bash
python figure_script/plot_bbeh_pareto.py
python figure_script/plot_gate_counterfactual.py
python scripts/recompute_table2.py --permutations 20000 --bootstrap 10000
```

The two plotting scripts carry their numbers in an editable block at the top,
with the provenance recorded in the docstring. `recompute_table2.py` derives
everything from run artifacts and reports the counterfactual table under both
definitions of the gate label, because they disagree on 129 of 690 queries and
"we froze the gate decision" is only well defined if they agree.

## Notes on honest evaluation

A few invariants in this codebase exist because violating them produces
plausible-looking numbers rather than errors. They are worth preserving in any
fork:

- **Arm parity is asserted, not assumed.** `selftest_arm_parity()` runs on every
  invocation of `run.py` and checks that the memory prompt with an empty slot is
  byte-identical to the baseline prompt.
- **Dry runs cannot poison real caches.** Fabricated scores refuse to be written
  to a non-`DRYRUN` cache path; a fabricated probe result would otherwise be read
  back as measured on the next real run.
- **Misalignment raises instead of truncating.** A memory version whose
  embeddings and records disagree in length is refused rather than trimmed to the
  shorter of the two, because trimming converts a corrupt bank into a
  *misaligned* one and hides it behind confident scores.
- **A collapsed abstractor is detected explicitly.** An abstractor told to strip
  everything specific reduces every mechanism to "gather the items, combine them,
  get the answer", after which Stage III cannot discriminate at all. The build
  reports distinct-action ratio and a degenerate-phrase rate and says so out loud.

## Citation

```bibtex
@misc{puremem,
  title  = {PureMem: Learner-Relative Memory Curation for Reliable
            Retrieval-Augmented Reasoning},
  author = {TODO},
  year   = {2026},
  note   = {Preprint}
}
```

## License and acknowledgements

The code in this repository is released under the MIT License (see `LICENSE`).

The benchmarks are third-party and are **not** redistributed here:

- **BIG-Bench Extra Hard (BBEH)** — Google DeepMind, Apache-2.0. Obtain it from
  the upstream repository; this harness treats it as read-only.
- **RealMemBench** — Apache-2.0. The `realmem/` overlay depends on the upstream
  evaluation scripts as the reference definition of the QA metrics and cross-checks
  its own metric implementation against them (`eval/verify_against_official.py`).

Please cite both benchmarks alongside this work when reporting results on them.
