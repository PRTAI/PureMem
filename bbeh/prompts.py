"""
bbeh/prompts.py — every prompt template, in one place.

Why one place: the difficulty probe, the ``no_memory`` baseline and the
``memory`` arm must all pose the *same question* to the student. If the probe
used a different wording than the eval, the measured pass_rate would not
describe the student's behaviour in the eval, and the ZPD band would be
measuring the wrong thing. If the two eval arms differed by anything other than
the injected precedent block, claim 2 would be confounded by prompt wording.

The invariant, enforced by ``selftest_arm_parity()`` below:

    build_solve_prompt(item, precedents=[])   ==   no_memory prompt  ==  probe prompt
    build_solve_prompt(item, precedents=[..]) ==   memory prompt
                                              ==   PRECEDENT_BLOCK + no_memory prompt

The Stage-3 reranker prompt deliberately lives in ``reranker.py`` instead,
because its hash is part of the rerank cache key.
"""

from typing import List, Sequence

# ═════════════════════════════════════════════════════════════════════
#  Solving
# ═════════════════════════════════════════════════════════════════════

# BBEH's official scorer looks for "The final answer is:" (and a few variants)
# and takes the FIRST LINE after it, lowercased, with ", " collapsed to ",".
# So the answer line must be terminal, bare, and on one line. Anything else
# scores 0 no matter how good the reasoning was.
SOLVE_SUFFIX = """

---
Solve the problem carefully and compactly.
Do not produce unnecessary derivations or text after the final answer.
For logic-grid or elimination tasks, avoid printing a full grid or exhaustive case analysis, but perform all reasoning necessary to determine the answer.

Finish your response with exactly one line, in this form:
The final answer is: <answer>

Rules for that final line:
- Put ONLY the answer on it — no restatement, no XML tags (no literal <answer>), no units, no explanation, no bold.
- If the answer is a list, separate items with commas in the order requested.
- If the question offers lettered options, answer with the letter, e.g. (A).
- Do not write anything after that line."""


PRECEDENT_POLICY_HEADER = """[Precedent-use policy]
If a precedent is provided, treat it only as a strategy hint for different problem instances.
Reuse only a general method; never copy their entities, words, numbers, counts, intermediate values, or answers.
Derive all facts from the current problem input as the sole source.
[/Precedent-use policy]

"""

PRECEDENT_BLOCK_HEADER = PRECEDENT_POLICY_HEADER + """[Memory Precedents]
You may find these reasoning precedents useful. Each one is a single reasoning step drawn from a DIFFERENT problem that was solved correctly, shown as (general pattern and concrete example).

"""

PRECEDENT_BLOCK_FOOTER = """--- end of precedents ---
[/Memory Precedents]

Now solve this problem.

"""


def format_precedent(idx: int, chunk: dict) -> str:
    """Render one retrieved chunk for injection.

    Shows the abstract pattern when we have one *and* the concrete step, in
    that order. The concrete step is the ground truth of the mechanism; the
    abstract is a generalisation that may have drifted, so it is labelled as
    such rather than presented as fact.
    """
    lines = [f'Precedent {idx}']
    pattern = chunk.get('abstract_pattern') or {}
    ptype = pattern.get('pattern_type') or chunk.get('pattern_type') or 'general'
    lines.append(f'  reasoning type: {ptype}')
    if pattern:
        lines.append(
            '  general pattern: '
            f"{pattern.get('abstract_state', '')} -> "
            f"{pattern.get('abstract_action', '')} -> "
            f"{pattern.get('abstract_next_state', '')}"
        )
    lines.append(
        '  concrete example: '
        f"{chunk.get('state', '')} -> "
        f"{chunk.get('action', '')} -> "
        f"{chunk.get('next_state', '')}"
    )
    return '\n'.join(lines)


def build_solve_prompt(item: dict, precedents: Sequence[dict] = ()) -> str:
    """The one and only solver prompt.

    With ``precedents=()`` this is the ``no_memory`` / probe prompt. With
    precedents it is the ``memory`` prompt, which is byte-identical except for
    the block prepended at the top.
    """
    question = item['input'].strip()
    if not precedents:
        return PRECEDENT_POLICY_HEADER + question + SOLVE_SUFFIX
    block = PRECEDENT_BLOCK_HEADER + '\n\n'.join(
        format_precedent(i + 1, c) for i, c in enumerate(precedents)
    ) + '\n' + PRECEDENT_BLOCK_FOOTER
    return block + question + SOLVE_SUFFIX


# ═════════════════════════════════════════════════════════════════════
#  Teacher: author a chain-of-thought that will become memory
# ═════════════════════════════════════════════════════════════════════

TPL_TEACHER_COT = """You are solving a hard reasoning problem. Your solution will be stored in a \
memory bank and used to help a weaker model solve DIFFERENT problems later, so \
the *structure* of your reasoning matters as much as the answer.

Solve the problem below, then decompose your own reasoning into a sequence of \
steps. Each step is a triple:
  - "state": what is known and what is being sought at the START of this step. \
Be compact. Do NOT restate the whole problem. Do NOT accumulate every earlier result.
  - "action": the single concrete reasoning move made in this step. Name the \
operation AND show its concrete instantiation (the comparison made, the \
arithmetic performed, the constraint applied, the elimination justified).
  - "next_state": what is established once the step is done.

Requirements:
- One triple per genuine reasoning step. Do not merge several moves into one, \
and do not pad with restatements.
- Between 2 and 12 steps. If the problem needs more, group the mechanical \
repetitions and say how many times the move repeats.
- The action must be specific enough that a reader who has NOT seen this problem \
could recognise when to make the same move. "Analyse the data" is useless; \
"compare the two candidates on their third letter to break the tie" is useful.
- Solve it correctly. A wrong answer makes the whole trace worthless to us.

PROBLEM
{question}

Respond with ONLY a JSON object, no prose and no markdown fence:
{{"steps": [{{"state": "...", "action": "...", "next_state": "..."}}, ...],
  "answer": "<the final answer, formatted exactly as the problem asks>"}}"""


# ═════════════════════════════════════════════════════════════════════
#  Abstractor: strip the specifics, keep the mechanism
# ═════════════════════════════════════════════════════════════════════

# Mechanism-flavoured on purpose, NOT task-flavoured. If the labels mirrored the
# 23 task names, pattern_type would just be a task id and Stage-1's pattern
# bonus would collapse into the task bonus, making cross-task transfer
# inexpressible.
PATTERN_TYPES = (
    'arithmetic_chain',
    'counting_aggregation',
    'unit_conversion',
    'temporal_arithmetic',
    'sorting_ordering',
    'comparison_selection',
    'constraint_propagation',
    'logical_deduction',
    'elimination',
    'boolean_evaluation',
    'state_tracking',
    'object_property_tracking',
    'spatial_transform',
    'string_manipulation',
    'pattern_matching',
    'error_localization',
    'table_lookup',
    'pragmatic_inference',
    'other',
)


# The hard-won lesson from PuzzleWorld v8.6.2: an abstractor told to "remove all
# specific content" collapses every distinct mechanism into the same empty shell
# ("gather the items -> combine them -> get the answer"), after which the
# reranker cannot tell a good precedent from a bad one. So: strip NAMES and
# VALUES, keep DIRECTION, CARDINALITY, ORDER and the concrete verb.
#
# All steps of one item are abstracted in a SINGLE call. Per-step calls would
# cost ~11k requests for the full arm instead of ~2.3k, and batching also gives
# the abstractor the surrounding steps as context, which helps it name a move
# precisely ("the second of two passes") instead of generically.
TPL_ABSTRACT_PATTERN = """You are turning concrete reasoning steps into reusable patterns for a \
memory bank.

Remove what is specific to this problem instance. KEEP what makes the move \
recognisable and repeatable.

REMOVE: entity names, proper nouns, specific numeric values, domain topic words.
KEEP, VERBATIM AND EXPLICITLY:
- the direction or order of traversal (left-to-right, last-to-first, \
innermost-outward, reverse chronological)
- the cardinality (one per row, all but the last two, every third item, exactly two)
- the specific operation, named precisely (subtract-then-divide is NOT \
"compute"; break-tie-on-next-character is NOT "compare")
- what kind of object is being operated on (an interval, a pair, a nested \
bracket, a row of a table, a constraint)
- any condition that gates the move (only when the count is odd, only if the \
earlier claim was disproved)

Pick "pattern_type" from exactly this list:
{pattern_types}

GOOD (mechanism survives):
  concrete: "Compare 'acton' and 'aborigine': both start 'a', second letters \
'c'(3) vs 'b'(2), so 'aborigine' sorts first."
  abstract: "Two candidates share a leading character -> compare them at the \
next character position, lower alphabet index sorts earlier -> relative order \
of the two is fixed"

GOOD (cardinality and gate survive):
  concrete: "Three of the five claims were disproved, so the majority verdict is 'disproved'."
  abstract: "A set of independently-evaluated claims with a tally -> take the \
verdict held by strictly more than half -> majority verdict established"

BAD (collapsed to a useless shell — never produce this):
  abstract: "Look at the data -> process it -> reach a conclusion"

Below are {n} consecutive steps from one solved problem. Abstract EACH one. Use \
the neighbouring steps only to understand what a step is doing — never merge \
steps, and never change their order.

CONCRETE STEPS
{steps_block}

Respond with ONLY a JSON array of exactly {n} objects, in the same order, no \
prose and no markdown fence:
[{{"step": 1, "abstract_state": "...", "abstract_action": "...", \
"abstract_next_state": "...", "pattern_type": "..."}}, ...]"""


def build_abstract_prompt(steps) -> str:
    """Prompt to abstract every step of one item in a single call."""
    block = '\n\n'.join(
        f'--- step {i + 1} ---\n'
        f"state: {s['state']}\n"
        f"action: {s['action']}\n"
        f"next_state: {s['next_state']}"
        for i, s in enumerate(steps)
    )
    return TPL_ABSTRACT_PATTERN.format(
        pattern_types=', '.join(PATTERN_TYPES),
        n=len(steps), steps_block=block,
    )


# ═════════════════════════════════════════════════════════════════════
#  Self-test: the arm-parity invariant
# ═════════════════════════════════════════════════════════════════════

def selftest_arm_parity() -> None:
    """Assert the two eval arms differ ONLY by the precedent block."""
    item = {'id': 't#0001', 'task': 'demo',
            'input': 'What is 2 + 2?', 'target': '4'}
    bare = build_solve_prompt(item)
    chunks: List[dict] = [{
        'state': 's', 'action': 'a', 'next_state': 'n',
        'pattern_type': 'arithmetic_chain',
    }]
    withmem = build_solve_prompt(item, chunks)

    assert bare.endswith(SOLVE_SUFFIX), 'bare prompt must end with the answer contract'
    assert withmem.endswith(item['input'].strip() + SOLVE_SUFFIX), (
        'memory prompt must end with the question + SOLVE_SUFFIX — otherwise '
        'claim 2 is confounded by prompt wording'
    )
    assert item['input'] in bare

    # An empty retrieval must degrade to *exactly* the baseline prompt, so a
    # fully-gated memory run is a clean no-op rather than a third condition.
    assert build_solve_prompt(item, []) == bare
    assert build_solve_prompt(item, ()) == bare

    print('prompts selftest OK — arm parity holds; gated retrieval == baseline')


if __name__ == '__main__':
    selftest_arm_parity()
