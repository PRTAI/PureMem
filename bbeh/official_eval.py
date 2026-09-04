"""
bbeh/official_eval.py — BIG-Bench Extra Hard official scoring.

Vendored VERBATIM from ``bbeh-main/bbeh/evaluate.py`` (Copyright 2025 Google
LLC, Apache-2.0) so that our numbers are directly comparable to the published
BBEH leaderboard. Two deliberate deviations, neither of which touches scoring
logic:

  1. The module-level ``print(...)`` self-test calls at the bottom of the
     original are moved into ``_selftest()`` so importing this module is silent
     and side-effect free.
  2. ``score_with_detail()`` is added — it returns the extracted prediction
     alongside the boolean so runs can log *what* was parsed out. This is the
     single most useful triage field: it separates "model was wrong" from
     "model was right but we failed to extract its answer".

Do not "improve" the matching rules. Any leniency we add that Google's harness
does not have would silently inflate every arm and make the comparison to
published baselines meaningless.
"""

from typing import Tuple


def strip_latex(response: str) -> str:
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response


def extract_answer(sample: str) -> str:
    """Extracts the final answer from the sample."""
    answer_prefixes = [
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is "
    ]
    answer = sample
    for answer_prefix in answer_prefixes:
        if answer_prefix in answer:
            answer = answer.split(answer_prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return strip_latex(answer)


def fuzzy_match(prediction: str, reference: str) -> bool:
    """Fuzzy match function for BigBench Extra Hard."""
    if prediction == reference:
        return True

    # (a) vs a
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction

    # Numbers
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass

    # quote issues
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True

    # Bracket issues
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True

    # Question mark issues
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True

    return False


def preprocess_sample(sample: str) -> str:
    prediction = extract_answer(sample.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    return prediction


def preprocess_reference(reference: str) -> str:
    reference = reference.strip().lower()
    reference = reference.replace(", ", ",")
    return reference


def evaluate_correctness(sample: str, reference: str) -> bool:
    prediction = preprocess_sample(sample)
    reference = preprocess_reference(reference)
    return fuzzy_match(prediction, reference)


# ─── additions (no scoring-logic change) ─────────────────────────────────

def score_with_detail(sample: str, reference: str) -> Tuple[bool, str, str]:
    """``(correct, extracted_prediction, normalized_reference)``.

    ``sample`` may be ``None`` or ``''`` (a failed/empty API response). Those
    score False with an empty prediction rather than raising — but note that an
    empty prediction is an *infra* signal, not a model-was-wrong signal. The
    runner distinguishes them; see ``run.py``.
    """
    if not sample:
        return False, '', preprocess_reference(reference or '')
    pred = preprocess_sample(sample)
    ref = preprocess_reference(reference or '')
    return fuzzy_match(pred, ref), pred, ref


def _selftest() -> None:
    """The original module's inline examples, as assertions."""
    cases = [
        ("Ok The final answer is: \\boxed{4}.", "4", True),
        ("[Reasoning] The final answer is: \\boxed{4}.", "3", False),
        ("Alright! The final answer is: 2, 3, 4", "2,3,4", True),
        ("blah blah The final answer is: 2, 3, 4", "2,3,5", False),
        ("Ok The answer is: (A)", "a", True),
        ("Ok The answer is: (A)", "b", False),
        ("Ok The answer is: **25**\nHere's why.", "25.0", True),
        ("Ok The answer is: **25**\nHere's why.", "26.0", False),
    ]
    for sample, ref, expected in cases:
        got = evaluate_correctness(sample, ref)
        assert got == expected, f'{sample!r} vs {ref!r}: got {got}, want {expected}'
    assert score_with_detail(None, '4') == (False, '', '4')
    print(f'official_eval selftest OK ({len(cases)} cases + empty-sample case)')


if __name__ == '__main__':
    _selftest()
