"""Single source of truth for reading RealMemBench dialogue files.

Ground-truth extraction used to be reimplemented in build_memory.py,
analyze.py and run_qa_eval.py. Three copies of a definition drift, and when the
definition *is* the measurement, drift shows up as an unexplained metric gap.
Everything that reads the dataset goes through this module instead.

Verified properties of the corpus (all 10 personas, 2055 sessions, 1415
is_query turns, 2319 gold references) — these are asserted by
selftest_metrics.py rather than assumed:

* ``session_identifier`` and ``session_uuid`` are unique within a persona, so
  either is safe as a corpus id. The official metric script keys on
  ``session_identifier``, so we do too.
* Gold comes from ``memory_session_uuids`` 100% of the time; the
  ``memory_used[].session_uuid`` fallback never fires but is kept because the
  official extractor has it.
* Gold never points forward in time: 0 of 2319 references resolve to a session
  later than the query, and only 3 to the query's own session. This is what
  makes the streaming protocol in the published README both correct and
  necessary.
* Gold never points at a session with an empty ``extracted_memory`` (0 of 2319).
"""

import re
from typing import Dict, Iterator, List, Optional, Tuple

# ── Session identifier grammar ──
#
# Two shapes occur:
#   'Literary_Creation_4:S1_01'  -> project 'Literary_Creation_4', topic 'Literary_Creation'
#   'Enhanced:S10027'            -> filler session, no project at all
#
# The 'Enhanced' sessions are 48% of the corpus (985/2055) and are never gold.
# They also never carry an extracted_memory, which is the content-side signal
# Stage 1 actually keys on — see config.STAGE1_ABSTRACT_WEIGHT.

FILLER_PREFIX = "Enhanced"


def parse_project(session_identifier: str) -> Optional[str]:
    """'Literary_Creation_4:S1_01' -> 'Literary_Creation_4'; filler -> None."""
    if not session_identifier:
        return None
    head = re.split(r"[:\-]", session_identifier, maxsplit=1)[0]
    return None if head == FILLER_PREFIX else head


def parse_topic(session_identifier: str) -> Optional[str]:
    """'Literary_Creation_4:S1_01' -> 'Literary_Creation'; filler -> None.

    Topic rather than project because a persona can run two projects on the same
    topic (Lin_Wanyu has Travel_Planning_2 and Travel_Planning_5) and gold does
    cross between them: same-topic agreement is 0.801 against 0.778 for
    same-project.
    """
    project = parse_project(session_identifier)
    if project is None:
        return None
    return re.sub(r"_\d+$", "", project)


def is_filler(session: dict) -> bool:
    """Filler sessions produce no structured memory. Content-side, not name-side."""
    return not session.get("extracted_memory")


# ── Text assembly ──

def session_text(session: dict) -> str:
    """Concatenate a session's turns the way the memory bank stores them."""
    parts = []
    for turn in session.get("dialogue_turns", []) or []:
        content = turn.get("content", "")
        if content:
            parts.append(f"Speaker {turn.get('speaker', 'Unknown')}: {content}")
    return "\n".join(parts)


def head_tail(text: str, head: int, tail: int) -> str:
    """Embedding-friendly window: MiniLM truncates at 256 word-pieces, so a long
    session would spend its whole window on opening pleasantries."""
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n...\n" + text[-tail:]


# ── Iteration ──

def iter_sessions(data: dict) -> Iterator[Tuple[int, dict]]:
    for idx, session in enumerate(data.get("dialogues", []) or []):
        yield idx, session


def iter_queries(data: dict) -> Iterator[Tuple[int, dict, int, dict, str]]:
    """Yield (session_idx, session, turn_idx, turn, question) for is_query turns.

    The published pipeline retrieves on ``turn.get('is_query')``; that subset is
    a strict subset of what the official metric script's ground-truth extractor
    accepts (which walks every User turn), and on the intersection the two agree
    on the gold set exactly. Since the official script iterates the *retrieval
    results* to pick its denominator, restricting to is_query keeps our numbers
    directly comparable with it.
    """
    for s_idx, session in iter_sessions(data):
        turns = session.get("dialogue_turns", []) or []
        for t_idx, turn in enumerate(turns):
            if not turn.get("is_query"):
                continue
            question = (turn.get("content") or "").strip()
            if question:
                yield s_idx, session, t_idx, turn, question


def _next_assistant(turns: List[dict], i: int) -> Optional[dict]:
    j = i + 1
    while j < len(turns) and turns[j].get("speaker") != "Assistant":
        j += 1
    return turns[j] if j < len(turns) else None


# ── Ground truth ──

def uuid_to_sid(data: dict) -> Dict[str, str]:
    out = {}
    for _, session in iter_sessions(data):
        suuid = session.get("session_uuid")
        if suuid:
            out[suuid] = session.get("session_identifier", "")
    return out


def corpus_ids(data: dict) -> List[str]:
    """Sorted session_identifiers — the corpus over which Recall/NDCG is defined.

    Matches the official script's ``sorted(list(all_session_ids))``.
    """
    return sorted(
        session.get("session_identifier", "")
        for _, session in iter_sessions(data)
        if session.get("session_identifier")
    )


def extract_retrieval_gold(data: dict) -> Dict[str, List[str]]:
    """question -> [gold session_identifier]. Mirrors the official extractor,
    restricted to is_query turns."""
    u2s = uuid_to_sid(data)
    gold_map: Dict[str, List[str]] = {}

    for _, session, t_idx, _turn, question in iter_queries(data):
        turns = session.get("dialogue_turns", []) or []
        # The official extractor looks at turns[i+1] specifically and requires it
        # to be the Assistant. Keep that, do not widen to _next_assistant here.
        if t_idx + 1 >= len(turns):
            continue
        nxt = turns[t_idx + 1]
        if nxt.get("speaker") != "Assistant":
            continue

        gold = set()
        for u in nxt.get("memory_session_uuids", []) or []:
            if u in u2s:
                gold.add(u2s[u])
        if not gold:
            for mem in nxt.get("memory_used", []) or []:
                if isinstance(mem, dict):
                    u = mem.get("session_uuid")
                    if u and u in u2s:
                        gold.add(u2s[u])
        if gold:
            gold_map[question] = sorted(gold)

    return gold_map


def extract_qa_gold(data: dict) -> Dict[str, Dict[str, str]]:
    """question -> {'answer', 'memory'} for the LLM judge.

    Mirrors ``compute_llm_metrics_for_realmem.load_ground_truth``, including its
    'No specific memory annotation found.' placeholder, which that script uses
    as the sentinel for skipping the memory-evaluation prompt.
    """
    out: Dict[str, Dict[str, str]] = {}
    for _, session, t_idx, _turn, question in iter_queries(data):
        turns = session.get("dialogue_turns", []) or []
        asst = _next_assistant(turns, t_idx)
        answer, memory_str = "", ""
        if asst is not None:
            answer = (asst.get("content") or "").strip()
            mems = asst.get("memory_used", []) or []
            if mems:
                contents = [m.get("content", "") for m in mems if isinstance(m, dict)]
                memory_str = "\n".join(c for c in contents if c)
        if not memory_str:
            memory_str = "No specific memory annotation found."
        out[question] = {"answer": answer, "memory": memory_str}
    return out


NO_MEMORY_ANNOTATION = "No specific memory annotation found."
