"""Verify that every persona was evaluated under the same settings.

Hyperparameters are tuned on the dev personas and then frozen for the test set.
Nothing enforces that freeze — every knob is an environment variable, so a
single stray `REALMEM_TAU=...` in one shell turns the comparison into two
different experiments reported as one. Worse, the resulting tables render
perfectly.

Each run writes a run_config.json beside its results. This diffs them.

    python -m eval.check_config_drift --all-personas
    python -m eval.check_config_drift --personas A,B --reference A
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import RETRIEVAL_RESULT_DIR, list_personas

# Differences here are expected and not drift.
IGNORE = {"written_at", "persona", "arms", "dry_run"}

# Settings that change what the numbers mean. Anything here differing is an
# error rather than a note.
CRITICAL = {
    "RETRIEVE_K", "POOL_SIZE", "TOP_N_SESSIONS", "RERANK_CANDIDATES_M",
    "TAU", "VOTE_THRESHOLD", "STAGE3_SAMPLES", "GATE_MODE",
    "STAGE1_ABSTRACT_WEIGHT", "STAGE1_TOPIC_WEIGHT", "STAGE1_RECENCY_WEIGHT",
    "CONCRETE_W", "ABSTRACT_W", "TOPIC_SOURCE", "EMBEDDING_MODEL",
    "STAGE3_MODEL", "GEN_TOP_K", "USE_KEYWORDS", "MAX_METRIC_K",
    "STAGE3_ABSTRACT_CHARS", "STAGE3_RAW_FALLBACK_CHARS",
    # External baseline settings: two personas evaluated with different mem0
    # extraction models are not the same experiment either.
    "MEM0_LLM_MODEL", "MEM0_EMBED_PROVIDER",
}


def load(persona, root):
    path = os.path.join(root, persona, "run_config.json")
    if not os.path.exists(path):
        return None, path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), path


def flatten(payload):
    out = {k: v for k, v in payload.items() if k not in IGNORE and k != "config"}
    out.pop("bank", None)
    for k, v in payload.get("config", {}).items():
        out[k] = v
    bank = payload.get("bank") or {}
    for k in ("bank_format", "embedding_model", "embedding_backend"):
        if k in bank:
            out[f"bank.{k}"] = bank[k]
    return out


def main():
    p = argparse.ArgumentParser(description="Detect config drift between personas")
    p.add_argument("--personas", default=None)
    p.add_argument("--all-personas", action="store_true")
    p.add_argument("--reference", default=None,
                   help="Persona to compare against (default: the first found)")
    p.add_argument("--retrieval-result-dir", default=RETRIEVAL_RESULT_DIR)
    args = p.parse_args()

    personas = (list_personas() if args.all_personas
                else [x.strip() for x in (args.personas or "").split(",") if x.strip()])
    if not personas:
        p.error("Specify --personas or --all-personas")

    loaded, missing = {}, []
    for persona in personas:
        payload, path = load(persona, args.retrieval_result_dir)
        if payload is None:
            missing.append((persona, path))
        else:
            loaded[persona] = flatten(payload)

    if missing:
        print("No run_config.json for:")
        for persona, path in missing:
            print(f"  {persona:16s} {path}")
        print("\nThese ran before run_config.json was written. Their settings "
              "cannot be\nverified — re-run them, or confirm by hand before "
              "pooling with the rest.\n")

    if len(loaded) < 2:
        print("Need at least two runs with a config to compare.")
        return 1 if missing else 0

    ref_name = args.reference if args.reference in loaded else sorted(loaded)[0]
    ref = loaded[ref_name]
    print(f"reference: {ref_name}\n")

    critical_drift = other_drift = 0
    for persona in sorted(loaded):
        if persona == ref_name:
            continue
        cur = loaded[persona]
        diffs = []
        for key in sorted(set(ref) | set(cur)):
            a, b = ref.get(key, "<absent>"), cur.get(key, "<absent>")
            if a != b:
                diffs.append((key, a, b))
        if not diffs:
            print(f"  ok    {persona:16s} identical to {ref_name}")
            continue
        crit = [d for d in diffs if d[0] in CRITICAL]
        rest = [d for d in diffs if d[0] not in CRITICAL]
        critical_drift += len(crit)
        other_drift += len(rest)
        tag = "DRIFT" if crit else "note "
        print(f"  {tag} {persona:16s} {len(diffs)} difference(s)")
        for key, a, b in crit:
            print(f"          ! {key}: {ref_name}={a!r} vs {persona}={b!r}")
        for key, a, b in rest:
            print(f"            {key}: {ref_name}={a!r} vs {persona}={b!r}")

    print()
    if critical_drift:
        print(f"FAILED: {critical_drift} difference(s) in settings that change what "
              f"the numbers mean.\nThese personas were not the same experiment; "
              f"do not pool them into one table.")
        return 1
    if other_drift:
        print(f"{other_drift} non-critical difference(s); metrics remain comparable.")
    print("Settings are consistent across all compared personas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
