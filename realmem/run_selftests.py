#!/usr/bin/env python3
"""Run every offline selftest. No network, no spend, numpy only.

    python run_selftests.py

Exists because `python -m eval.selftest_x` depends on the current directory
being importable, which is not true on every interpreter and every shell
(PYTHONSAFEPATH, `-P`, some conda launchers, an unexpected cwd). This script
resolves the repository root from its own location, so it works from anywhere:

    python /abs/path/to/RealMemBench-main/run_selftests.py

Also prints an environment report first — most "the tests fail" reports turn out
to be a missing encoder or a stale bank rather than a broken invariant.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(ROOT, "eval")

SUITES = [
    ("selftest_retrieval", "retrieval invariants: temporal causality, Stage 1/3 contracts"),
    ("selftest_metrics", "our Recall/NDCG vs the vendored official scorer"),
    ("selftest_run", "end-to-end harness: arm comparability, dry-run, bank checks"),
    ("selftest_qa", "generation + LLM judge parsing and denominators"),
]


def environment_report():
    print("environment")
    print("-" * 66)
    print(f"  python        {sys.version.split()[0]}  ({sys.executable})")
    print(f"  repo root     {ROOT}")
    print(f"  cwd           {os.getcwd()}")

    missing = [n for n in ("eval/__init__.py", "eval/config.py", "eval/schema.py",
                           "dataset", "eval/compute_auto_metrics_for_realmem.py")
               if not os.path.exists(os.path.join(ROOT, n))]
    if missing:
        print(f"  MISSING       {missing}")
        print("\n  The repository looks incomplete. selftest_metrics needs the "
              "vendored\n  official scorer and the dataset to compare against.")
        return False

    try:
        import numpy
        print(f"  numpy         {numpy.__version__}"
              + ("" if hasattr(numpy, "asfarray")
                 else "   (2.x: asfarray removed; shim injected where needed)"))
    except ImportError:
        print("  numpy         MISSING — required")
        return False

    try:
        import sentence_transformers as st
        print(f"  sentence-transformers {st.__version__}")
    except Exception as exc:
        print(f"  sentence-transformers  not importable ({type(exc).__name__})")
        print("                (fine for the selftests; required for real runs)")

    print(f"  HF_ENDPOINT   {os.environ.get('HF_ENDPOINT', '(unset)')}")
    n_personas = len([f for f in os.listdir(os.path.join(ROOT, "dataset"))
                      if f.endswith("_dialogues_256k.json")])
    print(f"  personas      {n_personas}")
    return True


def main():
    if not environment_report():
        return 1

    env = dict(os.environ)
    # Make the repo importable regardless of how this was invoked.
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")

    print("\nselftests")
    print("-" * 66)
    failed = []
    for name, blurb in SUITES:
        script = os.path.join(EVAL, name + ".py")
        if not os.path.exists(script):
            print(f"  MISSING  {name}")
            failed.append(name)
            continue
        # encoding must be explicit: on Windows the default is the ANSI code
        # page (GBK on a Chinese install), which cannot decode the em dashes
        # and arrows in the suites' output and kills the reader thread.
        proc = subprocess.run([sys.executable, script], cwd=ROOT, env=env,
                              capture_output=True, encoding="utf-8",
                              errors="replace")
        if proc.returncode == 0:
            print(f"  PASS  {name:22s} {blurb}")
        else:
            failed.append(name)
            print(f"  FAIL  {name:22s} {blurb}")
            for line in (proc.stdout + proc.stderr).splitlines():
                if "FAIL" in line or "Error" in line or "Traceback" in line:
                    print(f"          {line}")

    print("-" * 66)
    if failed:
        print(f"{len(failed)} suite(s) failed: {failed}")
        print(f"\nRe-run one for full output:\n  python {os.path.join('eval', failed[0] + '.py')}")
        return 1
    print("all offline invariants hold")
    print("\nNext:  python -m eval.embedding          # encoder preflight")
    print("       python -m eval.build_memory --all-personas --force")
    return 0


if __name__ == "__main__":
    sys.exit(main())
