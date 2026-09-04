#!/usr/bin/env python3
"""Run retrieval for several personas concurrently, then summarise.

Personas are independent — separate output directories, separate rerank caches —
so they parallelise with no code changes and no risk of interference. What does
NOT parallelise is the session loop inside one persona: that has to stay
sequential or the streaming contract breaks.

    python run_parallel.py --test-split           # the 8 held-out personas
    python run_parallel.py --personas A,B --jobs 2
    python run_parallel.py --test-split --qa      # generation + judging too

Concurrency budget matters more than job count. Each retrieval job issues up to
REALMEM_STAGE3_SAMPLE_WORKERS (default 3) concurrent judge calls, so 8 jobs is
~24 in flight. The QA phase is worse: it fans out MAX_WORKERS per persona, so
running it 8-way at the default 8 would put ~64 requests in flight and get you
rate-limited. This script therefore runs QA in ONE process with a raised worker
count instead, which reaches the same throughput with a bound you control.

Rate limiting shows up as a rising stage3_errors / gen failures, not as an
error — a throttled judge returns nothing parseable and the candidate silently
gets no vote. The summary at the end surfaces exactly that.
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

# Fixed before any results were seen; see HARNESS.md section 8.
DEV_SPLIT = ["Lin_Wanyu", "Adeleke_Okonjo"]
TEST_SPLIT = ["Ethan_Hunt", "Kenta_Tanaka", "Kim_Ji_young", "Liam_O_Connor",
              "Oliver_Smith", "Pak_Budi", "Sarah_Miller", "Sophie_Dubois"]


def launch(personas, jobs, extra_args, log_dir):
    """Start one process per persona, at most `jobs` at a time."""
    os.makedirs(log_dir, exist_ok=True)
    pending = list(personas)
    running = {}
    done = {}

    while pending or running:
        while pending and len(running) < jobs:
            persona = pending.pop(0)
            log_path = os.path.join(log_dir, f"{persona}.log")
            log = open(log_path, "w", encoding="utf-8")
            cmd = [sys.executable, "-m", "eval.run_eval", "--persona", persona] + extra_args
            env = dict(os.environ)
            env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
            running[persona] = (proc, log, log_path, time.time())
            print(f"  start  {persona:16s} -> {log_path}")

        time.sleep(2.0)
        for persona in list(running):
            proc, log, log_path, t0 = running[persona]
            if proc.poll() is None:
                continue
            log.close()
            elapsed = time.time() - t0
            done[persona] = (proc.returncode, elapsed, log_path)
            status = "ok  " if proc.returncode == 0 else "FAIL"
            print(f"  {status}   {persona:16s} {elapsed/60:5.1f} min"
                  + ("" if proc.returncode == 0 else f"  (exit {proc.returncode})"))
            del running[persona]
    return done


def summarise(personas, log_dir):
    """Report what the runs actually produced, including throttling symptoms."""
    from eval.config import persona_retrieval_dir

    print("\n" + "=" * 96)
    print("RETRIEVAL SUMMARY")
    print("=" * 96)
    print(f"{'persona':16s} {'arm':20s} {'queries':>8} {'depth':>7} {'empty':>7} "
          f"{'s3_calls':>9} {'cache_hit':>10} {'s3_err':>7}")
    print("-" * 96)

    total_err = 0
    for persona in personas:
        pdir = persona_retrieval_dir(persona)
        if not os.path.isdir(pdir):
            print(f"{persona:16s} (no output)")
            continue
        for fn in sorted(os.listdir(pdir)):
            if not fn.endswith("_retrieval_results.json") or fn.startswith("DRYRUN-"):
                continue
            arm = fn[: -len("_retrieval_results.json")]
            with open(os.path.join(pdir, fn), "r", encoding="utf-8") as f:
                res = json.load(f)
            lens = [len(r.get("ranked_items", [])) for r in res.values()]
            depth = sum(lens) / len(lens) if lens else 0
            empty = sum(1 for n in lens if n == 0) / len(lens) if lens else 0
            calls = hits = errs = "-"
            log_path = os.path.join(log_dir, f"{persona}.log")
            if os.path.exists(log_path):
                calls, hits, errs = _stats_from_log(log_path, arm)
            if isinstance(errs, int):
                total_err += errs
            print(f"{persona:16s} {arm:20s} {len(res):>8} {depth:>7.1f} {empty:>7.3f} "
                  f"{str(calls):>9} {str(hits):>10} {str(errs):>7}")

    if total_err:
        print(f"\n  ! {total_err} Stage-3 judge errors across all runs. A throttled "
              f"gateway\n    looks exactly like this: the call fails, the candidate "
              f"gets no vote, and\n    the gated arm quietly returns less. Lower "
              f"--jobs or REALMEM_STAGE3_SAMPLE_WORKERS\n    and re-run — resume "
              f"will only redo what is missing.")


def _stats_from_log(log_path, arm):
    """Pull the per-arm stats dict the diagnostics block prints."""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return "-", "-", "-"
    marker = f"'{arm}'"
    for line in text.splitlines():
        if "stage3_calls" in line and (marker in line or arm in line):
            try:
                blob = line[line.index("{"):]
                d = json.loads(blob.replace("'", '"'))
                return (d.get("stage3_calls", "-"), d.get("stage3_cache_hits", "-"),
                        d.get("stage3_errors", "-"))
            except Exception:
                pass
    return "-", "-", "-"


def main():
    p = argparse.ArgumentParser(description="Parallel per-persona evaluation")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--personas", help="Comma-separated")
    g.add_argument("--test-split", action="store_true",
                   help=f"The 8 held-out personas: {', '.join(TEST_SPLIT)}")
    g.add_argument("--dev-split", action="store_true",
                   help=f"The 2 tuning personas: {', '.join(DEV_SPLIT)}")
    p.add_argument("--jobs", type=int, default=4,
                   help="Concurrent persona processes (default 4)")
    p.add_argument("--arms", default=None)
    p.add_argument("--log-dir", default=os.path.join(ROOT, "logs"))
    p.add_argument("--qa", action="store_true",
                   help="After retrieval, run generation+judging in ONE process")
    p.add_argument("--qa-workers", type=int, default=16,
                   help="Thread count for the QA phase (default 16)")
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = p.parse_args()

    if args.test_split:
        personas = TEST_SPLIT
    elif args.dev_split:
        personas = DEV_SPLIT
    else:
        personas = [x.strip() for x in args.personas.split(",") if x.strip()]

    extra = []
    if args.arms:
        extra += ["--arms", args.arms]

    jobs = max(1, min(args.jobs, len(personas)))
    sample_workers = int(os.environ.get("REALMEM_STAGE3_SAMPLE_WORKERS", "3"))
    print(f"personas          {len(personas)}: {', '.join(personas)}")
    print(f"concurrent jobs   {jobs}")
    print(f"in-flight judge   ~{jobs * sample_workers} "
          f"({jobs} jobs x {sample_workers} samples)")
    print(f"logs              {args.log_dir}")

    if args.dry_run:
        for persona in personas:
            print(f"  python -m eval.run_eval --persona {persona} {' '.join(extra)}")
        return 0

    t0 = time.time()
    print("\nretrieval")
    print("-" * 60)
    done = launch(personas, jobs, extra, args.log_dir)
    failed = [k for k, (rc, _e, _l) in done.items() if rc != 0]
    print(f"\nretrieval finished in {(time.time()-t0)/60:.1f} min"
          + (f"  ({len(failed)} failed: {failed})" if failed else ""))

    sys.path.insert(0, ROOT)
    summarise(personas, args.log_dir)

    if failed:
        print(f"\nRe-run the failures (resume skips completed work):")
        print(f"  python run_parallel.py --personas {','.join(failed)} --jobs {jobs}")
        return 1

    if args.qa:
        print("\n" + "=" * 96)
        print(f"QA PHASE — single process, {args.qa_workers} threads")
        print("=" * 96)
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        env["REALMEM_MAX_WORKERS"] = str(args.qa_workers)
        rc = subprocess.run(
            [sys.executable, "-m", "eval.run_qa_eval", "--personas", ",".join(personas)],
            cwd=ROOT, env=env).returncode
        if rc != 0:
            print("QA phase failed; re-run it directly to see the error.")
            return rc

    plist = ",".join(personas)
    print("\nnext:")
    print(f"  python -m eval.analyze                 --personas {plist}")
    print(f"  python -m eval.verify_against_official --personas {plist}")
    print(f"  python -m eval.ablate_from_results     --personas {plist}")
    if args.qa:
        print(f"  python -m eval.analyze_qa              --personas {plist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
