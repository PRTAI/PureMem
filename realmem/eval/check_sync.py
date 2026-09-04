"""Verify that a copy of the harness matches the source tree, file by file.

Copying a tree between machines silently drops things — an editor lock, a
partial rsync, a .gitignore rule. The failure shows up much later as behaviour
nobody can explain, so it is worth ruling out in ten seconds rather than
suspecting it for an hour.

    # on the machine with the good copy
    python -m eval.check_sync --write manifest.json

    # copy manifest.json across, then
    python -m eval.check_sync --verify manifest.json

Hashes file *content*, not mtime or size alone, so a truncated or
line-ending-mangled copy is caught too. Data, results and caches are excluded:
they are large, legitimately differ between machines, and are not code.
"""

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Code and docs that must be identical. Everything else is data or output.
INCLUDE_DIRS = ["eval", "docs"]
INCLUDE_ROOT_FILES = ["run_selftests.py", "run_parallel.py",
                      "requirements-harness.txt", "HARNESS.md"]
INCLUDE_EXT = {".py", ".md", ".txt"}
EXCLUDE_PARTS = {"__pycache__", ".git", "retrieval_result", "_deprecated_20260826",
                 "memory_banks", "dataset", "mem0_store", "logs", ".ipynb_checkpoints"}


def _iter_files():
    for name in INCLUDE_ROOT_FILES:
        p = os.path.join(ROOT, name)
        if os.path.isfile(p):
            yield p
    for d in INCLUDE_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames if x not in EXCLUDE_PARTS]
            if any(part in EXCLUDE_PARTS for part in dirpath.split(os.sep)):
                continue
            for f in sorted(filenames):
                if os.path.splitext(f)[1] in INCLUDE_EXT and not f.startswith("_"):
                    yield os.path.join(dirpath, f)


def build_manifest() -> dict:
    entries = {}
    for path in sorted(_iter_files()):
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        with open(path, "rb") as f:
            raw = f.read()
        # Normalise line endings so a CRLF/LF difference is not reported as
        # corruption — it is not, and flagging it would bury real problems.
        norm = raw.replace(b"\r\n", b"\n")
        entries[rel] = {"sha256": hashlib.sha256(norm).hexdigest()[:16],
                        "bytes": len(norm)}
    return {"root_name": os.path.basename(ROOT), "n_files": len(entries),
            "files": entries}


def main():
    p = argparse.ArgumentParser(description="Check a copied tree against a manifest")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", metavar="FILE", help="Write a manifest of this tree")
    g.add_argument("--verify", metavar="FILE", help="Compare this tree against one")
    g.add_argument("--summary", action="store_true",
                   help="Print one hash covering the whole tree")
    args = p.parse_args()

    m = build_manifest()

    if args.summary:
        combined = hashlib.sha256(
            "".join(f"{k}:{v['sha256']}" for k, v in sorted(m["files"].items()))
            .encode()).hexdigest()[:16]
        print(f"files: {m['n_files']}")
        print(f"tree hash: {combined}")
        print("\nRun the same command on the other machine; identical hashes mean\n"
              "the code is identical.")
        return 0

    if args.write:
        with open(args.write, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=1, sort_keys=True)
        print(f"wrote {args.write}: {m['n_files']} files")
        return 0

    with open(args.verify, "r", encoding="utf-8") as f:
        ref = json.load(f)

    ours, theirs = m["files"], ref["files"]
    missing = sorted(set(theirs) - set(ours))
    extra = sorted(set(ours) - set(theirs))
    differ = sorted(k for k in set(ours) & set(theirs)
                    if ours[k]["sha256"] != theirs[k]["sha256"])

    print(f"reference: {ref['n_files']} files    here: {m['n_files']} files\n")
    if missing:
        print(f"MISSING here ({len(missing)}) — never copied across:")
        for k in missing:
            print(f"   {k}")
    if differ:
        print(f"\nDIFFERENT ({len(differ)}) — copied, but not the same content:")
        for k in differ:
            print(f"   {k:44s} {theirs[k]['bytes']:>7} -> {ours[k]['bytes']:>7} bytes")
    if extra:
        print(f"\nEXTRA here ({len(extra)}) — not in the reference:")
        for k in extra:
            print(f"   {k}")

    if not (missing or differ):
        print("IDENTICAL: every tracked file matches"
              + (" (plus some extra files, listed above)" if extra else ""))
        return 0
    print(f"\n{len(missing)} missing, {len(differ)} different. Re-copy those, or "
          f"rsync the whole tree:\n"
          f"  rsync -av --exclude '__pycache__' --exclude 'retrieval_result' "
          f"<src>/ <dst>/")
    return 1


if __name__ == "__main__":
    sys.exit(main())
