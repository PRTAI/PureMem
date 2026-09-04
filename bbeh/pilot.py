"""
bbeh/pilot.py — staged pilot driver: 3 tasks x 20 items, end to end.

Run this before any expensive pass. It cannot tell you whether either claim
holds (60 test items is nowhere near enough) — it answers the three mechanical
questions that decide whether the full run is worth funding at all:

    1. does the teacher's CoT clear verification often enough to build a bank?
    2. does the Stage-3 gate inject into a usable fraction of items?
    3. do cross-task precedents survive Stage 1 at all?

**Why this is staged rather than one script.** Choosing the pilot tasks requires
seeing the difficulty landscape first. BBEH is extra-hard and the student may sit
at pass_rate 0 on many tasks; if all three pilot tasks land on the floor, the ZPD
band is empty and the pilot fails for a reason that has nothing to do with the
harness. So the broad probe runs first, ``pick`` ranks the tasks from its output,
and only then does anything expensive happen. The probe is also the cheapest
stage and its cache is per-attempt and nested, so nothing spent on the broad
sweep is wasted — the full probe reuses every attempt verbatim.

Stages (each resumable; rerun any of them safely):

    python -m bbeh.pilot ping      ~$0   one 8-token call per model: endpoint, key,
                                         and every model name in one shot
    python -m bbeh.pilot check     free  selftests; refuses to continue if any fail
    python -m bbeh.pilot probe     ~$11  23 tasks x 20 train items, k=5
    python -m bbeh.pilot pick      free  rank tasks by ZPD yield, write the choice
    python -m bbeh.pilot build     ~$14  deepen probe, teacher CoT, abstract, bank
    python -m bbeh.pilot eval      ~$2   both arms + analysis
    python -m bbeh.pilot all       chains all of the above, pausing at `pick`

The chosen tasks are written to ``work/pilot_tasks.json`` and reused by later
stages. That is not just ergonomics: it guarantees both arms are evaluated over
the same task set. ``analyze.py`` detects a mismatch, but preventing it is
better than reporting it.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from bbeh import config, data

PILOT_TASKS_JSON = os.path.join(config.WORK_DIR, 'pilot_tasks.json')
PILOT_VERSION = 'pilot-zpd'

PROBE_PER_TASK = 20        # broad sweep width
DEEPEN_PER_TASK = 50       # for the chosen tasks, so the bank is thick enough
TEST_PER_TASK = 20
N_PILOT_TASKS = 3

# List prices, USD per million tokens. Your proxy will differ — these exist to
# get the order of magnitude right and to make the teacher's dominance visible,
# not to predict your invoice.
PRICES = {
    'haiku': (1.0, 5.0),
    'sonnet': (3.0, 15.0),
    'opus': (15.0, 75.0),
}

# Rough completion lengths, from the dry-run token accounting. Solve responses on
# BBEH run long; the teacher's run longer still because it writes full CoT.
OUT_TOK_SOLVE = 800
OUT_TOK_TEACHER = 1500
OUT_TOK_JUDGE = 100
TEACHER_ATTEMPTS = 1.5     # rejection sampling: every failed verification respends


# Models whose tier could not be recognised, so their estimates ran on the
# fallback price. Collected rather than logged per-call because the interesting
# unit is "this estimate is unpriced", stated once next to the dollar figure.
_UNPRICED = set()

FALLBACK_TIER = 'haiku'


def _tier(model: str) -> str:
    """Price tier for a model name, or ``''`` if the name is not recognised.

    Returning ``''`` rather than quietly answering ``'haiku'`` is the point. The
    caller then has to decide what to do about an unknown model, and the estimate
    can say it is a guess. A silent default here produces a confident dollar
    figure for a model nobody priced — and the number is used to decide whether
    to spend hundreds, which is exactly when a fabricated input must not look
    like a measured one.
    """
    m = model.lower()
    for t in ('opus', 'sonnet', 'haiku'):
        if t in m:
            return t
    return ''


def _cost(model: str, in_tok: float, out_tok: float) -> float:
    tier = _tier(model)
    if not tier:
        _UNPRICED.add(model)
        tier = FALLBACK_TIER
    pin, pout = PRICES[tier]
    return (in_tok * pin + out_tok * pout) / 1e6


def _pricing_caveat() -> str:
    """One line naming any model that was estimated at fallback prices."""
    if not _UNPRICED:
        return ''
    names = ', '.join(sorted(_UNPRICED))
    return (f'  NOTE: no price table for {names} — estimated at'
            f' {FALLBACK_TIER}-tier rates.\n        Treat the figure as an'
            ' order of magnitude, not a quote. Add the tier to\n        PRICES'
            ' in pilot.py if you want this to mean anything.')


def _tok(items: Sequence[dict]) -> float:
    """Prompt tokens for a set of items, ~4 chars/token."""
    return sum(len(it['input']) for it in items) / 4


class Mode:
    """Dry vs real, in one place.

    Dry runs must stay in their own namespace end to end — the memory version,
    the run labels, and every cache path. ``build_memory`` refuses a version id
    without the prefix and ``ApproachFitReranker`` refuses a real cache path, so
    getting this wrong fails loudly rather than quietly seeding a real cache with
    fabricated numbers. This class exists so the prefix is derived once instead
    of being spelled out at four call sites.
    """

    def __init__(self, dry: bool):
        self.dry = dry
        self.version = ('DRYRUN-' if dry else '') + PILOT_VERSION
        self.flag = ['--dry-run'] if dry else []

    @property
    def memory_arm(self) -> str:
        return ('DRYRUN-' if self.dry else '') + f'memory_{self.version}'

    @property
    def baseline_arm(self) -> str:
        return ('DRYRUN-' if self.dry else '') + 'no_memory'

    def probe_model_label(self, model: str) -> str:
        return ('DRYRUN-' if self.dry else '') + model


# ═════════════════════════════════════════════════════════════════════
#  Running stages
# ═════════════════════════════════════════════════════════════════════

def sh(args: Sequence[str], label: str = '') -> None:
    """Run a bbeh module, streaming its output. Abort the pilot on failure.

    Output is deliberately not captured: these stages run for minutes and their
    progress lines (running accuracy, ETA, infra-error counts) are how you catch
    a problem before it has consumed the budget.
    """
    cmd = [sys.executable, '-m'] + list(args)
    print(f'\n$ {" ".join(cmd[2:])}', flush=True)
    r = subprocess.run(cmd, cwd=config.REPO_ROOT)
    if r.returncode != 0:
        raise SystemExit(
            f'\n{"=" * 70}\nSTAGE FAILED{f" ({label})" if label else ""}: '
            f'exit {r.returncode}\n'
            'Nothing further will run. Cached work is preserved — fix the cause '
            'and rerun this same stage.\n' + '=' * 70)


def confirm(prompt: str, assume_yes: bool) -> None:
    if assume_yes:
        print(f'{prompt}  [--yes]')
        return
    try:
        ans = input(f'{prompt}  [y/N] ').strip().lower()
    except EOFError:
        ans = ''
    if ans not in ('y', 'yes'):
        raise SystemExit('aborted; nothing spent')


def load_tasks(explicit: Optional[Sequence[str]] = None) -> List[str]:
    if explicit:
        return list(explicit)
    if not os.path.exists(PILOT_TASKS_JSON):
        raise SystemExit(
            f'no task selection at {PILOT_TASKS_JSON}.\n'
            'Run `python -m bbeh.pilot pick` first, or pass --tasks explicitly.')
    with open(PILOT_TASKS_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)['tasks']


# ═════════════════════════════════════════════════════════════════════
#  Stage: check
# ═════════════════════════════════════════════════════════════════════

def stage_ping(models: Sequence[str], base_url: str = '', api_key: str = '',
               auth_scheme: str = '', quiet: bool = False) -> List[str]:
    """One trivial call per distinct model. Costs fractions of a cent.

    Every selftest in this harness is deliberately offline, which leaves exactly
    one thing unverified before the first paid stage: that the endpoint, the key
    and each model name actually work together. All three models are pinged, not
    just the student, because entitlements are per-model — a key can be good for
    haiku and rejected for opus, and the opus teacher is the single most
    expensive stage. Finding that out here costs a cent; finding it out at
    `pilot build` means having already paid for probe.

    ``base_url`` / ``api_key`` override config for this call only. Diagnosing a
    rejection means changing one variable at a time, and editing config.py
    between attempts both loses the comparison and risks leaving the file on
    whichever value was tried last.
    """
    from bbeh.api_client import GenClient

    url = base_url or config.API['base_url']
    key = api_key or config.API['api_key']
    scheme = auth_scheme or config.API.get('api_auth_scheme', 'bearer')
    print(f'{"=" * 70}\nSTAGE 0a  endpoint reachability (~$0.00)\n{"=" * 70}')
    print(f'endpoint {url}   protocol {config.API["api_protocol"]}'
          f'   key ...{key[-6:]}')
    print(f'auth     {"x-api-key: <key>" if scheme == "x-api-key" else "Authorization: Bearer <key>"}')
    if quiet:
        logging.disable(logging.ERROR)
    bad, harness_bugs = [], []
    for m in dict.fromkeys(models):
        try:
            # generate_detailed, not generate: the latter flattens everything to
            # a string, which throws away the distinction between "the endpoint
            # refused us" and "the model returned an empty body" — the two
            # things this stage exists to tell apart.
            r = GenClient(m, base_url=url, api_key=key, auth_scheme=scheme,
                          max_tokens=100, max_retries=1).generate_detailed(
                'Reply with the single word: ok')
            text = (r.text or '').strip()
            if r.error:
                print(f'  FAIL  {m:34s} -> {r.error[:110]}')
                bad.append((m, r.error))
            elif not text:
                print(f'  EMPTY {m:34s} -> HTTP 200 with no body')
                bad.append((m, 'empty body'))
            else:
                print(f'  OK    {m:34s} -> {text[:24]!r}')
        except ImportError as e:
            # GenClient imports openai lazily, so a missing package arrives here
            # looking like a per-model failure. It is neither — reporting it as
            # "unreachable" would send you to check the endpoint and the key,
            # neither of which is wrong.
            raise SystemExit(
                f'missing dependency: {e}\n'
                'This is an environment problem, not an endpoint one. '
                'Install it (`pip install openai`) and rerun.') from e
        except Exception as e:                                # noqa: BLE001
            # generate_detailed is documented never to raise, so anything
            # arriving here escaped our own code, not the network. Saying
            # otherwise would send you off to check a key that is fine.
            detail = f'{type(e).__name__}: {e}'
            print(f'  BUG   {m:34s} -> {detail[:110]}')
            harness_bugs.append((m, detail))

    if quiet:
        logging.disable(logging.NOTSET)

    if harness_bugs:
        print('\nThis is a bug in pilot.py/api_client.py, not in your configuration.')
        print('generate_detailed() is supposed to swallow every API failure and')
        print('report it via .error, so an exception escaping it means the ping')
        print('itself is broken. Nothing was learned about the endpoint.')
        raise SystemExit(f'ping is broken for {len(harness_bugs)} model(s)')
    if bad:
        blob = ' '.join(e for _, e in bad)
        print('\nThe harness never got a usable answer. Reading the status code:')
        # 401-vs-403 is the one distinction that pins down the auth scheme, and it
        # is easy to read backwards. 401 "no token" means the gateway never found
        # a credential where it looked, so the scheme is wrong. 403 means it read
        # the credential and refused it, so the scheme is RIGHT and the problem is
        # the key or its entitlements. Switching schemes on a 403 walks away from
        # the working half of the configuration.
        if '401' in blob:
            print('  401 / "no token" — the gateway found no credential where it')
            print('    looked, so the scheme is wrong for this host, not the key.')
            print(f'    Currently sending: {scheme}. Try the other one:')
            other = 'bearer' if scheme == 'x-api-key' else 'x-api-key'
            print(f'      python -m bbeh.pilot ping --auth-scheme {other}')
        elif '403' in blob:
            print('  403 / "access denied" — the credential WAS read and refused.')
            print('    The scheme is therefore correct; do not change it. This is')
            print('    the key, its entitlements, or its binding to this host.')
        else:
            print('  no 401/403 — if there is no JSON error body, you never reached')
            print('    the API at all, and this is network or DNS, not credentials.')
        print('\n  Anything to check by hand: base_url and api_key in bbeh/config.py')
        print('  (or BBEH_BASE_URL / BBEH_API_KEY / BBEH_API_AUTH_SCHEME).')
        if len(bad) == len(set(models)):
            print('\nEvery model failed identically, so this is not per-model'
                  ' entitlement.\nRun `python -m bbeh.pilot ping --sweep` for the'
                  ' endpoint x scheme matrix\n— one variable at a time, which'
                  ' guessing cannot do.')
        raise SystemExit(f'{len(bad)} of {len(set(models))} models unreachable')
    print('\nall models reachable')
    return [m for m, _ in bad]


# Extra endpoints for `pilot ping --sweep` to try alongside the configured one.
# config.API['base_url'] is prepended automatically, so it need not be listed.
#
# The sweep exists because a lone 403 is uninterpretable: it looks identical
# whether the key is wrong, the auth scheme is wrong, or that particular gateway
# has retired your account. Trying the same key against a second endpoint is what
# separates those. Supply your own comma-separated list if you have a fallback
# gateway; with none configured the sweep still varies the auth scheme and the
# host/path split, which is the more common failure.
#
#   export BBEH_CANDIDATE_URLS=https://gateway-a.example/v1,https://gateway-b.example/v1
CANDIDATE_URLS = [
    u.strip() for u in os.environ.get('BBEH_CANDIDATE_URLS', '').split(',')
    if u.strip()
]


def _classify(error: str) -> str:
    """Bucket one failure by what it rules out.

    The buckets are chosen so that each one eliminates a different candidate
    cause, which is what makes the sweep worth running: 401 clears the key and
    implicates the scheme, 403 clears the scheme and implicates the key, 404
    clears the host and implicates the path, and a bodyless connection error
    clears all three and implicates the network.
    """
    if '401' in error:
        return 'no_token'          # gateway looked, found no credential
    if '403' in error:
        return 'denied'            # gateway read the credential and refused it
    if '404' in error or 'not found' in error.lower():
        return 'not_found'         # host is alive, path is wrong
    return 'unreachable'           # never got an HTTP error body at all


def stage_ping_sweep(model: str, api_key: str = '') -> None:
    """One model, one key, against endpoint x auth-scheme. Changes one thing at a time.

    A 403 on its own is ambiguous — the key, the host, the path, or the way the
    credential is framed could each produce it, and they are indistinguishable
    from the error body. Sweeping the two dimensions that are cheap to vary
    resolves it in one pass. Sweeping is also strictly better than editing
    config.py between attempts, which loses the comparison and tends to leave
    the file on whatever was tried last.

    The previous endpoint is in the list on purpose: it is the control. If it
    answers and the new one does not, the key is fine and the host is the
    problem — a conclusion no amount of staring at a single 403 can reach.
    """
    from bbeh.api_client import GenClient, AUTH_SCHEMES

    key = api_key or config.API['api_key']
    urls = list(dict.fromkeys([config.API['base_url']] + CANDIDATE_URLS))
    print(f'{"=" * 70}\nSTAGE 0a  endpoint x auth sweep  ({model}, key ...{key[-6:]})\n'
          f'{"=" * 70}')
    print(f'{len(urls) * len(AUTH_SCHEMES)} combinations, 8 tokens each\n')
    print(f'  {"endpoint":34s} {"auth":11s} result')
    logging.disable(logging.ERROR)
    ok_rows: List[tuple] = []
    rows: Dict[str, List[tuple]] = defaultdict(list)
    try:
        for url in urls:
            for scheme in AUTH_SCHEMES:
                try:
                    r = GenClient(model, base_url=url, api_key=key,
                                  auth_scheme=scheme, max_tokens=100,
                                  max_retries=1, timeout=25
                                  ).generate_detailed('say ok')
                    if r.ok:
                        verdict = f'OK   {(r.text or "").strip()[:20]!r}'
                        ok_rows.append((url, scheme))
                        rows['ok'].append((url, scheme))
                    else:
                        verdict = f'FAIL {r.error[:70]}'
                        rows[_classify(r.error)].append((url, scheme))
                except Exception as e:                        # noqa: BLE001
                    verdict = f'BUG  {type(e).__name__}: {e}'[:76]
                    rows['bug'].append((url, scheme))
                print(f'  {url:34s} {scheme:11s} {verdict}')
    finally:
        logging.disable(logging.NOTSET)

    if ok_rows:
        url, scheme = ok_rows[0]
        print(f'\n{len(ok_rows)} working combination(s). Put this in bbeh/config.py:')
        print(f"    'base_url': '{url}',")
        print(f"    'api_auth_scheme': '{scheme}',")
        print('and mirror base_url into the repo-root config.py, which the'
              ' PuzzleWorld/GSM8K\nside reads independently.')
        return
    # The table alone is not the deliverable. Every legend printed here used to
    # be a rule the reader had to match against eight rows by eye, which is the
    # step where a wrong conclusion gets drawn — the same reason analyze.py
    # prints a verdict rather than a contingency table. The classification below
    # is mechanical, so state it.
    kinds = {k: sorted(v) for k, v in rows.items()}
    reached = kinds.get('denied', []) + kinds.get('no_token', [])
    hosts_reached = {u for u, _ in reached}

    def hosts(kind: str) -> List[str]:
        """The distinct hosts in one bucket, flagging any that split by scheme.

        DNS resolution and path routing both happen before the credential is
        looked at, so a host in the unreachable/404 buckets belongs there under
        *every* scheme. Reporting one line per combination would print the same
        host twice and inflate the count, making two facts look like four. If a
        host lands here under only some schemes, that breaks the assumption this
        collapsing rests on, so say so rather than quietly deduping it away.
        """
        by_host: Dict[str, set] = defaultdict(set)
        for u, s in kinds.get(kind, []):
            by_host[u].add(s)
        out = []
        for u in sorted(by_host):
            partial = by_host[u] != set(AUTH_SCHEMES)
            out.append(f'{u}   [only under {"/".join(sorted(by_host[u]))} —'
                       ' unexpected, this bucket should not depend on the'
                       ' scheme]' if partial else u)
        return out

    print('\nNothing worked. What the table establishes:')
    unreachable = hosts('unreachable')
    if unreachable:
        print(f'  {len(unreachable)} host(s) never answered at all (no JSON error'
              ' body), so\n  this is DNS or the network, not credentials:')
        for line in unreachable:
            print(f'      {line}')
    not_found = hosts('not_found')
    if not_found:
        print(f'  {len(not_found)} host(s) returned 404 — alive, but that path is'
              ' wrong:')
        for line in not_found:
            print(f'      {line}')
    # Read the bug bucket BEFORE drawing any conclusion. An exception that
    # escaped GenClient means the request may never have been sent, so those
    # rows are not evidence about the endpoint at all — and if they dominate,
    # every line below would be a confident statement about a network that was
    # never contacted. Printing "fix connectivity first" on top of a harness bug
    # is the failure this whole diagnostic exists to avoid.
    if kinds.get('bug'):
        hs = sorted({u for u, _ in kinds['bug']})
        print(f'  {len(kinds["bug"])} combination(s) raised inside the harness'
              f' ({len(hs)} host(s)).\n  Those rows say nothing about the'
              ' endpoint — the call may not have been\n  sent. Fix the BUG rows'
              ' above first; this is pilot.py/api_client.py,\n  not your'
              ' configuration.')
        if not reached:
            return

    if not reached:
        print('  No gateway ever answered. Nothing was learned about the key —'
              ' fix\n  connectivity first, then rerun.')
        return

    print(f'  {len(hosts_reached)} gateway(s) answered and refused the credential:')
    for u in sorted(hosts_reached):
        print(f'      {u}')
    if kinds.get('no_token') and kinds.get('denied'):
        print('  Both 401 and 403 appear, which pins the scheme down: the host'
              ' reads\n  Authorization (403 = credential seen and refused) and'
              ' ignores x-api-key\n  (401 = no credential found). So bearer is'
              ' correct and the header framing\n  is not the problem.')
    if len(hosts_reached) > 1:
        print('\n  CONCLUSION: distinct hosts refuse this key identically, so the'
              ' endpoint\n  is not the variable — including whichever one you'
              ' switched to or from.\n  The key itself lacks access to this API'
              ' route. A key can be valid for a\n  subscription/CLI plan and still'
              ' carry no raw-API entitlement, which is\n  exactly what an'
              ' authenticated-but-denied response looks like.')
        print('\n  Next step is not in this repo: obtain a key with API access,'
              ' then\n      python -m bbeh.pilot ping --api-key sk-...')
        print('  (--api-key applies to one invocation, so you can confirm before'
              ' writing\n  it into config.py.)')
    else:
        print('\n  Only one host answered, so the endpoint is still confounded with'
              ' the key.\n  Add a known-good endpoint to CANDIDATE_URLS and rerun'
              ' before concluding.')


def stage_check() -> None:
    print(f'{"=" * 70}\nSTAGE 0  free selftests\n{"=" * 70}')
    sh(['bbeh.data', 'build-splits'], 'splits')
    sh(['bbeh.data', 'verify'], 'split verification')
    for mod in ('bbeh.reranker', 'bbeh.selftest_retrieval',
                'bbeh.selftest_run', 'bbeh.selftest_analyze'):
        sh([mod], mod)
    print('\nall selftests passed — the harness is sane before any spend')


# ═════════════════════════════════════════════════════════════════════
#  Stage: probe (broad)
# ═════════════════════════════════════════════════════════════════════

def stage_probe(model: str, k: int, per_task: int, workers: int,
                assume_yes: bool, mode: Mode) -> None:
    items = data.select_items(data.load_split('train'), limit_per_task=per_task)
    in_tok = _tok(items) * k
    out_tok = len(items) * k * OUT_TOK_SOLVE
    est = _cost(model, in_tok, out_tok)
    print(f'\n{"=" * 70}\nSTAGE A  broad difficulty probe'
          f'{"  [DRY RUN]" if mode.dry else ""}\n{"=" * 70}')
    print(f'{len(items)} train items x k={k} = {len(items) * k} attempts on {model}')
    if mode.dry:
        print('fabricated locally; $0. Note the resulting pass_rates are invented, so')
        print('the ZPD yields below mean nothing — this only exercises the plumbing.')
    else:
        print(f'estimated ~${est:.0f} (~{in_tok / 1e6:.2f}M in, ~{out_tok / 1e6:.2f}M out)')
        if _pricing_caveat():
            print(_pricing_caveat())
        print('Not wasted if you later probe the full train set: the cache unit is one')
        print('attempt, and select_items is nested, so every attempt is reused verbatim.')
        confirm('proceed?', assume_yes)
    sh(['bbeh.probe', '--model', model, '--k', str(k),
        '--limit-per-task', str(per_task), '--max-workers', str(workers)]
       + mode.flag, 'probe')


# ═════════════════════════════════════════════════════════════════════
#  Stage: pick  (free)
# ═════════════════════════════════════════════════════════════════════

def stage_pick(model: str, n: int, exclude: Sequence[str],
               explicit: Optional[Sequence[str]] = None,
               mode: Optional[Mode] = None) -> List[str]:
    """Rank tasks by ZPD yield, penalising input size. Writes the selection."""
    mode = mode or Mode(False)
    print(f'\n{"=" * 70}\nSTAGE B  choose pilot tasks (free)\n{"=" * 70}')
    path = config.probe_path(mode.probe_model_label(model))
    if not os.path.exists(path):
        raise SystemExit(f'no probe results at {path} — run `pilot probe` first')
    recs = [r for r in data.read_jsonl(path) if r.get('pass_rate') is not None]
    if not recs:
        raise SystemExit('probe file has no scored items')

    chars: Dict[str, List[int]] = defaultdict(list)
    for it in data.load_split('train'):
        chars[it['task']].append(len(it['input']))

    by_task: Dict[str, List[dict]] = defaultdict(list)
    for r in recs:
        by_task[r['task']].append(r)

    rows = []
    for task, rs in by_task.items():
        band = sum(1 for x in rs
                   if config.ZPD_LOW <= x['pass_rate'] <= config.ZPD_HIGH)
        mean_chars = sum(chars[task]) / max(1, len(chars[task]))
        rows.append({
            'task': task, 'n': len(rs), 'zpd': band,
            'yield': band / len(rs),
            'mean_pass': sum(x['pass_rate'] for x in rs) / len(rs),
            'chars': mean_chars,
            # Yield per unit of input size: a task with a great band but 13k-char
            # inputs costs 3x per item at every later stage.
            'score': (band / len(rs)) / (mean_chars / 4000) ** 0.5,
        })
    rows.sort(key=lambda r: r['score'], reverse=True)

    print(f'{"task":34s} {"n":>4s} {"zpd":>5s} {"yield":>7s} {"pass":>6s} '
          f'{"chars":>7s} {"score":>6s}')
    for r in rows:
        flag = '  EXCLUDED' if r['task'] in exclude else (
            '  <- no ZPD items' if r['zpd'] == 0 else '')
        print(f'{r["task"]:34s} {r["n"]:4d} {r["zpd"]:5d} {r["yield"]:7.2f} '
              f'{r["mean_pass"]:6.3f} {r["chars"]:7.0f} {r["score"]:6.2f}{flag}')

    if explicit:
        chosen = list(explicit)
        print(f'\nusing the tasks you named: {chosen}')
    else:
        pool = [r for r in rows if r['task'] not in exclude and r['zpd'] > 0]
        if len(pool) < n:
            raise SystemExit(
                f'\nonly {len(pool)} tasks have a non-empty ZPD band. This student is '
                'at or near the floor\non BBEH, and the ZPD framing does not apply at '
                'this difficulty. Options:\n'
                '  (a) raise k so the band resolves more finely (k=5 gives 6 values);\n'
                '  (b) widen the band via --zpd-low/--zpd-high, or use --zpd-strict;\n'
                '  (c) use a stronger student;\n'
                '  (d) drop the size exclusions with --exclude (currently '
                f'{list(exclude)});\n'
                '  (e) report honestly that the premise does not hold here.')
        chosen = [r['task'] for r in pool[:n]]
        print(f'\nrecommended: {chosen}')
        print('Ranked by ZPD yield divided by sqrt(mean input size), so a task with a')
        print('good band but 13k-char inputs loses to a comparable cheaper one.')
        print('\nOne thing this ranking CANNOT judge: mechanism diversity. Cross-task')
        print('transfer is only testable if the three tasks need genuinely different')
        print('reasoning moves. If the top three look like variations on one skill,')
        print('override with --tasks — the ranking has no idea what these tasks do.')

    os.makedirs(config.WORK_DIR, exist_ok=True)
    with open(PILOT_TASKS_JSON, 'w', encoding='utf-8') as f:
        json.dump({'tasks': chosen, 'model': model,
                   'zpd_band': [config.ZPD_LOW, config.ZPD_HIGH]},
                  f, indent=2, ensure_ascii=False)
    print(f'\n-> {PILOT_TASKS_JSON}  (later stages read this, so both arms match)')
    return chosen


# ═════════════════════════════════════════════════════════════════════
#  Stage: build
# ═════════════════════════════════════════════════════════════════════

def stage_build(tasks: Sequence[str], model: str, teacher: str, k: int,
                per_task: int, workers: int, assume_yes: bool, mode: Mode) -> None:
    items = data.select_items(data.load_split('train'), tasks=tasks,
                              limit_per_task=per_task)
    # Only ZPD-band items get a teacher CoT, so size the estimate off the
    # measured yield for these tasks rather than assuming all of them qualify.
    path = config.probe_path(mode.probe_model_label(model))
    yields = []
    if os.path.exists(path):
        by_task = defaultdict(list)
        for r in data.read_jsonl(path):
            if r.get('pass_rate') is not None and r['task'] in tasks:
                by_task[r['task']].append(r['pass_rate'])
        for t in tasks:
            ps = by_task.get(t, [])
            if ps:
                yields.append(sum(1 for p in ps
                                  if config.ZPD_LOW <= p <= config.ZPD_HIGH) / len(ps))
    y = sum(yields) / len(yields) if yields else 0.3
    n_zpd = max(1, int(len(items) * y))

    probe_extra = len(items) * k                    # upper bound; cache covers most
    c_probe = _cost(model, _tok(items) * k, probe_extra * OUT_TOK_SOLVE)
    c_teach = _cost(teacher, _tok(items[:n_zpd]) * TEACHER_ATTEMPTS,
                    n_zpd * TEACHER_ATTEMPTS * OUT_TOK_TEACHER)

    print(f'\n{"=" * 70}\nSTAGE C  deepen probe -> teacher CoT -> memory bank'
          f'{"  [DRY RUN]" if mode.dry else ""}\n{"=" * 70}')
    print(f'tasks   {list(tasks)}')
    print(f'train   {len(items)} items ({per_task}/task); measured ZPD yield '
          f'~{y:.0%} -> ~{n_zpd} items get a teacher CoT')
    if not mode.dry:
        print(f'probe   ~${c_probe:.0f} at most (most attempts cached from stage A)')
        print(f'teacher ~${c_teach:.0f} on {teacher} — the dominant line')
        if _pricing_caveat():
            print(_pricing_caveat())
        if n_zpd < 15:
            print(f'\nWARNING: only ~{n_zpd} ZPD items across {len(tasks)} tasks. That '
                  'yields maybe a\nhundred chunks, thin enough that retrieval may find '
                  'nothing — which would look\nlike a gate failure rather than a data '
                  'shortage. Consider raising --deepen-per-task.')
        confirm('proceed?', assume_yes)

    sh(['bbeh.probe', '--model', model, '--k', str(k), '--tasks', *tasks,
        '--limit-per-task', str(per_task), '--max-workers', str(workers)]
       + mode.flag, 'deepen probe')
    sh(['bbeh.teacher', '--model', teacher, '--student-model', model,
        '--select', 'zpd', '--tasks', *tasks,
        '--limit-per-task', str(per_task), '--max-workers', str(workers)]
       + mode.flag, 'teacher')
    sh(['bbeh.abstract', '--teacher-model', teacher, '--tasks', *tasks,
        '--max-workers', str(workers)] + mode.flag, 'abstract')
    sh(['bbeh.build_memory', 'build', '--version-id', mode.version,
        '--method', 'zpd', '--student-model', model, '--teacher-model', teacher,
        '--tasks', *tasks, '--overwrite'] + mode.flag, 'build bank')
    sh(['bbeh.build_memory', 'verify', mode.version], 'verify alignment')


# ═════════════════════════════════════════════════════════════════════
#  Stage: eval
# ═════════════════════════════════════════════════════════════════════

def stage_eval(tasks: Sequence[str], model: str, judge: str, per_task: int,
               workers: int, assume_yes: bool, mode: Mode) -> None:
    items = data.select_items(data.load_split('test'), tasks=tasks,
                              limit_per_task=per_task)
    c_solve = 2 * _cost(model, _tok(items), len(items) * OUT_TOK_SOLVE)
    n_judge = len(items) * config.RERANK_SAMPLES_N
    c_judge = _cost(judge, n_judge * 2000, n_judge * OUT_TOK_JUDGE)

    print(f'\n{"=" * 70}\nSTAGE D  both arms + analysis'
          f'{"  [DRY RUN]" if mode.dry else ""}\n{"=" * 70}')
    print(f'{len(items)} test items x 2 arms on {model}')
    if not mode.dry:
        print(f'solve  ~${c_solve:.1f}')
        print(f'Stage-3 judge: {n_judge} calls on {judge}   ~${c_judge:.1f}')
        if _pricing_caveat():
            print(_pricing_caveat())
        confirm('proceed?', assume_yes)

    sh(['bbeh.run', '--arm', 'no_memory', '--model', model, '--tasks', *tasks,
        '--limit-per-task', str(per_task), '--max-workers', str(workers)]
       + mode.flag, 'no_memory arm')
    sh(['bbeh.run', '--arm', 'memory', '--memory-version', mode.version,
        '--model', model, '--judge-model', judge, '--tasks', *tasks,
        '--limit-per-task', str(per_task), '--max-workers', str(workers)]
       + mode.flag, 'memory arm')
    sh(['bbeh.analyze', '--model', model, '--main-arm', mode.memory_arm,
        '--baseline', mode.baseline_arm, '--no-per-task']
       + (['--include-dryrun'] if mode.dry else []), 'analysis')

    print(f'\n{"=" * 70}\nWHAT TO READ\n{"=" * 70}')
    print('The pilot cannot settle either claim — 60 items is far too few, and')
    print('analyze.py will most likely say INCONCLUSIVE. Read it for machinery:')
    print('  1. the teacher VERDICT block (stage C output). It reads the rate against')
    print('     a 50% floor AND splits rejections into capability vs format, because')
    print('     the same rate means "fix the prompt" or "the teacher is a peer of the')
    print('     student" depending on the mix. Matters most when teacher and student')
    print('     are the same tier, which is when the pool quietly goes thin.')
    print('  2. injection rate (memory arm summary). Below 15% the gate is refusing')
    print('     nearly everything and the memory arm IS the baseline.')
    print('  3. same-task vs cross-task share (injection diagnostics). If it is ~100%')
    print('     same-task, this is per-task few-shot retrieval and the cross-task')
    print('     mechanism-transfer story is not being tested.')
    print('  4. truncation count. A wave of it means raise --max-tokens; those are')
    print('     format failures being scored as reasoning failures.')


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description='Staged BBEH pilot: 3 tasks x 20 items, end to end',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Stages (each')[-1])
    p.add_argument('stage', choices=('ping', 'check', 'probe', 'pick',
                                     'build', 'eval', 'all'))
    p.add_argument('--model', default=config.STUDENT_MODEL, help='the student')
    p.add_argument('--teacher', default=config.TEACHER_MODEL)
    p.add_argument('--judge', default=config.JUDGE_MODEL)
    p.add_argument('--tasks', nargs='*', default=None,
                   help='override the pilot task selection')
    p.add_argument('--k', type=int, default=config.PROBE_K)
    p.add_argument('--probe-per-task', type=int, default=PROBE_PER_TASK)
    p.add_argument('--deepen-per-task', type=int, default=DEEPEN_PER_TASK)
    p.add_argument('--test-per-task', type=int, default=TEST_PER_TASK)
    p.add_argument('--n-tasks', type=int, default=N_PILOT_TASKS)
    p.add_argument('--exclude', nargs='*',
                   default=['bbeh_shuffled_objects', 'bbeh_zebra_puzzles'],
                   help='tasks to keep out of the pick (default: the two largest)')
    p.add_argument('--max-workers', type=int, default=8)
    p.add_argument('--base-url', default='',
                   help='ping: override the endpoint for this call only')
    p.add_argument('--api-key', default='',
                   help='ping: override the key for this call only')
    p.add_argument('--auth-scheme', default='', choices=('', 'bearer', 'x-api-key'),
                   help='ping: how the credential goes on the wire, for this call only')
    p.add_argument('--sweep', action='store_true',
                   help='ping: try endpoint x auth-scheme with one model, to '
                        'separate a bad key from a bad host from a bad header')
    p.add_argument('--yes', action='store_true', help='skip confirmations')
    p.add_argument('--dry-run', action='store_true',
                   help='fabricate every response; $0. Exercises the real code '
                        'paths in a DRYRUN- namespace. Run this once first.')
    a = p.parse_args()

    config.ensure_dirs()
    mode = Mode(a.dry_run)
    if a.dry_run:
        print('[DRY RUN] every stage fabricates locally and writes only to '
              'DRYRUN- paths.\n          Numbers below are invented — this checks '
              'plumbing, not results.')

    if a.stage in ('ping', 'all'):
        # Skipped under --dry-run: a dry run promises zero network and zero
        # spend, and honouring that matters more than the reachability check,
        # which is meaningless for a run that will not call the API anyway.
        if a.dry_run:
            print('\n[DRY RUN] skipping the endpoint ping — no network is touched.')
        elif a.sweep:
            stage_ping_sweep(a.model, a.api_key)
        else:
            stage_ping([a.model, a.teacher, a.judge], a.base_url, a.api_key,
                       a.auth_scheme)
    if a.stage in ('check', 'all'):
        stage_check()
    if a.stage in ('probe', 'all'):
        stage_probe(a.model, a.k, a.probe_per_task, a.max_workers, a.yes, mode)
    if a.stage in ('pick', 'all'):
        tasks = stage_pick(a.model, a.n_tasks, a.exclude, a.tasks, mode)
        if a.stage == 'all' and not a.dry_run:
            confirm(f'\nrun the rest of the pilot on {tasks}?', a.yes)
    if a.stage in ('build', 'all'):
        tasks = load_tasks(a.tasks)
        stage_build(tasks, a.model, a.teacher, a.k, a.deepen_per_task,
                    max(2, a.max_workers // 2), a.yes, mode)
    if a.stage in ('eval', 'all'):
        tasks = load_tasks(a.tasks)
        stage_eval(tasks, a.model, a.judge, a.test_per_task, a.max_workers,
                   a.yes, mode)


if __name__ == '__main__':
    main()
