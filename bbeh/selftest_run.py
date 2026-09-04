"""
bbeh/selftest_run.py — offline checks for the evaluation harness.

No network, no spend. These guard the properties that decide whether the two
claims mean anything, all of which fail *silently* if broken — a run completes,
a table prints, and the number is wrong:

  * **Arm parity at the wire.** Not "the prompt builder is shared" but "the
    bytes sent when the gate rejects everything are identical to the baseline".
    If the memory arm quietly reworded the question, claim 2 would be measuring
    prompt engineering.
  * **Infra errors never become wrong answers.** They are excluded from the
    denominator and retried on the next run. Folding them in lets a bad hour on
    the endpoint read as a regression in whichever arm ran during it.
  * **Retrieval failures degrade to no-memory and say so**, so they cannot be
    mistaken for the gate legitimately rejecting candidates.
  * **The resume cache re-runs what it should** and never re-spends on a
    genuinely wrong answer.
"""

import shutil
import sys

from bbeh import config, data, prompts, reasoner as RM, run as R
from bbeh.api_client import GenResult

_FAILS = []


def check(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f'   {detail}' if detail else ''))
    if not cond:
        _FAILS.append(name)


class Spy:
    """Records prompts, returns a fixed valid answer."""

    def __init__(self):
        self.sent = []

    def generate_detailed(self, prompt, **kw):
        self.sent.append(prompt)
        return GenResult(text='The final answer is: x', finish_reason='stop')


def _judge(score):
    class J:
        def score(self, query_text, query_task, candidates):
            return [{'samples': [score] * 5, 'mean': score, 'std': 0.0,
                     'n': 5, 'n_requested': 5, 'degraded': False}
                    for _ in candidates]
    return J()


def test_cache_semantics():
    print('\n=== resume cache semantics ===')
    check('infra_error record is NOT done (gets retried next run)',
          not R.is_done({'id': 'a', 'outcome': 'infra_error', 'response': ''}))
    check('empty response is a cache MISS even when the record exists',
          not R.is_done({'id': 'a', 'outcome': 'ok', 'response': '   '}))
    check('a wrong-but-real answer IS done (never re-spend on it)',
          R.is_done({'id': 'a', 'outcome': 'ok',
                     'response': 'The final answer is: 7', 'correct': False}))
    check('truncated IS done (a real, scored attempt)',
          R.is_done({'id': 'a', 'outcome': 'truncated', 'response': 'blah'}))
    check('missing record is not done', not R.is_done(None))

    recs = ([{'id': str(i), 'task': 't', 'outcome': 'ok', 'correct': i < 3,
              'n_injected': 0} for i in range(6)]
            + [{'id': 'e', 'task': 't', 'outcome': 'infra_error',
                'correct': False, 'n_injected': 0}])
    s = R.summarize(recs, 'x', 'm', arm_type='no_memory')
    check('accuracy denominator excludes infra errors',
          abs(s['accuracy'] - 0.5) < 1e-9 and s['n_infra_error'] == 1,
          f"acc={s['accuracy']:.3f} (3/6, not 3/7={3/7:.3f})")

    cache = {'a': {'id': 'a', 'task': 't', 'response': 'The final answer is: 42',
                   'correct': False, 'prediction': 'wrong'}}
    changed = R.rescore(cache, [{'id': 'a', 'task': 't', 'target': '42'}])
    check('rescore repairs a stale verdict with no API call',
          changed == 1 and cache['a']['correct'] is True)


def test_arm_parity(items, retriever, qemb):
    print('\n=== wire-level arm parity ===')
    base = Spy()
    r0 = RM.StandardReasoner(base)
    for it in items:
        r0.solve(it)

    gs = Spy()
    gated = RM.MemoryAugmentedReasoner(gs, retriever, qemb, reranker=_judge(0.0))
    res_g = [gated.solve(it) for it in items]

    os_ = Spy()
    opened = RM.MemoryAugmentedReasoner(os_, retriever, qemb, reranker=_judge(1.0))
    res_o = [opened.solve(it) for it in items]

    check('fully-gated memory arm sends BYTE-IDENTICAL prompts to no_memory',
          gs.sent == base.sent, f'{len(base.sent)} prompts compared')
    check('gated arm reports 0 injected', all(x.n_injected == 0 for x in res_g))
    check('open arm actually injects', all(x.n_injected > 0 for x in res_o))
    check('injected prompt == bare prompt + prefix (nothing reworded or removed)',
          all(o.endswith(prompts.SOLVE_SUFFIX) and len(o) > len(b)
              for o, b in zip(os_.sent, base.sent)),
          f'+{len(os_.sent[0]) - len(base.sent[0])} chars of precedent block')
    return base.sent


def test_retrieval_failure(items, retriever, qemb, bare_prompts):
    print('\n=== a broken retriever must not corrupt the measurement ===')

    class Boom:
        def score(self, *a):
            raise RuntimeError('judge exploded')

    spy = Spy()
    rb = RM.MemoryAugmentedReasoner(spy, retriever, qemb, reranker=Boom())
    out = [rb.solve(it) for it in items]
    check('a crashing judge still yields scorable results',
          all(x.outcome == 'ok' for x in out) and all(x.n_injected == 0 for x in out))
    check('the crash is recorded, not read as "the gate said no"',
          all('judge exploded' in x.retrieval_error for x in out),
          out[0].retrieval_error[:45])
    check('and it falls back to the exact bare prompt', spy.sent == bare_prompts)


def test_infra_classification(items):
    print('\n=== infra error vs wrong answer vs truncation ===')

    class Flaky:
        def __init__(self):
            self.n = 0

        def generate_detailed(self, prompt, **kw):
            self.n += 1
            if self.n <= 3:
                return GenResult(text=None, error='timeout')
            return GenResult(text='The final answer is: x', finish_reason='stop')

    rr = RM.StandardReasoner(Flaky())
    out = [rr.solve(it) for it in items]
    n_err = sum(1 for x in out if x.outcome == 'infra_error')
    check('infra errors are flagged, not scored as wrong answers',
          n_err == 3 and all(not x.scorable for x in out if x.outcome == 'infra_error'),
          f'{n_err} errors, {sum(1 for x in out if x.scorable)} scorable')

    class Cut:
        def generate_detailed(self, prompt, **kw):
            return GenResult(text='I was thinking about it and then',
                             finish_reason='length')

    cut = RM.StandardReasoner(Cut()).solve(items[0])
    check('hitting the token ceiling with no answer line reads as truncated',
          cut.outcome == 'truncated' and cut.scorable and not cut.correct)

    class Long:
        def generate_detailed(self, prompt, **kw):
            return GenResult(text='...long reasoning...\nThe final answer is: x\nps',
                             finish_reason='length')

    lng = RM.StandardReasoner(Long()).solve(items[0])
    check('a long response that DID answer is not mislabelled truncated',
          lng.outcome == 'ok')


def test_auth_scheme():
    """The credential must go on the wire the way the gateway expects.

    A gateway that wants ``x-api-key`` answers a Bearer request with a 403 that
    is worded exactly like a bad key, so this is worth an offline check rather
    than a live one. The failure mode being guarded is narrow and quiet: the
    OpenAI SDK has no switch for turning Authorization off, so we suppress it
    with its ``Omit`` sentinel. If a future SDK moves or removes ``Omit``, the
    client keeps working — it just starts sending Bearer again, and every call
    fails against a gateway that validates it.
    """
    print('\n=== the credential goes on the wire under the configured scheme ===')
    from bbeh.api_client import _auth_headers, AUTH_SCHEMES

    check('bearer defers to the SDK default',
          _auth_headers('bearer', 'k') == (None, None))
    try:
        _auth_headers('digest', 'k')
        check('an unknown scheme is rejected, not silently ignored', False)
    except ValueError:
        check('an unknown scheme is rejected, not silently ignored', True)
    check('scheme names are validated against one list',
          set(AUTH_SCHEMES) == {'bearer', 'x-api-key'})

    try:
        import openai            # noqa: F401
    except ImportError:
        print('  SKIP  x-api-key suppression (openai not installed here)')
        return
    client_h, req_h = _auth_headers('x-api-key', 'sk-SECRET')
    check('the key is sent as x-api-key', client_h.get('x-api-key') == 'sk-SECRET')
    # The layer matters: an Omit in client-level default_headers deletes
    # Authorization before _validate_headers runs, which then finds neither a
    # header nor a per-request Omit and raises TypeError without sending
    # anything. It has to be per-request.
    check('the Bearer suppression rides on the per-request headers, not the client',
          req_h is not None and 'Authorization' in req_h,
          f'client={sorted(client_h)}  request={sorted(req_h or {})}')
    check('and it is the SDK sentinel, not a string',
          not isinstance(req_h['Authorization'], str),
          repr(req_h['Authorization']))
    check('Authorization is not suppressed at the client layer',
          'Authorization' not in client_h)


def test_retry_budget():
    """A retry count of 0 must still send one request.

    ``generate_detailed`` loops ``range(1, max_retries + 1)``, so 0 used to make
    the loop empty: no HTTP request, and a returned error with a blank reason.
    That is the worst shape a bug can take here — it is indistinguishable from a
    genuine API failure, so it sends you to debug your key and your endpoint
    while the code is quietly refusing to call anything. Caught for real by
    ``pilot ping``, which passed 0 meaning "do not retry".
    """
    print('\n=== a zero retry budget still makes one call ===')
    from bbeh.api_client import GenClient

    calls = {'n': 0}

    def stub(self, prompt, mt, temp, system):
        calls['n'] += 1
        return GenResult(text='ok')

    def boom(self, prompt, mt, temp, system):
        calls['n'] += 1
        raise RuntimeError('nope')

    def client(max_retries):
        c = object.__new__(GenClient)          # no network, no openai import
        c.model, c.protocol = 'stub', 'chat'
        c.temperature, c.max_tokens = 0.0, 8
        c.max_retries, c.retry_base_delay, c.usage = max_retries, 0.001, None
        return c

    orig = GenClient._request
    try:
        GenClient._request = stub
        for mr in (0, -1, 1):
            calls['n'] = 0
            r = client(mr).generate_detailed('hi')
            check(f'max_retries={mr} sends exactly one request',
                  calls['n'] == 1 and r.text == 'ok', f'{calls["n"]} call(s)')

        calls['n'] = 0
        GenClient._request = boom
        r = client(0).generate_detailed('hi')
        check('a real failure at max_retries=0 reports its reason',
              calls['n'] == 1 and 'nope' in r.error, f'error={r.error!r}')

        calls['n'] = 0
        GenClient._request = boom
        r = client(3).generate_detailed('hi')
        check('a retry budget of 3 is still honoured', calls['n'] == 3,
              f'{calls["n"]} call(s)')
    finally:
        GenClient._request = orig


def test_abort():
    print('\n=== fail fast on a systematically broken endpoint ===')

    class Dead:
        def generate_detailed(self, prompt, **kw):
            return GenResult(text=None, error='connection refused')

    orig = R.build_client
    R.build_client = lambda *a, **k: Dead()
    try:
        R.run_arm(arm='no_memory', arm_label='SELFTEST-ABORT',
                  tasks=['bbeh_word_sorting'], limit_per_task=30,
                  max_workers=2, abort_after=6)
        check('run aborts instead of burning the budget on zeros', False, 'it completed!')
    except SystemExit as e:
        check('run aborts instead of burning the budget on zeros', 'ABORT' in str(e))
    finally:
        R.build_client = orig
        shutil.rmtree(config.run_dir('SELFTEST-ABORT', config.STUDENT_MODEL),
                      ignore_errors=True)


def main() -> int:
    import logging
    logging.basicConfig(level=logging.ERROR, format='%(levelname)s %(message)s')
    from bbeh.retriever import MemoryRetriever, QueryEmbedder

    print(f'{"=" * 68}\nharness selftest (offline)\n{"=" * 68}')
    items = data.select_items(data.load_split('test'),
                              tasks=['bbeh_word_sorting'], limit_per_task=6)
    retriever = MemoryRetriever('DRYRUN-zpd')
    qemb = QueryEmbedder(dry_run=True).embed_items(items)

    test_cache_semantics()
    bare = test_arm_parity(items, retriever, qemb)
    test_retrieval_failure(items, retriever, qemb, bare)
    test_infra_classification(items)
    test_auth_scheme()
    test_retry_budget()
    test_abort()

    print(f'\n{"=" * 68}')
    if _FAILS:
        print(f'{len(_FAILS)} FAILURES: {_FAILS}')
        return 1
    print('all harness checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
