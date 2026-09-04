"""Locate which layer breaks mem0's LLM call. Run inside the failing environment.

    python -m eval.diagnose_mem0

Tests four layers with the same gateway, from the bottom up. The first one that
fails is the culprit:

  1. raw urllib          no openai SDK, no proxy handling beyond urllib's
  2. openai SDK          the client our harness uses successfully
  3. mem0's LLM object   same SDK, but constructed by mem0
  4. realistic payload   a full-size session, since size can trigger timeouts

Prints proxy environment variables first: an HTTPS_PROXY set for reaching
foreign endpoints will also capture a domestic gateway, which typically shows
up as a slow, regular-interval "Connection error" rather than a refusal.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

# Must precede any mem0 import: it reads this at module load time.
os.environ.setdefault("MEM0_TELEMETRY", "False")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.config import API_KEY, BASE_URL, GENERATION_MODEL  # noqa: E402

BIG = ("Speaker User: I want to plan a two-week trip to Kyoto in April, budget "
       "around 120000 JPY, and I would prefer ryokan over hotels.\n"
       "Speaker Assistant: Noted. Kyoto in April means cherry blossom season, so "
       "accommodation books out early.\n") * 22   # ~6k chars, a real session


def hdr(t):
    print(f"\n{'=' * 66}\n{t}\n{'=' * 66}")


def show_env():
    hdr("0. environment")
    print(f"  python              {sys.version.split()[0]}")
    for mod in ("openai", "httpx", "httpcore", "mem0", "sentence_transformers"):
        try:
            m = __import__(mod)
            print(f"  {mod:20s}{getattr(m, '__version__', '?')}")
        except Exception as e:
            print(f"  {mod:20s}absent ({type(e).__name__})")
    print()
    any_proxy = False
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy",
                "https_proxy", "all_proxy", "NO_PROXY", "no_proxy"):
        v = os.environ.get(var)
        if v:
            any_proxy = True
            print(f"  {var:20s}{v}")
    if any_proxy:
        host = BASE_URL.split("//")[-1].split("/")[0]
        print(f"\n  ! A proxy is configured. If it cannot reach {host}, requests")
        print(f"    stall and retry — exactly the symptom being debugged.")
        print(f"    Try:  export NO_PROXY=$NO_PROXY,{host}")
    else:
        print("  (no proxy variables set)")
    print(f"\n  gateway             {BASE_URL}")
    print(f"  model               {GENERATION_MODEL}")


def t1_urllib():
    hdr("1. raw urllib (no openai SDK)")
    body = json.dumps({"model": GENERATION_MODEL,
                       "messages": [{"role": "user", "content": "say hi"}]}).encode()
    req = urllib.request.Request(
        BASE_URL + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            json.loads(r.read())
        print(f"  OK  {time.time() - t0:.1f}s")
        return True
    except Exception as e:
        print(f"  FAILED after {time.time() - t0:.1f}s: {type(e).__name__}: {e}")
        return False


def t2_sdk():
    hdr("2. openai SDK (what our harness uses)")
    try:
        from openai import OpenAI
    except ImportError as e:
        print(f"  openai not importable: {e}")
        return False
    t0 = time.time()
    try:
        c = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        c.chat.completions.create(model=GENERATION_MODEL,
                                  messages=[{"role": "user", "content": "say hi"}],
                                  timeout=90)
        print(f"  OK  {time.time() - t0:.1f}s")
        return True
    except Exception as e:
        print(f"  FAILED after {time.time() - t0:.1f}s: {type(e).__name__}: {e}")
        return False


def t3_mem0_llm():
    hdr("3. mem0's own LLM object (JSON mode, as used for extraction)")
    try:
        from mem0.llms.openai import OpenAILLM
        from mem0.configs.llms.openai import OpenAIConfig
    except ImportError as e:
        print(f"  mem0 not importable: {e}")
        return False
    t0 = time.time()
    try:
        llm = OpenAILLM(OpenAIConfig(model=GENERATION_MODEL, api_key=API_KEY,
                                     openai_base_url=BASE_URL))
        print(f"  client base_url: {llm.client.base_url}")
        out = llm.generate_response(
            messages=[{"role": "user",
                       "content": 'Return JSON: {"facts": ["user likes tea"]}'}],
            response_format={"type": "json_object"})
        print(f"  OK  {time.time() - t0:.1f}s  -> {str(out)[:80]!r}")
        return True
    except Exception as e:
        print(f"  FAILED after {time.time() - t0:.1f}s: {type(e).__name__}: {e}")
        return False


def t4_big():
    hdr("4. realistic payload (~6k chars, one full session)")
    try:
        from openai import OpenAI
    except ImportError:
        print("  skipped")
        return False
    t0 = time.time()
    try:
        c = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        r = c.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user",
                       "content": "Extract the user's facts as JSON.\n\n" + BIG}],
            response_format={"type": "json_object"}, timeout=120)
        print(f"  OK  {time.time() - t0:.1f}s  "
              f"in~{len(BIG)//4} tokens, out={len(r.choices[0].message.content or '')} chars")
        return True
    except Exception as e:
        print(f"  FAILED after {time.time() - t0:.1f}s: {type(e).__name__}: {e}")
        return False


def main():
    show_env()
    results = [("raw urllib", t1_urllib()), ("openai SDK", t2_sdk()),
               ("mem0 LLM", t3_mem0_llm()), ("big payload", t4_big())]
    hdr("verdict")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [n for n, ok in results if not ok]
    if not failed:
        print("\n  Every layer works. The failure is intermittent — likely gateway\n"
              "  rate limiting under mem0's request volume. Retry, or lower\n"
              "  concurrency; mem0 issues one extraction call per session.")
    elif failed == ["big payload"]:
        print("\n  Only the large payload fails: a size or duration limit on the\n"
              "  gateway. mem0 sends whole sessions, so this blocks it entirely.")
    elif "raw urllib" in failed:
        print("\n  Even raw urllib fails: this is network/proxy, not mem0 and not\n"
              "  the SDK. Check the proxy variables printed above.")
    elif "mem0 LLM" in failed and "openai SDK" not in failed:
        print("\n  The SDK works but mem0's client does not — a mem0-specific\n"
              "  construction issue. Send this output back.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
