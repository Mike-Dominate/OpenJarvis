"""Smoke tests for the OpenAI SDK retry + per-org concurrency hardening.

We deliberately don't hit the real OpenAI API. Instead we monkey-patch
the underlying call to simulate the failure modes we care about:

- Sustained ``RateLimitError`` walls.
- ``APITimeoutError`` blips.
- ``APIConnectionError`` blips.
- ``InternalServerError`` 5xx blips.

Then we hammer ``openai.OpenAI().chat.completions.create`` from a thread
pool with a deliberately tight semaphore (concurrency=2, retries=4) and
confirm:

(a) no exceptions escape when the failure clears within the retry budget,
(b) backoff actually fires (call count > attempted call count),
(c) all requests eventually complete,
(d) when the failure exceeds the retry budget, the final exception
    propagates so the runner records ``error=...`` (no silent drop),
(e) local vLLM-style clients (api_key="EMPTY" or localhost base_url)
    bypass both the throttle and the retry loop.

Run with:

    .venv/bin/python -m pytest tests/agents/hybrid/test_openai_retry.py -v
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List
from unittest.mock import MagicMock

import pytest

# Make sure the patch picks up tight test settings, not the prod defaults.
# Must be set BEFORE _openai_retry is imported.
os.environ["OPENJARVIS_OPENAI_MAX_CONCURRENCY"] = "2"
os.environ["OPENJARVIS_OPENAI_MAX_RETRIES"] = "4"
os.environ["OPENJARVIS_OPENAI_RETRY_BASE"] = "0.05"   # fast tests
os.environ["OPENJARVIS_OPENAI_RETRY_CAP"] = "0.2"


# Reload the module fresh so env vars take effect (imports earlier in
# the process may have frozen the defaults).
import importlib

from openjarvis.agents.hybrid import _openai_retry as _retry_mod

importlib.reload(_retry_mod)
_retry_mod.patch_openai_globally()


def _make_fake_response() -> Any:
    """Minimal stand-in for an OpenAI ChatCompletion response."""
    r = MagicMock()
    r.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None))]
    r.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
    return r


def _swap_underlying_create(
    monkeypatch: pytest.MonkeyPatch, side_effect: Any,
) -> List[int]:
    """Replace the wrapped-underlying-orig with one that fires ``side_effect``.

    Returns a mutable call-counter list so tests can assert on attempts.
    """
    from openai.resources.chat import completions as _comp_mod

    counter: List[int] = [0]
    wrapped = _comp_mod.Completions.create
    # The patcher stashed the original at ``__wrapped__``.
    orig = getattr(wrapped, "__wrapped__", None)
    assert orig is not None, "patch_openai_globally didn't stash __wrapped__"

    def fake(self: Any, *args: Any, **kwargs: Any) -> Any:
        counter[0] += 1
        if callable(side_effect):
            return side_effect(counter[0])
        if isinstance(side_effect, list):
            i = min(counter[0] - 1, len(side_effect) - 1)
            v = side_effect[i]
            if isinstance(v, BaseException):
                raise v
            return v
        if isinstance(side_effect, BaseException):
            raise side_effect
        return side_effect

    # Replace the wrapped underlying directly.
    new_wrap = _retry_mod._wrap_create(fake)
    monkeypatch.setattr(_comp_mod.Completions, "create", new_wrap)
    return counter


def _rate_limit_error() -> BaseException:
    """Build an ``openai.RateLimitError`` that the SDK would normally raise."""
    import openai

    # The SDK's RateLimitError wants (message, response, body) — we use a
    # MagicMock for response so ``response.headers.get("retry-after")``
    # returns None.
    resp = MagicMock()
    resp.headers = {}
    resp.status_code = 429
    return openai.RateLimitError("rate limit", response=resp, body=None)


def _api_timeout_error() -> BaseException:
    import openai

    return openai.APITimeoutError(request=MagicMock())


def _api_conn_error() -> BaseException:
    import openai

    return openai.APIConnectionError(request=MagicMock())


def _internal_500_error() -> BaseException:
    import openai

    resp = MagicMock()
    resp.headers = {}
    resp.status_code = 500
    return openai.InternalServerError("server error", response=resp, body=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_patches_installed() -> None:
    import openai
    from openai.resources.chat import completions

    assert getattr(completions.Completions.create, "_hybrid_patched", False)
    assert getattr(openai.OpenAI.__init__, "_hybrid_patched", False)


def test_rate_limit_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two 429s then a real response — retry path should swallow both."""
    import openai

    counter = _swap_underlying_create(
        monkeypatch,
        [_rate_limit_error(), _rate_limit_error(), _make_fake_response()],
    )
    client = openai.OpenAI(api_key="sk-fake")
    resp = client.chat.completions.create(
        model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}]
    )
    assert resp.choices[0].message.content == "ok"
    assert counter[0] == 3  # backoff fired twice, then succeeded


def test_timeout_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    counter = _swap_underlying_create(
        monkeypatch, [_api_timeout_error(), _make_fake_response()]
    )
    client = openai.OpenAI(api_key="sk-fake")
    resp = client.chat.completions.create(
        model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}]
    )
    assert resp.choices[0].message.content == "ok"
    assert counter[0] == 2


def test_500_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    counter = _swap_underlying_create(
        monkeypatch, [_internal_500_error(), _make_fake_response()]
    )
    client = openai.OpenAI(api_key="sk-fake")
    resp = client.chat.completions.create(
        model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}]
    )
    assert resp.choices[0].message.content == "ok"
    assert counter[0] == 2


def test_retry_exhaustion_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sustained 429 wall beyond retry budget → exception propagates."""
    import openai

    # max_retries=4 → 5 total attempts; feed 6 errors so the loop runs out.
    errors = [_rate_limit_error() for _ in range(6)]
    counter = _swap_underlying_create(monkeypatch, errors)
    client = openai.OpenAI(api_key="sk-fake")
    with pytest.raises(openai.RateLimitError):
        client.chat.completions.create(
            model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}]
        )
    # 5 attempts: 1 initial + 4 retries.
    assert counter[0] == 5


def test_non_retryable_propagates_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """``BadRequestError`` is NOT retryable — should raise on attempt 1."""
    import openai

    resp = MagicMock()
    resp.headers = {}
    resp.status_code = 400
    bad = openai.BadRequestError("nope", response=resp, body=None)
    counter = _swap_underlying_create(monkeypatch, [bad])
    client = openai.OpenAI(api_key="sk-fake")
    with pytest.raises(openai.BadRequestError):
        client.chat.completions.create(
            model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}]
        )
    assert counter[0] == 1  # no retries


def test_local_vllm_bypasses_throttle_and_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local vLLM clients (api_key=EMPTY) must NOT pay the per-org throttle.

    Verifies (a) detection works, (b) a non-retryable error fires only
    once (no retry attempts for local endpoints — they're either up or
    down). Also (c) the semaphore isn't acquired (which we can't directly
    observe, but we can confirm the wrapper short-circuits).
    """
    import openai

    counter = _swap_underlying_create(monkeypatch, _rate_limit_error())
    client = openai.OpenAI(base_url="http://localhost:8001/v1", api_key="EMPTY")
    # Should raise on attempt 1 (no retries for local endpoints).
    with pytest.raises(openai.RateLimitError):
        client.chat.completions.create(
            model="qwen", messages=[{"role": "user", "content": "hi"}]
        )
    assert counter[0] == 1


def test_concurrent_hammering_no_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    """16 concurrent calls, each fails twice with 429 then succeeds.

    With concurrency=2 the semaphore queues, with retry budget=4 every
    call eventually gets through. We confirm no exception escapes and
    all 16 calls complete with the fake response.
    """
    import openai

    # Per-call counter for failure injection.
    call_state: dict = {}
    state_lock = threading.Lock()

    def side_effect(call_idx: int) -> Any:
        # Tag by thread so each "logical call" fails its first 2 attempts.
        tid = threading.get_ident()
        with state_lock:
            n = call_state.get(tid, 0) + 1
            call_state[tid] = n
        if n <= 2:
            raise _rate_limit_error()
        # Reset for the next logical call from this thread.
        with state_lock:
            call_state[tid] = 0
        return _make_fake_response()

    _swap_underlying_create(monkeypatch, side_effect)
    client = openai.OpenAI(api_key="sk-fake")

    def one() -> str:
        r = client.chat.completions.create(
            model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}]
        )
        return r.choices[0].message.content

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(one) for _ in range(16)]
        results = [f.result() for f in as_completed(futures)]
    elapsed = time.time() - t0
    assert all(r == "ok" for r in results)
    assert len(results) == 16
    # Sanity: with concurrency=2 + backoff>=0.05s * 2 retries per call,
    # 16 calls can't possibly finish instantly. Just confirm we spent
    # *some* time in backoff, not that we measured it precisely.
    assert elapsed > 0.1


def test_retry_after_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the SDK exposes a Retry-After header, we sleep at least that long."""
    import openai

    resp = MagicMock()
    resp.headers = {"retry-after": "0.15"}
    resp.status_code = 429
    err = openai.RateLimitError("rate limit", response=resp, body=None)

    counter = _swap_underlying_create(monkeypatch, [err, _make_fake_response()])
    client = openai.OpenAI(api_key="sk-fake")
    t0 = time.time()
    client.chat.completions.create(
        model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}]
    )
    elapsed = time.time() - t0
    assert counter[0] == 2
    # We sleep at least 0.15s (Retry-After) — jitter is additive ≤ 0.5.
    assert elapsed >= 0.15
