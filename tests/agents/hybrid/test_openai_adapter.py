"""Regression tests for the mini-SWE-agent OpenAI cloud + bash adapter.

Two failure modes caught in the n=100 hybrid SWE sweep (May 2026) — both
deterministic enough to pin down here without hitting any real model or
shelling out for real binary data:

1. **Bug 1 — null assistant content.** ``_loop_cloud_openai`` used to
   append ``{"role": "assistant", "content": text or None}`` on each
   turn. On a tool-only turn (model produced no text alongside its
   ``bash`` call) that wrote ``content: null`` into the message list;
   the next ``chat.completions.create`` then 400'd with
   ``Invalid value for 'content': expected a string, got null``. Fix
   uses ``""`` (or omitted) per OpenAI's schema.

2. **Bug 3 — binary bash output crashes the loop.** ``_run_bash`` used
   to pass ``text=True`` to ``subprocess.Popen``, so any command that
   emitted non-UTF-8 bytes (cat'ing a compiled artifact, an image, a
   PDF) raised ``UnicodeDecodeError`` from inside
   ``Popen.communicate()`` and killed the whole task. Fix captures
   bytes and decodes via ``_decode_bash_output`` with ``errors="replace"``
   plus a binary-detection stub.

Run with:

    .venv/bin/python -m pytest tests/agents/hybrid/test_openai_adapter.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from openjarvis.agents.hybrid.mini_swe_agent import (
    _decode_bash_output,
    _loop_cloud_openai,
    _run_bash,
)


# ---------------------------------------------------------------------------
# Bug 3 — binary bash output
# ---------------------------------------------------------------------------


class TestDecodeBashOutput:
    def test_pure_ascii_passes_through(self) -> None:
        assert _decode_bash_output(b"hello world\n", 0) == "hello world\n"

    def test_empty_bytes_returns_empty_string(self) -> None:
        assert _decode_bash_output(b"", 0) == ""

    def test_nul_byte_substituted_with_stub(self) -> None:
        raw = b"some text\x00more bytes after"
        out = _decode_bash_output(raw, 0)
        assert out.startswith("[binary output:")
        assert f"{len(raw)} bytes" in out
        assert "exit=0" in out

    def test_invalid_utf8_partial_substitutes_replacement_char(self) -> None:
        # ~25% replacement chars after decode — well above the 5% binary
        # threshold, should swap to the stub. The exact byte sequence
        # 0xe0 is the one observed in the astropy__astropy-14539 row.
        raw = b"abc\xe0\xe0\xe0def"
        out = _decode_bash_output(raw, 1)
        assert out.startswith("[binary output:")
        assert "exit=1" in out

    def test_mostly_valid_utf8_keeps_decoded_text(self) -> None:
        # One stray bad byte in a long string → below 5% replacement,
        # keep the (mostly intact) decoded text rather than stubbing.
        raw = ("readable text " * 200).encode("utf-8") + b"\xe0"
        out = _decode_bash_output(raw, 0)
        assert "readable text" in out
        assert not out.startswith("[binary output:")

    def test_real_bash_run_on_binary_does_not_raise(
        self, tmp_path: Path
    ) -> None:
        # End-to-end: a model-issued ``head`` on a binary file. Pre-fix
        # this raised UnicodeDecodeError out of ``_run_bash`` →
        # propagated all the way up to the runner. Post-fix it returns
        # a normal observation dict with the binary-output stub.
        bin_path = tmp_path / "blob.bin"
        bin_path.write_bytes(bytes(range(256)) * 4)
        result = _run_bash(
            f"head -c 1024 {bin_path}", tmp_path,
            timeout=10, output_cap=10_000,
        )
        assert result["exit_code"] == 0
        assert result["timed_out"] is False
        assert "[binary output:" in result["stdout"]


# ---------------------------------------------------------------------------
# Bug 1 — null content on tool-only assistant turn
# ---------------------------------------------------------------------------


class _FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, tc_id: str, name: str, arguments: str) -> None:
        self.id = tc_id
        self.type = "function"
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content: Any, tool_calls: List[Any]) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(
        self, message: _FakeMessage, finish_reason: str = "tool_calls"
    ) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5


class _FakeResp:
    def __init__(self, choice: _FakeChoice) -> None:
        self.choices = [choice]
        self.usage = _FakeUsage()


class _FakeCompletions:
    """Records every (messages=...) the model is asked to score against.

    Returns a scripted sequence: turn 1 → tool_only (no text), turn 2 →
    final summary text, no tool_calls. We then assert that turn 2's
    inbound messages contain the turn-1 assistant message with
    ``content == ""`` (or omitted) — never ``None``, which is the bug.
    """

    def __init__(self, scripted: List[_FakeResp]) -> None:
        self._scripted = list(scripted)
        self.calls: List[List[Dict[str, Any]]] = []

    def create(self, **kwargs: Any) -> _FakeResp:
        # Deep enough copy so the agent appending to its messages list
        # post-call doesn't mutate what we recorded here.
        self.calls.append([dict(m) for m in kwargs["messages"]])
        return self._scripted.pop(0)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


def test_assistant_message_content_never_none_on_tool_only_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OpenAI SDK 400s with ``content: null`` on replay. We must
    serialize tool-only assistant turns as ``content == ""`` (or omit
    the field) — never ``None``. Pre-fix this test failed because turn
    2's recorded messages had ``messages[2]["content"] is None``.
    """

    tool_turn = _FakeResp(_FakeChoice(
        _FakeMessage(
            content=None,
            tool_calls=[_FakeToolCall("c1", "bash", '{"command": "echo hi"}')],
        ),
    ))
    done_turn = _FakeResp(_FakeChoice(
        _FakeMessage(content="all done", tool_calls=[]),
        finish_reason="stop",
    ))
    fake = _FakeCompletions([tool_turn, done_turn])
    fake_client = _FakeClient(fake)

    def _fake_openai_ctor(**kwargs: Any) -> _FakeClient:
        return fake_client

    # Swap in our fake at the import site inside _loop_cloud_openai.
    import openai
    monkeypatch.setattr(openai, "OpenAI", _fake_openai_ctor)

    out = _loop_cloud_openai(
        "fake problem", tmp_path,
        model="gpt-5-mini-2025-08-07",
        max_turns=4, bash_timeout=10, output_cap=10_000,
        turn_max_tokens=64, trace_prefix="test",
    )

    # Two real model calls: tool-issuing turn + final turn.
    assert len(fake.calls) == 2

    second_call_messages = fake.calls[1]
    # First two messages are system + user. The assistant turn from turn
    # 1 should be index 2; its content must be a string (empty is fine),
    # never None — that's the bug we're regressing.
    assistant_msg = second_call_messages[2]
    assert assistant_msg["role"] == "assistant"
    assert "content" in assistant_msg
    assert assistant_msg["content"] is not None
    assert isinstance(assistant_msg["content"], str)
    # Tool calls must still be present (we only fixed the content shape).
    assert assistant_msg.get("tool_calls")

    # Sanity: the loop terminated normally on the no-tool turn.
    assert out["final_summary"] == "all done"
    assert out["turns"] == 2
