"""Tests for trajectory compaction in mini_swe_agent._compact_local_messages."""
from __future__ import annotations

import copy
from typing import Any, Dict, List

import pytest

from openjarvis.agents.hybrid.mini_swe_agent import (
    _compact_local_messages,
    _estimate_prompt_tokens,
    _get_tiktoken_enc,
)


TIKTOKEN_OK = bool(_get_tiktoken_enc())


def _validate_openai_message_shape(messages: List[Dict[str, Any]]) -> None:
    seen_tool_call_ids: set[str] = set()
    for i, m in enumerate(messages):
        role = m.get("role")
        assert role in {"system", "user", "assistant", "tool"}, f"bad role at {i}: {role}"
        if role == "assistant":
            for tc in (m.get("tool_calls") or []):
                tid = tc.get("id")
                assert tid, f"assistant tool_call missing id at msg {i}"
                seen_tool_call_ids.add(tid)
        elif role == "tool":
            tid = m.get("tool_call_id")
            assert tid, f"tool message at {i} missing tool_call_id"
            assert tid in seen_tool_call_ids, (
                f"tool message at {i} references unknown tool_call_id={tid!r}"
            )
            assert m.get("content"), f"tool message at {i} has empty content"


def _make_synthetic_messages() -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Fix the bug in foo.py."},
    ]
    big_idx = 7  # one of the assistant+tool pairs gets a giant blob
    for t in range(19):
        tc_id = f"call_{t:03d}"
        messages.append({
            "role": "assistant",
            "content": f"Turn {t}: I'll run a command.",
            "tool_calls": [{
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": '{"command": "ls -la dir_' + str(t) + '"}',
                },
            }],
        })
        if t == big_idx:
            body = "OUTPUT " + ("x" * 29950)
            obs = f"$ ls\n{body}\nexit_code=0"
        else:
            body = ("line " + str(t) + " ") * 800  # ~8000 chars
            obs = f"$ ls\n{body}\nexit_code=0"
        messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": obs[:8000] if t != big_idx else obs[:30000],
        })
    return messages


def test_compaction_token_budget_and_shape():
    messages = _make_synthetic_messages()
    orig = copy.deepcopy(messages)
    before = _estimate_prompt_tokens(messages)
    assert before > 24_000, f"sanity: synthetic should exceed budget, got {before}"

    compacted = _compact_local_messages(
        messages,
        client=None,  # forces stage-2 summary fallback (no API call)
        model="dummy",
        keep_last=4,
        trace_prefix="test",
        compact_at_tokens=24_000,
    )

    _validate_openai_message_shape(compacted)

    if TIKTOKEN_OK:
        after = _estimate_prompt_tokens(compacted)
        assert after <= 24_000, f"after compaction tokens={after} > 24_000"

    # System + initial user intact.
    assert compacted[0] == orig[0]
    assert compacted[1] == orig[1]

    # Recent 4 turns (assistant + its tool replies) intact and deep-equal.
    # In synthetic input, every turn is exactly 1 assistant + 1 tool, so the
    # last 4 turns = last 8 messages.
    assert compacted[-8:] == orig[-8:], "recent 4 turns must be intact"


def test_stage1_only_when_sufficient():
    """If stage 1 alone drops us below the budget, stage 2 must not run."""
    # Build a smaller input that exceeds budget only because of giant tool outputs.
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    for t in range(8):
        tc_id = f"call_{t}"
        messages.append({
            "role": "assistant",
            "content": f"t{t}",
            "tool_calls": [{
                "id": tc_id, "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            }],
        })
        # Use random-ish text to avoid BPE collapse from repetition.
        import random
        rng = random.Random(t)
        body = " ".join(rng.choice(["alpha", "beta", "gamma", "delta", "lambda",
                                     "foo", "bar", "baz", "qux", "zeta"])
                         for _ in range(3500))
        messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": body + " exit_code=0",
        })

    before = _estimate_prompt_tokens(messages)
    if not TIKTOKEN_OK:
        pytest.skip("tiktoken unavailable; stage-1-only ordering is timing-sensitive without it")
    assert before > 24_000

    compacted = _compact_local_messages(
        messages, client=None, model="dummy",
        keep_last=4, trace_prefix="test", compact_at_tokens=24_000,
    )
    _validate_openai_message_shape(compacted)
    # No stage-2 fold → length identical (only tool contents shortened).
    assert len(compacted) == len(messages)
    # The synthetic stage-2 system stub should NOT be present.
    for m in compacted:
        if m.get("role") == "system":
            assert not str(m.get("content", "")).startswith("[turns "), \
                "stage 2 should not have run"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
