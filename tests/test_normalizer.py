"""normalizer 离线单测：不联网、不起容器。"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.normalizer import normalize_request, summarize_actions, _supports_reasoning


def _base_body(**over):
    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
        ],
    }
    body.update(over)
    return body


def test_strip_top_level_strict():
    body = _base_body(strict=True)
    out = normalize_request(body)
    assert "strict" not in out
    assert body["strict"] is True


def test_strip_tool_level_strict():
    body = _base_body(tools=[{"type": "function", "function": {
        "name": "f", "description": "d",
        "parameters": {"type": "object", "properties": {}}, "strict": True}}])
    out = normalize_request(body)
    assert "strict" not in out["tools"][0]["function"]


def test_strip_response_format_strict():
    body = _base_body(response_format={"type": "json_schema", "json_schema": {
        "name": "x", "strict": True, "schema": {}}})
    out = normalize_request(body)
    assert "strict" not in out["response_format"]["json_schema"]


def test_no_strict_no_change():
    body = _base_body()
    assert normalize_request(body) == body
    assert summarize_actions(body) == []


def test_repair_orphan_tool_call():
    body = _base_body(messages=[
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
    ])
    out = normalize_request(body)
    tail = out["messages"][-1]
    assert tail["role"] == "tool"
    assert tail["tool_call_id"] == "call_1"


def test_paired_tool_call_untouched():
    body = _base_body(messages=[
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "42"},
    ])
    out = normalize_request(body)
    assert len(out["messages"]) == len(body["messages"])


def test_partial_orphan_only_fills_missing():
    body = _base_body(messages=[
        {"role": "user", "content": "do two"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            {"id": "call_b", "type": "function", "function": {"name": "g", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_a", "content": "ok"},
    ])
    out = normalize_request(body)
    added = [m for m in out["messages"]
             if m.get("role") == "tool" and m.get("tool_call_id") == "call_b"]
    assert len(added) == 1


def test_reinject_missing_reasoning_on_reasoner():
    body = _base_body(model="deepseek-reasoner", messages=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    out = normalize_request(body)
    assert out["messages"][1]["reasoning_content"] == ""


def test_existing_reasoning_preserved():
    body = _base_body(model="deepseek-reasoner", messages=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "reasoning_content": "I thought..."},
    ])
    out = normalize_request(body)
    assert out["messages"][1]["reasoning_content"] == "I thought..."


def test_remove_reasoning_on_non_reasoning_model():
    body = _base_body(model="deepseek-chat", messages=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "reasoning_content": "old"},
    ])
    out = normalize_request(body)
    assert "reasoning_content" not in out["messages"][1]


def test_keep_reasoning_on_reasoning_model():
    body = _base_body(model="deepseek-reasoner", messages=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "reasoning_content": "old"},
    ])
    out = normalize_request(body)
    assert out["messages"][1]["reasoning_content"] == "old"


def test_supports_reasoning_detection():
    assert _supports_reasoning("deepseek-reasoner")
    assert _supports_reasoning("deepseek-r1")
    assert not _supports_reasoning("deepseek-chat")
    assert not _supports_reasoning("gpt-4o")


def test_combined_and_idempotent():
    body = _base_body(model="deepseek-chat", strict=True, messages=[
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "reasoning_content": "chain", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
    ])
    once = normalize_request(body)
    twice = normalize_request(once)
    assert once == twice
    assert "strict" not in once
    assert once["messages"][-1]["role"] == "tool"
    assert "reasoning_content" not in once["messages"][1]


def test_summarize_actions_reports_all():
    body = _base_body(model="deepseek-chat", strict=True, messages=[
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "reasoning_content": "chain", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
    ])
    actions = summarize_actions(body)
    assert "strip_strict" in actions
    assert "repair_orphan_tool_calls" in actions
    assert "normalize_reasoning_cross_model" in actions


def test_empty_body_safe():
    for weird in ({}, {"messages": None}, {"messages": "nope"}, {"tools": "x"}):
        out = normalize_request(copy.deepcopy(weird))
        assert isinstance(out, dict)
