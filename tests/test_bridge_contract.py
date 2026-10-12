"""接线验证：不依赖网络，用 mock 上游验证 bridge 的请求/响应契约。

验证范围（明确写死，不膨胀）：
  - /bridge/dry-run 返回规范化动作与改动标记
  - /v1/chat/completions 把规范化后的 body 转发给上游
  - 上游 4xx/5xx 原样回传状态码
  - 非流式响应体原样回传
  - 非法 JSON 返回 400
  - 上游不可达返回 502 且标记为 bridge 错误
不验证：流式时序、真实上游、并发。

两种跑法：
  pytest tests/test_bridge_contract.py      （标准收集）
  python3 tests/test_bridge_contract.py     （自带 runner，无 pytest 依赖）
"""
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest  # noqa: F401  (声明测试依赖；手动 runner 不依赖它)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from fastapi.testclient import TestClient
from app.bridge import app

client = TestClient(app)

CLEAN = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]}
STRICT_BODY = {
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "hi"}],
    "response_format": {"type": "json_schema", "json_schema": {"strict": True}},
    "tools": [{"type": "function", "function": {"name": "f", "strict": True, "parameters": {}}}],
}


def _mock_response(status=200, body=b'{"ok":true}', headers=None):
    req = httpx.Request("POST", "http://upstream/v1/chat/completions")
    return httpx.Response(
        status_code=status,
        content=body,
        headers=headers or {"content-type": "application/json"},
        request=req,
    )


# --- dry-run -------------------------------------------------------------

def test_dry_run_strips_strict_and_reports_actions():
    r = client.post("/bridge/dry-run", json=STRICT_BODY)
    assert r.status_code == 200
    d = r.json()
    assert d["changed"] is True
    assert any("strict" in a for a in d["actions"])
    assert "strict" not in json.dumps(d["normalized"].get("response_format", {}))


def test_dry_run_leaves_clean_request_unchanged():
    r = client.post("/bridge/dry-run", json=CLEAN)
    assert r.status_code == 200
    assert r.json()["changed"] is False


# --- 转发 -----------------------------------------------------------------

def test_forwards_normalized_body_to_upstream():
    captured = {}

    async def fake_send(req, stream=False):
        captured["body"] = json.loads(req.content)
        captured["url"] = str(req.url)
        return _mock_response()

    with patch("httpx.AsyncClient.send", new=AsyncMock(side_effect=fake_send)):
        r = client.post("/v1/chat/completions", json=STRICT_BODY)

    assert r.status_code == 200
    assert captured["url"].endswith("/v1/chat/completions")
    assert "strict" not in json.dumps(captured["body"])
    assert r.json() == {"ok": True}


def test_upstream_error_status_passthrough():
    with patch(
        "httpx.AsyncClient.send",
        new=AsyncMock(side_effect=lambda req, stream=False: _mock_response(429, b'{"error":"rate"}')),
    ):
        r = client.post("/v1/chat/completions", json=CLEAN)
    assert r.status_code == 429


def test_invalid_json_returns_400():
    r = client.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400


def test_unreachable_upstream_returns_502_bridge_error():
    with patch("httpx.AsyncClient.send", new=AsyncMock(side_effect=httpx.ConnectError("refused"))):
        r = client.post("/v1/chat/completions", json=CLEAN)
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "bridge_upstream_error"


# --- 手动 runner（可选，不参与 pytest 收集）--------------------------------

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print("FAIL", name, "->", exc)
            failed += 1
    print(f"\n=== {passed} passed, {failed} failed, {passed + failed} total ===")
    sys.exit(1 if failed else 0)
