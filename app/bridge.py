"""接线层：把 normalizer 挂到反向代理前面。

职责边界刻意收窄：本模块只做「收到请求 → 规范化 → 转发给上游 → 回传响应」。
不做业务逻辑、不解析响应（响应侧限制见 known_limitations.md）。

设计：
  - 上游地址由 UPSTREAM_BASE 环境变量指定（默认本机 deeperseeker:4000）
  - 规范化后的请求体转发；上游响应原样回传（含流式）
  - 若上游不可达，返回 502 并带明确错误（不伪装成上游错误）
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.normalizer import normalize_request, summarize_actions

logger = logging.getLogger("bridge")

UPSTREAM_BASE = os.getenv("BRIDGE_UPSTREAM_BASE", "http://127.0.0.1:4000")
UPSTREAM_TIMEOUT = float(os.getenv("BRIDGE_UPSTREAM_TIMEOUT", "300"))

app = FastAPI(title="deeperseeker-agent-bridge", version="0.1.0")


@app.get("/bridge/health")
async def health() -> Dict[str, Any]:
    """健康检查 + 上游可达性探测（不打印任何密钥）。"""
    reachable = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{UPSTREAM_BASE}/")
            reachable = r.status_code < 500
    except Exception as exc:
        logger.warning("upstream health probe failed: %s", exc)
    return {
        "status": "ok",
        "upstream_base": UPSTREAM_BASE,
        "upstream_reachable": reachable,
    }


@app.post("/bridge/dry-run")
async def dry_run(request: Request) -> Dict[str, Any]:
    """只跑规范化，不转发。返回会执行哪些动作。

    用途：客户端在正式转发前自检，或调试某个请求体为何被改。
    """
    body = await request.json()
    actions = summarize_actions(body)
    normalized = normalize_request(body)
    return {
        "actions": actions,
        "changed": normalized != body,
        "normalized": normalized,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """规范化后转发到上游。

    - 请求体先过 normalizer（剥离 strict / 修孤儿 tool_call / reasoning 处理）
    - 流式与非流式都按原样透传（不解析响应，见 known_limitations.md）
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400)

    actions = summarize_actions(body)
    if actions:
        logger.warning("normalizing request: %s", ", ".join(actions))
    normalized = normalize_request(body)

    # 透传客户端认证头；不记录其内容。
    fwd_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() in ("authorization", "content-type", "accept")
    }

    stream = bool(normalized.get("stream"))

    try:
        client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
        upstream_req = client.build_request(
            "POST",
            f"{UPSTREAM_BASE}/v1/chat/completions",
            json=normalized,
            headers=fwd_headers,
        )
        upstream_resp = await client.send(upstream_req, stream=stream)
    except Exception as exc:
        logger.exception("upstream request failed")
        return JSONResponse(
            {"error": {"message": f"upstream unreachable: {exc}", "type": "bridge_upstream_error"}},
            status_code=502,
        )

    if not stream:
        content = await upstream_resp.aread()
        await upstream_resp.aclose()
        await client.aclose()
        return JSONResponse(
            content=_safe_json(content),
            status_code=upstream_resp.status_code,
        )

    async def _relay():
        try:
            async for chunk in upstream_resp.aiter_raw():
                yield chunk
        finally:
            await upstream_resp.aclose()
            await client.aclose()

    return StreamingResponse(
        _relay(),
        status_code=upstream_resp.status_code,
        media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
    )


def _safe_json(raw: bytes):
    """上游非 JSON 时包一层，避免 bridge 自身崩在解析上。"""
    import json

    try:
        return json.loads(raw)
    except Exception:
        return {"error": {"message": raw.decode("utf-8", "replace")[:2000], "type": "bridge_upstream_nonjson"}}
