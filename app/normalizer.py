"""请求规范化器（Request Normalizer）。

职责：在 dsh（或任何 OpenAI 兼容客户端）发出的 /v1/chat/completions 请求
转发给上游 deeperseeker 之前，把「上游不认 / 会导致 400」的字段与历史结构
洗成上游能吃的形状。这是纯适配层，不改业务语义。

覆盖的 4 类问题（来自实证任务清单，编号沿用原表）：
  #4 reasoning_content 丢失导致 400
  #6 孤儿 tool_call 毒化会话
  #7 strict 字段上游不认
  #8 思维链跨模型不兼容

设计原则：
  - 纯函数、无 IO、无全局状态 —— 可离线单测，可独立开关。
  - 只做「删/补/改名」，不做「猜意图」的重写。
  - 每个动作都记一条 warning，便于回放定位是哪个客户端触发的。
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bridge.normalizer")

# 上游不认、需从请求体里剥掉的顶层/工具级字段
_STRIP_TOP_LEVEL = {"strict"}
# 工具定义里需要剥离的字段（上游 tools[].function 只认 name/description/parameters）
_STRIP_TOOL_FUNCTION = {"strict"}


def normalize_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """规范化一次 chat.completions 请求体。

    返回一个**新的** dict，不修改入参（便于对比测试与失败回滚）。
    """
    out = copy.deepcopy(body)

    _strip_strict(out)
    _repair_orphan_tool_calls(out)
    _reinject_reasoning_content(out)
    _normalize_reasoning_cross_model(out)

    return out


# --------------------------------------------------------------------------
# #7 strict 字段上游不认
# --------------------------------------------------------------------------
def _strip_strict(body: Dict[str, Any]) -> None:
    """剥掉 strict 字段。

    客户端（dsh / OpenAI SDK 新版）默认会带 `strict: true` 或工具级
    `function.strict`，上游 deeperseeker 不认这个键，会直接 400。
    剥掉即可 —— 结构化输出的强约束在上游由 prompt 层兜底，不是硬需求。
    """
    removed = []

    for key in _STRIP_TOP_LEVEL:
        if key in body:
            body.pop(key, None)
            removed.append(key)

    # response_format: {"type":"json_schema","json_schema":{"strict":true,...}}
    rf = body.get("response_format")
    if isinstance(rf, dict):
        js = rf.get("json_schema")
        if isinstance(js, dict) and "strict" in js:
            js.pop("strict", None)
            removed.append("response_format.json_schema.strict")

    # tools[].function.strict
    tools = body.get("tools")
    if isinstance(tools, list):
        for i, tool in enumerate(tools):
            fn = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(fn, dict):
                for key in _STRIP_TOOL_FUNCTION:
                    if key in fn:
                        fn.pop(key, None)
                        removed.append(f"tools[{i}].function.{key}")

    if removed:
        logger.warning("stripped unsupported fields: %s", ", ".join(removed))


# --------------------------------------------------------------------------
# #6 孤儿 tool_call 毒化会话
# --------------------------------------------------------------------------
def _repair_orphan_tool_calls(body: Dict[str, Any]) -> None:
    """修补孤儿 tool_call。

    场景：assistant 消息里声明了 tool_calls（可能多个），但后面的 tool
    结果消息缺失（客户端中断、超时、被裁剪）。上游收到「有 call 无 result」
    的历史会拒绝或行为异常，且这个坏历史一旦进会话就持续毒化后续请求。

    策略：
      1. 逐条扫描，收集 assistant 声明的 tool_call id；
      2. 收集后续 role=tool 的 tool_call_id；
      3. 差集即孤儿 —— 为每个孤儿补一条占位 tool 消息，
         内容写明「结果缺失」，让上游看到配对完整的序列。
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return

    declared: List[str] = []
    answered: set = set()

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            for call in msg.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id"):
                    declared.append(call["id"])
        elif role == "tool":
            cid = msg.get("tool_call_id")
            if cid:
                answered.add(cid)

    orphans = [cid for cid in declared if cid not in answered]
    if not orphans:
        return

    # 在末尾补占位 tool 消息。上游只要求配对，不要求语义正确。
    for cid in orphans:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": cid,
                "content": (
                    "[bridge] tool result missing: the client did not return a "
                    "result for this tool call. Treat as an empty/failed tool "
                    "execution and continue."
                ),
            }
        )
    logger.warning(
        "repaired %d orphan tool_call(s): %s", len(orphans), ", ".join(orphans)
    )


# --------------------------------------------------------------------------
# #4 reasoning_content 丢失导致 400
# --------------------------------------------------------------------------
def _reinject_reasoning_content(body: Dict[str, Any]) -> None:
    """给缺失 reasoning_content 的 assistant 消息补字段。

    场景：上游 deepseek-reasoner 系模型在多轮里要求历史 assistant 消息
    带上 `reasoning_content`；若客户端把上一轮的 reasoning 丢了（很多
    客户端不存这个字段），下一轮请求会 400。

    策略：**不造假思维链**（那会污染模型判断），只补一个空字符串占位，
    让字段存在即可。上游对空串的容忍度高于「字段缺失」。
    注意：只补 assistant 且**确实没有**该键的消息，已有内容的原样保留。
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return

    filled = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        # 只处理带 content 的正式回复，跳过纯 tool_calls 的中转消息
        if "reasoning_content" not in msg:
            msg["reasoning_content"] = ""
            filled += 1

    if filled:
        logger.warning("filled missing reasoning_content on %d assistant msg(s)", filled)


# --------------------------------------------------------------------------
# #8 思维链跨模型不兼容
# --------------------------------------------------------------------------
def _normalize_reasoning_cross_model(body: Dict[str, Any]) -> None:
    """处理跨模型思维链不兼容。

    场景：会话先用 reasoner 模型产生 reasoning_content，中途切到非
    reasoner 模型（如 deepseek-chat），后者不认这个字段，同样 400。

    策略：若本次请求的 model 不在「支持思维链」白名单内，则把所有历史
    assistant 消息的 `reasoning_content` 键**整体移除**，而不是置空 ——
    非 reasoner 模型对「键存在但为空」也可能挑刺。
    """
    model = str(body.get("model") or "")
    if _supports_reasoning(model):
        return

    messages = body.get("messages")
    if not isinstance(messages, list):
        return

    removed = 0
    for msg in messages:
        if isinstance(msg, dict) and "reasoning_content" in msg:
            msg.pop("reasoning_content", None)
            removed += 1

    if removed:
        logger.warning(
            "model %r is not reasoning-capable; removed reasoning_content from %d msg(s)",
            model,
            removed,
        )


def _supports_reasoning(model: str) -> bool:
    """判定模型是否支持/需要 reasoning_content 字段。"""
    m = model.lower()
    return "reasoner" in m or "reasoning" in m or "r1" in m


def summarize_actions(body: Dict[str, Any]) -> List[str]:
    """返回对给定请求体会执行的动作名列表（dry-run，不修改）。

    用途：中间件启动时的自检，或 /bridge/dry-run 端点。
    """
    actions: List[str] = []
    probe = copy.deepcopy(body)

    before = copy.deepcopy(probe)
    _strip_strict(probe)
    if probe != before:
        actions.append("strip_strict")

    before = copy.deepcopy(probe)
    _repair_orphan_tool_calls(probe)
    if probe != before:
        actions.append("repair_orphan_tool_calls")

    before = copy.deepcopy(probe)
    _reinject_reasoning_content(probe)
    if probe != before:
        actions.append("reinject_reasoning_content")

    before = copy.deepcopy(probe)
    _normalize_reasoning_cross_model(probe)
    if probe != before:
        actions.append("normalize_reasoning_cross_model")

    return actions