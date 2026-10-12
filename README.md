# deeperseeker-agent-bridge

dsh（及任何 OpenAI 兼容 agent harness）与
[deeperseeker](https://github.com/AmanCode22/deeperseeker) 上游之间的适配中间件。

**定位**：纯适配层。在请求转发给上游之前，把上游不认的字段与历史结构洗成
它能吃的形状。**不改业务语义，不做推理，不碰上游代码。**

---

## 解决的问题（请求侧）

上游 deeperseeker 对某些 OpenAI 兼容字段与历史结构不容忍，客户端（dsh 等）
直连时会 400 或行为异常。本中间件在转发前统一规范化：

| 问题 | 现象 | 规范化动作 |
|---|---|---|
| `strict` 字段上游不认 | 客户端带 `strict: true` → 400 | 剥除顶层 / `response_format.json_schema` / `tools[].function` 三处的 `strict` |
| 孤儿 `tool_call` 毒化会话（= 工具结果被静默丢弃） | assistant 声明了 `tool_calls` 但结果消息缺失 → 后续请求持续异常 | 为每个孤儿补占位 `role:"tool"` 消息，**协议配对完整** |
| ↳ 其中「真实工具结果丢失」 | 客户端侧就没把结果发出来，bridge 从未收到 | **不恢复、不造假**（补的是"失败占位"，让模型知道该次调用失败后继续）——这是唯一正确处理，非缺口 |
| `reasoning_content` 丢失致 400 | 多轮会话中 assistant 消息缺 `reasoning_content`（客户端不存）→ 400 | 补空串占位（**不造假思维链**） |
| 思维链跨模型不兼容 | 会话中途从 reasoner 切到非 reasoner 模型 → 400 | 非 reasoner 模型下整体移除 `reasoning_content` 键 |

设计约束：**纯函数、无 IO、不修改入参（deepcopy 隔离）、幂等**。
每个动作都记 warning 日志，便于线上回放定位。

---

## 状态声明（Status）

**本中间件当前为「契约级完成」，非「链路级完成」。**

- **契约级完成**：4 类请求侧问题的规范化逻辑已实现，输入输出形状经单元测试确认。
- **链路级完成 = 未达成**：未经真实 DeepSeek 上游端到端验证，也未经真实 DSH
  实际请求形态验证。见下方「已验证范围」与 `known_limitations.md` 的 LIM-3。

> 换句话说：**"按契约修好了"不等于"真实链路上验证过"。** 接入真实上游前，
> 请把本仓库视为"逻辑已就绪、链路未证实"的中间件。

---

## 已验证范围（**请仔细读**）

本中间件的验证**仅覆盖请求规范化契约**：

- ✅ **单元测试**：`tests/test_normalizer.py`，15 个用例全绿。
  覆盖 4 类问题的正例与反例、组合场景、幂等性、空输入健壮性。
- ✅ **契约级验证**：请求进出 normalizer 的**输入输出形状**已确认。

**明确未验证**（勿假设已完成）：

- ❌ **未与真实 DeepSeek 上游做端到端功能验证** —— 需要真实账号 token，
  本次工作范围明确不涉及。
- ❌ **未验证真实链路上的响应行为** —— 本中间件只处理请求侧；
  响应侧的已知限制见 `known_limitations.md`。
- ⏳ **docker-compose 接线验证**：仅验证中间件能作为反向代理前置启动、
  正确转发请求，**不等于功能验证**。

如需完整功能验证，需接入真实上游并复测。

---

## 已知限制

上游 deeperseeker 有两个**架构级限制**，不在本中间件修复范围内。
**接入前必读** [`known_limitations.md`](./known_limitations.md)：

- **LIM-1**：工具调用在流式期间完全不可见（正文/工具时序错位）——对做流式
  UI 或流式决策的 harness 是**语义损失**，不只是延迟。
- **LIM-2**：工具调用的 SSE 增量退化为"一次性完整帧"——协议合规，
  但非标准增量语义。

`known_limitations.md` 末尾另附：原始任务清单中"#5 流式 `name: null`
覆盖工具名"经实证核对为**误报**的完整推理链。

---

## 运行测试

```bash
# 有 pytest 环境：
python3 -m pytest tests/ -q

# 无 pytest（标准库 runner）：
python3 -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('t', 'tests/test_normalizer.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fns = [(n, getattr(m, n)) for n in dir(m) if n.startswith('test_')]
p = f = 0
for n, fn in fns:
    try: fn(); p += 1; print('PASS', n)
    except Exception as e: f += 1; print('FAIL', n, e)
print(f'{p} passed, {f} failed')
"
```

---

## 结构

```
app/normalizer.py      请求规范化器（纯函数，无 IO）
tests/test_normalizer.py  15 个单元测试
known_limitations.md   上游架构级限制 + #5 误报推理链
```

---

## 与上游 fork 的分工

- **本中间件**（请求侧适配）：`strict` / 孤儿 tool_call / `reasoning_content`
  / 跨模型思维链 —— 这些是"请求进来先洗一遍"，放中间件比塞进上游干净。
- **上游 fork 内**（改动位于另一个仓库：deeperseeker 的上游 fork，不在本仓库内）：
  L1 rollover 阈值、L3 空响应自愈 —— 这些涉及上游会话生命周期，必须在内部修。

两者**不重叠**：中间件管"进来的请求长什么样"，fork 管"会话怎么活"。
