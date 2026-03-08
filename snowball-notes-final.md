# Snowball Notes — 最终技术方案

> 合并 v7 和 v8，补齐所有实现层面的设计漏洞。
> 这份文档可以直接作为工程蓝图：每一个被调用的函数都有定义，每一个数据结构都有 schema。
>
> 核心立场：**Agent 可以自主决策，但不能无约束地产生副作用。**
> 推理阶段只产出意图（Proposal），提交阶段由 Committer 统一落地副作用。
> 每次运行都有完整 Trace 和 ReplayBundle，可以被离线复现和调试。

---

## 一、项目定位

### 1.1 产品目标

- 从 Codex transcript 中异步捕获已完成 turn
- 由 Agent 独立决定：跳过 / 归档 / 创建新知识笔记 / 追加旧笔记 / 标记人工审核
- 所有结果落入 Obsidian Vault，支持个人长期使用
- 所有决策具备 Trace、Metrics、Eval、Replay、Review 能力

### 1.2 求职目标

这个项目要能体现：

- 你理解 Agent 与 workflow 的本质差异
- 你能控制 Agent 的边界，不依赖 LLM 自觉遵守约束
- 你知道如何做 Agent Observability、Evaluation、Replay、Safety Guardrails
- 你能把一个 LLM demo 做成接近生产系统的工程化原型
- 你的后端工程背景体现在：状态机、事务、副作用控制，而不只是会调 API

### 1.3 非目标（MVP 阶段）

- 不追求支持所有笔记软件（只做 Obsidian）
- 不追求多 Agent 协作
- 不追求开放域联网研究
- 不追求全自动覆盖和重写既有笔记

---

## 二、为什么必须是 Agent

### 2.1 这个问题为什么不适合 workflow

普通 workflow 的强项是路径稳定、分支有限、决策边界明确。
Snowball 的难点在于：

1. 同一个 turn，可能既包含新知识，又与旧笔记局部重合
2. 只有在读过候选旧笔记后，才能判断是 append 还是 create
3. 风险不只是"答错"，而是"把知识库写脏"
4. 最佳动作往往不是"继续执行"，而是"停止并升级人工审核"

> **如何让系统基于中间观察动态决定下一步动作，并在不确定时主动收缩。**
> 这是 Agent 问题，不是 workflow 问题。

### 2.2 Agent 的四个判断标准

只有满足这四点才算 Agent：

1. 模型负责决定下一步调用哪个 Tool
2. 下一步依赖前一步 Observation
3. 系统允许中途停止、转人工、或改变计划
4. 决策质量有 Trace 和 Eval 支撑

---

## 三、系统架构

```text
┌──────────────────────────────────────────────────────────┐
│                    Intake Layer                          │
│   transcript_poll / transcript_watch / parser            │
│   compute_source_confidence()                            │
│   → StandardEvent                                        │
└──────────────────────────┬───────────────────────────────┘
                           │ StandardEvent（含 source_confidence）
                           ▼
┌──────────────────────────────────────────────────────────┐
│                   Event Queue（SQLite）                   │
│         turn_id UNIQUE / claim / retry / RunState        │
└──────────────────────────┬───────────────────────────────┘
                           │ RECEIVED → PREPARED
                           ▼
┌──────────────────────────────────────────────────────────┐
│                  Agent Runtime                           │
│                                                          │
│  RunState Machine                                        │
│    PREPARED → RUNNING → PROPOSED_ACTIONS                 │
│    → COMMITTING → COMPLETED | FLAGGED | FAILED           │
│                                                          │
│  ReAct Loop（RUNNING 阶段）                               │
│    Decision Tools: assess / extract / search / read      │
│    Action Tools:   propose_* / flag_for_review           │
│    Proposals 收集到 ActionProposal list                   │
│                                                          │
│  Committer（COMMITTING 阶段）                             │
│    校验 proposals → 原子写入 Vault + SQLite               │
│                                                          │
│  AgentState + Guardrails（贯穿全程）                      │
└──────────────────────────┬───────────────────────────────┘
                           │ 原子提交
                           ▼
┌──────────────────────────────────────────────────────────┐
│                 SQLite + Obsidian Vault                  │
│   action_proposals / agent_traces / replay_bundles       │
│   Inbox / Archive / Knowledge                            │
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│         Review UI + Trace Replay + Eval Runner           │
└──────────────────────────────────────────────────────────┘
```

---

## 四、输入与触发

### 4.1 触发源

主触发源：`~/.codex/sessions/**/*.jsonl`（不使用 `history.jsonl`）

完成信号：

```json
{
  "type": "event_msg",
  "payload": { "type": "task_complete", "turn_id": "..." }
}
```

### 4.2 标准输入事件（StandardEvent）

```json
{
  "event_id": "evt_20260307_xxx",
  "session_file": "/Users/7fish/.codex/sessions/2026/03/07/rollout-....jsonl",
  "conversation_id": "019cc689-fdaf-7ad0-bac6-e8d01676b7d2",
  "turn_id": "019cc702-ac95-79d2-838b-3a7dfb67035b",
  "user_message": "可以通过 ~/.codex/ 的 transcript 文件来做事件触发吗",
  "assistant_final_answer": "可以，而且从你本机现状看......",
  "displayed_at": "2026-03-07T06:36:45Z",
  "source_completeness": "full",
  "source_confidence": 0.92,
  "parser_version": "v1",
  "context_meta": { "client": "codex", "cwd": "/Users/7fish/project" }
}
```

### 4.3 source_confidence 计算规则

`source_confidence` 在 Intake 层计算，随 StandardEvent 携带，下游不重算。
它直接决定 Agent 能做什么，是最关键的上游输入，不能是黑盒。

```python
CURRENT_STABLE_PARSER = "v1"

def compute_source_confidence(
    turn_events: list[dict],
    final_answer: str | None,
    user_message: str | None,
    source_completeness: str,
    parser_version: str,
) -> float:
    score = 1.0

    if not final_answer:
        score -= 0.50   # 核心字段缺失，最重
    if not user_message:
        score -= 0.20

    if source_completeness == "partial":
        score -= 0.20   # 来源完整性不足

    if parser_version != CURRENT_STABLE_PARSER:
        score -= 0.10   # 兼容模式解析，可信度降低

    if final_answer and len(final_answer.strip()) < 50:
        score -= 0.15   # 极短 answer 通常是截断或中间态

    task_complete_count = sum(
        1 for e in turn_events
        if e.get("payload", {}).get("type") == "task_complete"
    )
    if task_complete_count > 1:
        score -= 0.20   # 多个 task_complete 通常是 session 文件损坏

    return round(max(0.0, min(1.0, score)), 2)
```

这是启发式规则，不追求精确概率。后续可通过 `confidence_feedback` 表的人工标注迭代调整。

### 4.4 Intake 原则

- `turn_id` 是主幂等键
- `source_confidence < 0.5` 的 turn 在入队前过滤，不启动 Agent（节省成本）
- `len(assistant_final_answer) < 120` 的 turn 直接跳过（闲聊/调试片段）
- Parser 只负责"重建事实"，不做价值判断

---

## 五、运行状态机

这是 Agent Runtime 的生命周期骨架，让每个问题都有明确的归属状态。

### 5.1 RunState 定义

```python
class RunState(str, Enum):
    RECEIVED          = "received"           # 已入队，尚未开始
    PREPARED          = "prepared"           # 输入冻结，ReplayBundle 初始化
    RUNNING           = "running"            # Agent 正在 ReAct Loop 中
    PROPOSED_ACTIONS  = "proposed_actions"   # Agent 完成推理，proposals 已收集
    COMMITTING        = "committing"         # Committer 正在原子提交
    COMPLETED         = "completed"          # 成功结束
    FLAGGED           = "flagged"            # 升级人工审核
    FAILED_RETRYABLE  = "failed_retryable"   # 可重试失败（API 限流等）
    FAILED_FATAL      = "failed_fatal"       # 致命失败（数据损坏等）
```

### 5.2 状态迁移与触发时机

```text
RECEIVED ──────────────────────────────► PREPARED
  触发：Worker claim task 成功，输入冻结，ReplayBundle 初始化

PREPARED ──────────────────────────────► RUNNING
  触发：Agent.__init__ 完成，开始第一次 model.respond()

RUNNING ────────────────────────────────► PROPOSED_ACTIONS
  触发：ReAct Loop 正常结束（end_turn 或所有步骤完成）

RUNNING ────────────────────────────────► FLAGGED
  触发：flag_for_review Tool 调用，state.is_terminated = True

RUNNING ────────────────────────────────► FAILED_RETRYABLE
  触发：model API 调用失败超过重试次数

RUNNING ────────────────────────────────► FAILED_FATAL
  触发：未知异常 / 数据损坏

PROPOSED_ACTIONS ───────────────────────► COMMITTING
  触发：Committer 校验全部通过

PROPOSED_ACTIONS ───────────────────────► FLAGGED
  触发：Committer 校验失败（违反 guardrails）

COMMITTING ─────────────────────────────► COMPLETED
  触发：Vault + SQLite 原子写入成功

COMMITTING ─────────────────────────────► FAILED_RETRYABLE
  触发：Vault 写入 I/O 失败（磁盘满等）
```

### 5.3 状态机实现

状态存入 `tasks.status` 字段，所有迁移通过 `transition_state()` 统一执行：

```python
VALID_TRANSITIONS = {
    RunState.RECEIVED:         {RunState.PREPARED},
    RunState.PREPARED:         {RunState.RUNNING},
    RunState.RUNNING:          {RunState.PROPOSED_ACTIONS, RunState.FLAGGED,
                                RunState.FAILED_RETRYABLE, RunState.FAILED_FATAL},
    RunState.PROPOSED_ACTIONS: {RunState.COMMITTING, RunState.FLAGGED},
    RunState.COMMITTING:       {RunState.COMPLETED, RunState.FAILED_RETRYABLE},
    RunState.COMPLETED:        set(),
    RunState.FLAGGED:          set(),
    RunState.FAILED_RETRYABLE: {RunState.RECEIVED},  # 重入队
    RunState.FAILED_FATAL:     set(),
}

def transition_state(
    db: Database,
    task_id: str,
    current: RunState,
    target: RunState,
    reason: str = "",
) -> None:
    if target not in VALID_TRANSITIONS[current]:
        raise InvalidStateTransition(f"{current} → {target} 不合法")

    updated = db.execute("""
        UPDATE tasks
        SET status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE task_id = ? AND status = ?
    """, (target.value, task_id, current.value))

    if updated.rowcount != 1:
        raise StateTransitionConflict(f"task {task_id} 状态已被其他 worker 修改")

    write_audit_log(db, "state_transition", {
        "task_id": task_id, "from": current, "to": target, "reason": reason
    })
```

**关键设计**：迁移必须检查 `rowcount == 1`，防止多 Worker 并发时产生竞态。

---

## 六、AgentState

AgentState 是 Guardrails 的执行载体，贯穿整个 ReAct Loop。

```python
@dataclass
class AgentState:
    # 不可变输入
    event: StandardEvent
    task_id: str
    trace_id: str

    # Session 上下文（运行前加载）
    session_memory: SessionMemory

    # 本次 run 的写入计数（Guardrails 依赖）
    write_count: int = 0
    append_count: int = 0

    # 终止标志（flag_for_review 后设为 True）
    is_terminated: bool = False
    terminal_reason: str = ""

    # 本次 run 收集的 proposals（推理阶段只增不删）
    proposals: list[ActionProposal] = field(default_factory=list)

    # Replay 数据收集（运行时填充）
    tool_results_for_replay: list[dict] = field(default_factory=list)
    knowledge_snapshot_refs: list[dict] = field(default_factory=list)
```

---

## 七、ReAct Loop 与 Proposal 收集

### 7.1 完整 Runtime 实现

```python
class SnowballAgent:
    def __init__(self, model_adapter, tools, max_steps=8):
        self.model_adapter = model_adapter
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def run(
        self,
        event: StandardEvent,
        task_id: str,
        trace: AgentTrace,
        db: Database,
    ) -> AgentResult:

        # 加载 Session Memory，注入 initial messages
        session_memory = load_session_memory(db, event.conversation_id)
        state = AgentState(
            event=event,
            task_id=task_id,
            trace_id=trace.trace_id,
            session_memory=session_memory,
        )

        # PREPARED → RUNNING
        transition_state(db, task_id, RunState.PREPARED, RunState.RUNNING)
        messages = build_initial_messages(event, session_memory)

        # ── ReAct Loop（RUNNING 阶段）──────────────────────────────
        for step_index in range(self.max_steps):
            response = self._call_model(messages, trace, step_index)

            # Agent 正常结束（end_turn）
            if response.stop_reason == "end_turn":
                break

            # Agent 发起 Tool 调用
            if response.stop_reason == "tool_use":
                observations = []

                for tool_call in response.tool_use_blocks:

                    # Guardrails 前置检查（在执行前拦截）
                    guardrail = check_guardrail(state, tool_call.name)
                    if not guardrail.allowed:
                        obs = ToolResult.blocked(tool_call.id, guardrail.reason)
                        trace.record_guardrail_block(step_index, tool_call.name, guardrail.reason)
                        observations.append(obs)
                        continue

                    # 执行 Tool
                    obs = self._execute_tool(tool_call, state, trace, step_index)
                    observations.append(obs)

                    # Replay 数据收集：每次 Tool 执行后立即记录
                    state.tool_results_for_replay.append({
                        "step": step_index,
                        "tool": tool_call.name,
                        "input": tool_call.input,
                        "output": obs.data,
                        "success": obs.success,
                    })

                    # 终止检测：flag_for_review 后立即退出
                    if state.is_terminated:
                        messages = advance_messages(messages, response, observations)
                        trace.record_step(step_index, response, observations,
                                          runtime_state=RunState.RUNNING)
                        self._finalize(state, db, trace)
                        transition_state(db, task_id, RunState.RUNNING,
                                         RunState.FLAGGED, reason=state.terminal_reason)
                        return AgentResult.flagged(state.terminal_reason, trace)

                messages = advance_messages(messages, response, observations)
                trace.record_step(step_index, response, observations,
                                  runtime_state=RunState.RUNNING)
                continue

            # 未知 stop_reason，降级
            self._finalize(state, db, trace)
            transition_state(db, task_id, RunState.RUNNING, RunState.FLAGGED,
                              reason="unexpected_model_response")
            return AgentResult.flagged("unexpected_model_response", trace)

        # ── 退出 Loop，进入 PROPOSED_ACTIONS ─────────────────────
        transition_state(db, task_id, RunState.RUNNING, RunState.PROPOSED_ACTIONS)
        trace.set_runtime_state(RunState.PROPOSED_ACTIONS)

        # ── Committer 校验并提交 ──────────────────────────────────
        committer = Committer(db=db, vault=get_vault(), state=state)
        commit_result = committer.run(task_id, trace)

        if commit_result.success:
            self._flush_session_memory(state, db)
            self._save_replay_bundle(state, trace, db)
            transition_state(db, task_id, RunState.COMMITTING, RunState.COMPLETED)
            return AgentResult.completed(commit_result, trace)
        else:
            self._save_replay_bundle(state, trace, db)
            transition_state(db, task_id, RunState.PROPOSED_ACTIONS,
                              RunState.FLAGGED, reason=commit_result.rejection_reason)
            return AgentResult.flagged(commit_result.rejection_reason, trace)

    def _execute_tool(self, tool_call, state, trace, step_index) -> ToolResult:
        tool = self.tools.get(tool_call.name)
        if not tool:
            return ToolResult.error(tool_call.id, "unknown_tool", tool_call.name)
        try:
            result = tool.execute(tool_call.input, state)
            # 更新写入计数
            if tool_call.name in ACTION_TOOLS and result.success:
                state.write_count += 1
            if tool_call.name == "propose_append_to_note" and result.success:
                state.append_count += 1
            return result
        except Exception as e:
            write_audit_log(db=None, event_type="tool_execution_error",
                            detail={"tool": tool_call.name, "error": str(e)})
            return ToolResult.error(tool_call.id, "execution_error", str(e))

    def _call_model(self, messages, trace, step_index):
        """带重试的 model 调用，API 失败时指数退避"""
        for attempt in range(3):
            try:
                response = self.model_adapter.respond(
                    system_prompt=load_prompt("agent_system/current"),
                    messages=messages,
                    tools=[t.schema for t in self.tools.values()],
                )
                trace.record_token_usage(step_index, response.usage)
                return response
            except APIRateLimitError:
                time.sleep(2 ** attempt)
        raise RetryExhaustedError("model API 调用失败超过重试次数")

    def _finalize(self, state, db, trace):
        """运行结束前统一处理，不管成功还是失败"""
        self._flush_session_memory(state, db)
        self._save_replay_bundle(state, trace, db)

    def _flush_session_memory(self, state, db):
        update_session_memory(
            db=db,
            conversation_id=state.event.conversation_id,
            turn_id=state.event.turn_id,
            created_note_ids=[p.target_note_id for p in state.proposals
                               if p.action_type == "create_note"],
            appended_note_ids=[p.target_note_id for p in state.proposals
                                if p.action_type == "append_note"],
        )

    def _save_replay_bundle(self, state, trace, db):
        bundle = ReplayBundle(
            trace_id=trace.trace_id,
            event_json=json.dumps(dataclasses.asdict(state.event)),
            prompt_snapshot=load_prompt("agent_system/current"),
            config_snapshot_json=json.dumps(load_config()),
            tool_results_json=json.dumps(state.tool_results_for_replay),
            knowledge_snapshot_refs_json=json.dumps(state.knowledge_snapshot_refs),
            model_name=self.model_adapter.model_name,
            model_adapter_version=self.model_adapter.version,
        )
        save_replay_bundle(db, bundle)
```

### 7.2 Session Memory 注入方式

选择**注入 initial user message**，而不是独立 Tool——Session Memory 是 Agent 的前提上下文，不能让 Agent "忘记查询"。

```python
def build_initial_messages(event: StandardEvent, memory: SessionMemory) -> list[dict]:
    session_ctx = ""
    if memory.processed_turns:
        recent = memory.processed_turns[-5:]  # 只注入最近 5 条，控制 token
        session_ctx = f"""
## 本 Session 上下文

本次对话中已处理 {len(memory.processed_turns)} 条 turn，最近操作：
{_format_recent_actions(recent, memory)}

注意：不要对同一笔记重复 create 或 append。
"""
    content = f"""{session_ctx}
## 当前 Turn

用户问题：{event.user_message}

助手回答：
{event.assistant_final_answer}

元信息：
- turn_id: {event.turn_id}
- source_confidence: {event.source_confidence}
- source_completeness: {event.source_completeness}
- 项目目录: {event.context_meta.get('cwd', '未知')}

请开始处理。"""
    return [{"role": "user", "content": content.strip()}]
```

---

## 八、Tool 分层设计

### 8.1 Decision Tools（不产生副作用）

| Tool | 作用 |
|------|------|
| `assess_turn_value` | 判断 skip/archive/note 可能性 |
| `extract_knowledge_points` | 提取候选命题 |
| `search_similar_notes` | 三层检索（标题/标签/embedding） |
| `read_note` | 读取笔记正文 |

`search_similar_notes` 执行时同步写入 `state.knowledge_snapshot_refs`：

```python
class SearchSimilarNotesTool:
    def execute(self, input: dict, state: AgentState) -> ToolResult:
        results = knowledge_index.search(input["query"], input.get("top_k", 5))

        # 写入 knowledge snapshot，用于 ReplayBundle
        for r in results:
            state.knowledge_snapshot_refs.append({
                "note_id": r.note_id,
                "content_hash": r.content_hash,  # 记录命中时的版本
                "title": r.title,
                "similarity": r.similarity,
            })

        return ToolResult.success(data=[r.to_dict() for r in results])
```

### 8.2 Action Tools（生成 Proposal，不直接写入）

| Tool | 作用 |
|------|------|
| `propose_create_note` | 提议新建笔记 |
| `propose_append_to_note` | 提议追加到 `## Updates` |
| `propose_archive_turn` | 提议写 Archive note |
| `propose_link_notes` | 提议建立笔记双向链接 |
| `flag_for_review` | 终止型动作，设置 `is_terminated`（不进入 proposal batch） |

所有 Action Tool 执行后只生成 `ActionProposal`，挂到 `state.proposals`：

```python
class ProposeCreateNoteTool:
    name = "propose_create_note"

    def execute(self, input: dict, state: AgentState) -> ToolResult:
        # Schema 校验（由 validated_tool_execute 在外层完成）
        proposal = ActionProposal(
            proposal_id=new_id(),
            trace_id=state.trace_id,
            turn_id=state.event.turn_id,
            action_type="create_note",
            target_note_id=None,  # 新建，还没有 note_id
            payload_json=json.dumps({
                "title": input["title"],
                "content": input["content"],
                "tags": input.get("tags", []),
                "source_turn_id": state.event.turn_id,
            }),
            idempotency_key=f"create:{state.event.turn_id}:{input['title']}",
        )
        state.proposals.append(proposal)
        state.write_count += 1

        return ToolResult.success(data={
            "proposal_id": proposal.proposal_id,
            "status": "proposed",
            "message": f"已提议创建笔记：{input['title']}，将在推理结束后提交",
        })
```

### 8.3 flag_for_review 的终止合约

```python
class FlagForReviewTool:
    name = "flag_for_review"

    def execute(self, input: dict, state: AgentState) -> ToolResult:
        reason = input["reason"]

        # 写入 DB（这是唯一一个在推理阶段就直接写 DB 的动作）
        db.execute("""
            INSERT INTO review_actions (review_id, turn_id, trace_id, final_action, reason)
            VALUES (?, ?, ?, 'pending_review', ?)
        """, (new_id(), state.event.turn_id, state.trace_id, reason))

        # 设置终止标志 —— ReAct Loop 检测到后立即 break
        state.is_terminated = True
        state.terminal_reason = reason

        # 清空未提交的 proposals（终止意味着放弃所有推理阶段的写入意图）
        state.proposals.clear()

        return ToolResult.success(
            data={"flagged": True, "reason": reason},
            metadata={
                "flag_reason": reason,
                "conflict_note_id": input.get("conflict_note_id"),
                "suggested_action": input.get("suggested_action"),
            }
        )
```

### 8.4 Tool Schema 校验

所有 Tool 输入在执行前经过 JSON Schema 校验：

```python
def validated_tool_execute(
    tool_name: str, payload: dict, state: AgentState
) -> ToolResult:
    schema = TOOL_SCHEMAS[tool_name]
    errors = jsonschema_validate(schema, payload)
    if errors:
        return ToolResult.validation_error(
            error_code="invalid_tool_input",
            error_message="; ".join(errors),
        )
    return TOOLS[tool_name].execute(payload, state)
```

---

## 九、Committer：两阶段提交

这是整个 Runtime 最重要的安全边界。

### 9.1 Committer 类定义

```python
class Committer:
    def __init__(self, db: Database, vault: Vault, state: AgentState):
        self.db = db
        self.vault = vault
        self.state = state

    def run(self, task_id: str, trace: AgentTrace) -> CommitResult:
        """
        完整的两阶段提交流程：
        1. 校验所有 proposals
        2. 通过则原子提交 Vault + SQLite
        3. 失败则写 audit log，不产生任何副作用
        """
        transition_state(self.db, task_id,
                          RunState.PROPOSED_ACTIONS, RunState.COMMITTING)

        # ── Phase 1：校验 ────────────────────────────────────────
        validation_errors = self._validate_proposals()
        if validation_errors:
            write_audit_log(self.db, "commit_blocked", {
                "task_id": task_id,
                "errors": validation_errors,
            })
            return CommitResult.rejected(
                reason="; ".join(validation_errors)
            )

        # ── Phase 2：原子提交 ─────────────────────────────────────
        try:
            with self.db.transaction():
                committed_note_ids = []
                for proposal in self.state.proposals:
                    note_id = self._commit_proposal(proposal)
                    committed_note_ids.append(note_id)
                    self.db.execute("""
                        UPDATE action_proposals
                        SET status = 'committed', committed_at = CURRENT_TIMESTAMP
                        WHERE proposal_id = ?
                    """, (proposal.proposal_id,))

            return CommitResult.success(committed_note_ids=committed_note_ids)

        except VaultWriteError as e:
            write_audit_log(self.db, "commit_vault_error", {"error": str(e)})
            return CommitResult.retryable(reason=str(e))
        except Exception as e:
            write_audit_log(self.db, "commit_fatal_error", {"error": str(e)})
            return CommitResult.fatal(reason=str(e))

    def _validate_proposals(self) -> list[str]:
        errors = []
        event = self.state.event

        # 约束 1：proposal 数量上限
        write_proposals = [p for p in self.state.proposals
                            if p.action_type != "archive_turn"]
        if len(write_proposals) > MAX_WRITES_PER_RUN:
            errors.append(f"proposals 数量 {len(write_proposals)} 超过上限 {MAX_WRITES_PER_RUN}")

        # 约束 2：低 confidence 不允许创建知识笔记
        if event.source_confidence < MIN_CONFIDENCE_FOR_NOTE:
            create_proposals = [p for p in self.state.proposals
                                  if p.action_type == "create_note"]
            if create_proposals:
                errors.append(
                    f"source_confidence={event.source_confidence} < {MIN_CONFIDENCE_FOR_NOTE}，"
                    f"不允许创建知识笔记"
                )

        # 约束 3：append 目标笔记必须仍然存在
        for p in self.state.proposals:
            if p.action_type == "append_note" and p.target_note_id:
                if not self.vault.exists(p.target_note_id):
                    errors.append(f"append 目标笔记 {p.target_note_id} 不存在")

        # 约束 4：目标笔记的 content_hash 未被本次 run 中其他 proposal 修改
        touched_notes = set()
        for p in self.state.proposals:
            if p.target_note_id:
                if p.target_note_id in touched_notes:
                    errors.append(f"笔记 {p.target_note_id} 在本次 run 中被多个 proposal 操作")
                touched_notes.add(p.target_note_id)

        return errors

    def _commit_proposal(self, proposal: ActionProposal) -> str:
        """提交单个 proposal，返回 note_id"""
        payload = json.loads(proposal.payload_json)

        if proposal.action_type == "create_note":
            note_id = new_id()
            note_path = self.vault.write_new_note(
                note_id=note_id,
                title=payload["title"],
                content=payload["content"],
                tags=payload.get("tags", []),
            )
            self.db.execute("""
                INSERT INTO notes (note_id, note_type, title, vault_path,
                                   content_hash, status, created_at, updated_at)
                VALUES (?, 'atomic', ?, ?, ?, 'inbox', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (note_id, payload["title"], str(note_path),
                   sha256(payload["content"])))
            self.db.execute("""
                INSERT INTO note_sources (note_id, event_id, relation_type)
                VALUES (?, ?, 'derived_from')
            """, (note_id, self.state.event.event_id))
            return note_id

        elif proposal.action_type == "append_note":
            self.vault.append_to_updates_section(
                note_id=proposal.target_note_id,
                content=payload["content"],
                turn_id=payload["source_turn_id"],
            )
            self.db.execute("""
                UPDATE notes SET content_hash = ?, updated_at = CURRENT_TIMESTAMP
                WHERE note_id = ?
            """, (self.vault.get_content_hash(proposal.target_note_id),
                   proposal.target_note_id))
            return proposal.target_note_id

        elif proposal.action_type == "archive_turn":
            note_id = new_id()
            note_path = self.vault.write_archive_note(note_id, payload)
            self.db.execute("""
                INSERT INTO notes (note_id, note_type, title, vault_path,
                                   content_hash, status, created_at, updated_at)
                VALUES (?, 'archive', ?, ?, ?, 'archived', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (note_id, payload["title"], str(note_path),
                   sha256(payload["content"])))
            return note_id

        elif proposal.action_type == "link_notes":
            self.vault.add_bidirectional_link(
                payload["source_note_id"], payload["target_note_id"]
            )
            return payload["source_note_id"]

        else:
            raise ValueError(f"未知 action_type: {proposal.action_type}")
```

---

## 十、Guardrails

Guardrails 在 Tool 执行前拦截，不依赖 LLM 自觉遵守。

```python
ACTION_TOOLS = {
    "propose_create_note", "propose_append_to_note",
    "propose_archive_turn", "propose_link_notes"
}
NOTE_CREATION_TOOLS = {"propose_create_note"}

def check_guardrail(state: AgentState, tool_name: str) -> GuardrailResult:
    event = state.event

    if tool_name in ACTION_TOOLS:
        # 写入次数上限
        if state.write_count >= MAX_WRITES_PER_RUN:
            return GuardrailResult.blocked(
                f"已达到单次 run 写入上限（{MAX_WRITES_PER_RUN}）"
            )

    if tool_name in NOTE_CREATION_TOOLS:
        # confidence 不足禁止创建知识笔记
        if event.source_confidence < MIN_CONFIDENCE_FOR_NOTE:
            return GuardrailResult.blocked(
                f"source_confidence={event.source_confidence} < {MIN_CONFIDENCE_FOR_NOTE}"
            )

    if tool_name == "propose_append_to_note":
        if state.append_count >= MAX_APPENDS_PER_RUN:
            return GuardrailResult.blocked(
                f"已达到单次 run append 上限（{MAX_APPENDS_PER_RUN}）"
            )

    return GuardrailResult.allowed()
```

---

## 十一、Memory 设计

### 11.1 Session Memory（关系型存储）

防止当前 conversation 内的重复动作。

```sql
CREATE TABLE session_turns (
  conversation_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  processed_at TEXT NOT NULL,
  final_decision TEXT NOT NULL,
  PRIMARY KEY (conversation_id, turn_id)
);

CREATE TABLE session_note_actions (
  conversation_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  note_id TEXT NOT NULL,
  action_type TEXT NOT NULL,   -- 'created' | 'appended' | 'linked'
  note_title TEXT,
  created_at TEXT NOT NULL,
  PRIMARY KEY (conversation_id, turn_id, note_id, action_type)
);
```

加载方式（注入前最多取最近 5 条，控制 token）：

```python
def load_session_memory(db: Database, conversation_id: str) -> SessionMemory:
    rows = db.fetchall("""
        SELECT t.turn_id, t.processed_at, t.final_decision,
               a.note_id, a.action_type, a.note_title
        FROM session_turns t
        LEFT JOIN session_note_actions a USING (conversation_id, turn_id)
        WHERE t.conversation_id = ?
        ORDER BY t.processed_at DESC
        LIMIT 20
    """, (conversation_id,))
    return SessionMemory.from_rows(rows)
```

### 11.2 Long-term Memory（知识库索引）

```python
class KnowledgeIndex(Protocol):
    def search_by_title(self, title: str) -> list[NoteMatch]: ...
    def search_by_metadata(self, tags: list[str], topics: list[str]) -> list[NoteMatch]: ...
    def search_by_embedding(self, text: str, top_k: int) -> list[NoteMatch]: ...
    def load_note(self, note_id: str) -> NoteRecord: ...
    def upsert_embedding(self, note_id: str, text: str) -> None: ...
```

---

## 十二、Embedding 设计

### 12.1 Provider 策略

| Provider | 模型 | 向量维度 | 场景 |
|----------|------|---------|------|
| `voyage` | `voyage-3` | 1024 | 默认，效果最好，需要 API Key |
| `local` | `all-MiniLM-L6-v2` | 384 | 离线，无需 API Key，隐私友好 |

```python
class EmbeddingProvider(Protocol):
    model_name: str
    def embed(self, text: str) -> np.ndarray: ...
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...
```

### 12.2 VectorStore（接口抽象，便于替换）

```python
class VectorStore(Protocol):
    def upsert(self, note_id: str, text: str, vector: np.ndarray) -> None: ...
    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]: ...
    def delete(self, note_id: str) -> None: ...

class SQLiteBlobVectorStore(VectorStore):
    """当前实现：全量余弦计算，适合 < 2000 条"""

class SqliteVecStore(VectorStore):
    """升级路径：sqlite-vec 插件，支持 ANN 索引"""
```

---

## 十三、Observability 设计

### 13.1 AgentTrace（记录 decision log，不记录 CoT）

```python
@dataclass
class AgentTrace:
    trace_id: str
    event_id: str
    turn_id: str
    prompt_version: str
    model_name: str
    started_at: datetime
    finished_at: datetime
    total_steps: int
    exceeded_max_steps: bool
    terminal_reason: str       # completed/flagged/exceeded_max_steps/retryable/fatal
    final_decision: str
    final_confidence: float | None
    total_input_tokens: int
    total_output_tokens: int
    total_duration_ms: int
    steps: list[TraceStep]

@dataclass
class TraceStep:
    step_index: int
    runtime_state: str        # running / proposed_actions / committing
    decision_summary: str     # 从 response text 提取的摘要（不是完整 CoT）
    tool_name: str | None
    tool_input_json: str | None
    tool_result_json: str | None
    tool_success: bool | None
    proposal_ids: list[str]   # 本步骤产生的 proposal id 列表
    guardrail_blocked: bool
    duration_ms: int
    input_tokens: int
    output_tokens: int
```

### 13.2 结构化日志（JSON Lines）

```json
{"ts":"2026-03-07T10:23:45Z","level":"INFO","event":"state.transition","task_id":"t_001","from":"prepared","to":"running"}
{"ts":"2026-03-07T10:23:46Z","level":"INFO","event":"tool.execute","trace_id":"tr_001","step":2,"tool":"search_similar_notes","duration_ms":45,"results":3}
{"ts":"2026-03-07T10:23:47Z","level":"INFO","event":"proposal.created","trace_id":"tr_001","action_type":"create_note","proposal_id":"p_001"}
{"ts":"2026-03-07T10:23:48Z","level":"WARN","event":"guardrail.blocked","trace_id":"tr_001","tool":"propose_create_note","reason":"write_count_exceeded"}
{"ts":"2026-03-07T10:23:49Z","level":"INFO","event":"commit.success","trace_id":"tr_001","committed_count":1}
{"ts":"2026-03-07T10:23:49Z","level":"WARN","event":"commit.blocked","trace_id":"tr_001","reason":"source_confidence < 0.7"}
```

### 13.3 snowball status 输出

```text
Snowball Status（最近 7 天）
────────────────────────────────────────────────
已处理 turn：            142
  → completed：          54  (38.0%)
    ├─ created：          23  (16.2%)
    ├─ appended：         31  (21.8%)
    └─ archived：         67  (47.2%)  ← 含 skip 类
  → flagged：             12  ( 8.5%)
  → failed_retryable：    2  ( 1.4%)

Agent 健康度：
  平均步数：             3.2
  超过最大步数：          2   ( 1.4%)
  Tool 错误率：          0.8%
  Guardrail 拦截率：     2.1%
  Commit 拒绝率：        1.5%   ← v8 新增
  平均耗时：             2.3s
  平均 tokens/run：      1840

审核负担：
  Review rate：          8.5%
  Review 精准率：        81.0%
  自动操作接受率：        88.0%
  追加去重命中：          7 次

Parser 健康度：
  近 50 条平均 confidence：0.89
  低 confidence 率（< 0.6）：2.0%  ✓

Agent 健康度：
  近 100 次 flag rate：  8.5%  ✓
  近 100 次 max_steps exceeded：1.4%  ✓

Prompt 版本：agent_system/v1.1.md
上次 Vault-DB reconcile：2026-03-07 03:00（无异常）
────────────────────────────────────────────────
```

---

## 十四、Deterministic Replay

### 14.1 ReplayBundle（冻结输入）

```python
@dataclass
class ReplayBundle:
    trace_id: str
    event_json: str                    # StandardEvent 的完整 JSON
    prompt_snapshot: str               # 当时使用的 system prompt 全文
    config_snapshot_json: str          # 当时的配置快照
    tool_results_json: str             # 每个 Tool 调用的输入和输出
    knowledge_snapshot_refs_json: str  # 命中的 note_id + content_hash + 摘要
    model_name: str
    model_adapter_version: str
    created_at: datetime
```

`tool_results_json` 来自 `state.tool_results_for_replay`，在每次 Tool 执行后实时写入。
`knowledge_snapshot_refs_json` 来自 `state.knowledge_snapshot_refs`，在 `search_similar_notes` 执行时写入。

### 14.2 两种 Replay 模式

```python
class ReplayRunner:

    def logical_replay(self, trace_id: str) -> ReplayResult:
        """
        用冻结的 tool_results 重放 Agent 决策。
        Tool 调用返回预录的输出，不访问真实知识库。
        用途：debug 某次历史决策错误，验证 prompt/runtime 修改是否修复问题。
        """
        bundle = load_replay_bundle(trace_id)
        frozen_tool_outputs = json.loads(bundle.tool_results_json)

        mock_tools = [FrozenOutputTool(name=r["tool"], output=r["output"])
                      for r in frozen_tool_outputs]

        agent = SnowballAgent(model_adapter=self._make_adapter(bundle),
                               tools=mock_tools, max_steps=8)
        return agent.run(
            event=StandardEvent.from_json(bundle.event_json),
            task_id="replay",
            trace=AgentTrace.new(),
            db=SandboxDatabase(),
        )

    def live_replay(self, trace_id: str) -> ReplayResult:
        """
        用当前知识库重新执行 Tool，其余输入（prompt、config）保持和历史一致。
        用途：观察知识库演化后 Agent 行为是否发生漂移。
        """
        bundle = load_replay_bundle(trace_id)

        agent = SnowballAgent(model_adapter=self._make_adapter(bundle),
                               tools=LIVE_TOOLS, max_steps=8)
        return agent.run(
            event=StandardEvent.from_json(bundle.event_json),
            task_id="live_replay",
            trace=AgentTrace.new(),
            db=SandboxDatabase(),
        )
```

---

## 十五、source_confidence 闭环

### 15.1 问题

`source_confidence` 是启发式规则，没有闭环意味着它只能靠人工经验更新，不能从实际运行中学习。

### 15.2 Feedback 表

```sql
CREATE TABLE confidence_feedback (
  feedback_id TEXT PRIMARY KEY,
  turn_id TEXT NOT NULL,
  source_confidence REAL NOT NULL,
  human_label TEXT NOT NULL,   -- 'trustworthy' | 'partial' | 'bad_parse'
  annotator TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### 15.3 校准流程

人工审核时，如果发现 confidence 和实际解析质量明显不符，在 Review UI 里写一条 feedback。
每周运行一次校准分析：

```python
def analyze_confidence_calibration(db: Database) -> CalibrationReport:
    """
    对比 source_confidence 区间与人工 label 分布，
    输出每个区间的准确率，帮助识别规则权重需要调整的地方。
    """
    feedbacks = db.fetchall("""
        SELECT source_confidence, human_label FROM confidence_feedback
    """)
    # 按 confidence 区间分桶：[0,0.3), [0.3,0.6), [0.6,0.8), [0.8,1.0]
    buckets = group_by_bucket(feedbacks)
    return CalibrationReport(
        buckets=buckets,
        recommendation=generate_weight_recommendation(buckets),
    )
```

短期目标不是上 ML，而是：看报告 → 调整启发式权重 → 跑一版 parser confidence eval。

---

## 十六、Eval 体系

### 16.1 四个维度

1. **决策质量**：`decision_accuracy`、`target_note_accuracy`
2. **安全性**：`false_write_rate`、`unsafe_merge_rate`、`proposal_rejection_rate`
3. **成本**：`avg_steps`、`avg_tokens`、`avg_duration_ms`
4. **人工负担**：`review_rate`、`review_precision`、`auto_action_acceptance_rate`

### 16.2 Replay 一致性指标（v8 新增）

```
logical_replay_match_rate
  同一 ReplayBundle 逻辑重放后，决策与原始一致的比例。
  → 验证 runtime 自身的确定性

live_replay_drift_rate
  用当前知识库 live replay 时，决策发生变化的比例。
  → 量化知识库演化对 Agent 行为的影响
```

### 16.3 安全性指标的标注方法

`false_write_rate` 和 `unsafe_merge_rate` 需要额外标注字段：

```python
@dataclass
class EvalCase:
    case_id: str
    turn: StandardEvent
    expected_decision: str              # create/append/archive/flag/skip
    expected_target_note: str | None
    expected_risk_level: str            # safe / needs_review / unsafe
    unsafe_if_written: bool             # Agent 写入时人工是否认为这是错的
    difficulty: str                     # easy / medium / hard
    annotator: str
    notes: str
```

标注规则：`unsafe_if_written = True` 的条件（满足任意一条）：

1. turn 与已有笔记存在结论冲突，Agent 选择 append 而非 flag
2. `source_confidence < 0.7`，Agent 却创建了知识笔记
3. turn 是闲聊/调试片段，Agent 却创建了知识笔记

计算公式：

```python
false_write_rate = len([
    c for c in cases
    if agent_result(c).has_write and c.unsafe_if_written
]) / len(cases)

unsafe_merge_rate = len([
    c for c in cases
    if c.expected_risk_level == "needs_review"
    and agent_result(c).decision != "flagged"
    and agent_result(c).has_write
]) / len([c for c in cases if c.expected_risk_level == "needs_review"])

proposal_rejection_rate = len([
    c for c in cases
    if agent_result(c).commit_rejected
]) / len([c for c in cases if agent_result(c).has_proposals])
```

### 16.4 Eval Runner（沙箱模式）

```python
class EvalRunner:
    def run(self, dataset: list[EvalCase], prompt_version: str) -> EvalReport:
        results = []
        for case in dataset:
            with sandbox_environment() as sandbox:
                result = run_agent(
                    event=case.turn,
                    db=sandbox.db,       # 隔离的测试 DB
                    vault=sandbox.vault, # 临时 Vault 目录
                    prompt_version=prompt_version,
                )
            score = self.score_case(case, result)
            results.append(score)
        return EvalReport.from_scores(results, prompt_version)
```

### 16.5 Eval 输出示例

```text
Eval Results — agent_system/v1.2 vs v1.1
──────────────────────────────────────────────
决策质量：
  Decision accuracy：      82.0% → 85.0%  (+3.0%) ✓
  Target note accuracy：   74.0% → 76.0%  (+2.0%) ✓

安全性：
  False write rate：        4.0% →  3.0%  (-1.0%) ✓
  Unsafe merge rate：       2.0% →  1.0%  (-1.0%) ✓
  Proposal rejection rate：      1.5%  (new)

成本：
  Avg steps：               3.4  →   3.1  (-0.3)  ✓
  Avg tokens：              1840  →  1790   (-50)  ✓

Replay 一致性：
  Logical replay match：        97.0%  (new)
  Live replay drift rate：       3.5%  (new)

Regressions：2 cases（见 eval/regressions/v1.2.md）
──────────────────────────────────────────────
PASS：v1.2 整体更好，建议推为 current。
```

---

## 十七、审核系统

### 17.1 Review UI 最低能力

审核界面展示：

- 原始问题 + 最终回答
- Agent 最终决策 + RunState 轨迹
- Proposals 列表（提议了什么、Committer 是否拒绝）
- source confidence
- Trace replay（每步 Tool call + observation + runtime_state）
- Transcript snippet

### 17.2 审核动作

- approve create / approve append / approve archive
- discard / create separate / mark as conflict

人工动作写回 `review_actions` 表，可选择性写入 `eval_cases`（转为 ground truth）。

---

## 十八、数据库设计

### 18.1 继承自 v4 的基础表

`conversation_events`、`tasks`、`notes`、`note_sources`、
`merge_logs`、`note_embeddings`、`transcript_cursors`

### 18.2 全部新增表

```sql
-- Agent 执行记录
CREATE TABLE agent_traces (
  trace_id TEXT PRIMARY KEY,
  turn_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  model_name TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT NOT NULL,
  total_steps INTEGER NOT NULL,
  exceeded_max_steps INTEGER NOT NULL DEFAULT 0,
  terminal_reason TEXT NOT NULL,
  final_decision TEXT NOT NULL,
  final_confidence REAL,
  total_input_tokens INTEGER,
  total_output_tokens INTEGER,
  total_duration_ms INTEGER,
  trace_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Session Memory（关系型，便于查询和 debug）
CREATE TABLE session_turns (
  conversation_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  processed_at TEXT NOT NULL,
  final_decision TEXT NOT NULL,
  PRIMARY KEY (conversation_id, turn_id)
);

CREATE TABLE session_note_actions (
  conversation_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  note_id TEXT NOT NULL,
  action_type TEXT NOT NULL,
  note_title TEXT,
  created_at TEXT NOT NULL,
  PRIMARY KEY (conversation_id, turn_id, note_id, action_type)
);

-- 两阶段提交的 Proposal Journal
CREATE TABLE action_proposals (
  proposal_id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  action_type TEXT NOT NULL,
  target_note_id TEXT,
  payload_json TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL DEFAULT 'proposed',  -- proposed / committed / discarded
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  committed_at TEXT
);

-- Replay 数据
CREATE TABLE replay_bundles (
  trace_id TEXT PRIMARY KEY,
  event_json TEXT NOT NULL,
  prompt_snapshot TEXT NOT NULL,
  config_snapshot_json TEXT NOT NULL,
  tool_results_json TEXT NOT NULL,
  knowledge_snapshot_refs_json TEXT NOT NULL,
  model_name TEXT NOT NULL,
  model_adapter_version TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 人工审核结果
CREATE TABLE review_actions (
  review_id TEXT PRIMARY KEY,
  turn_id TEXT NOT NULL,
  trace_id TEXT NOT NULL,
  final_action TEXT NOT NULL,
  final_target_note_id TEXT,
  reviewer TEXT,
  reason TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Eval 数据集
CREATE TABLE eval_cases (
  case_id TEXT PRIMARY KEY,
  turn_id TEXT,
  input_json TEXT NOT NULL,
  expected_decision TEXT NOT NULL,
  expected_target_note TEXT,
  expected_risk_level TEXT NOT NULL,
  unsafe_if_written INTEGER NOT NULL DEFAULT 0,
  difficulty TEXT NOT NULL,
  annotator TEXT,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Eval 运行记录
CREATE TABLE eval_runs (
  run_id TEXT PRIMARY KEY,
  prompt_version TEXT NOT NULL,
  model_name TEXT NOT NULL,
  total_cases INTEGER NOT NULL,
  decision_accuracy REAL NOT NULL,
  target_note_accuracy REAL NOT NULL,
  false_write_rate REAL NOT NULL,
  unsafe_merge_rate REAL NOT NULL,
  proposal_rejection_rate REAL,
  logical_replay_match_rate REAL,
  live_replay_drift_rate REAL,
  review_precision REAL,
  auto_action_acceptance_rate REAL,
  avg_steps REAL,
  avg_tokens REAL,
  avg_duration_ms REAL,
  result_json TEXT NOT NULL,
  ran_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Source confidence 校准反馈
CREATE TABLE confidence_feedback (
  feedback_id TEXT PRIMARY KEY,
  turn_id TEXT NOT NULL,
  source_confidence REAL NOT NULL,
  human_label TEXT NOT NULL,  -- 'trustworthy' | 'partial' | 'bad_parse'
  annotator TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 审计日志（统一出口）
CREATE TABLE audit_logs (
  audit_id TEXT PRIMARY KEY,
  event_type TEXT NOT NULL,    -- state_transition / commit_blocked / reconcile_mismatch 等
  level TEXT NOT NULL,         -- info / warn / error
  trace_id TEXT,
  turn_id TEXT,
  task_id TEXT,
  detail_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

---

## 十九、Vault-DB 一致性

### 19.1 Reconciliation 实现

```python
def reconcile_vault_and_db(vault_path: Path, db: Database) -> ReconcileReport:
    vault_files = {
        str(p.relative_to(vault_path))
        for p in vault_path.rglob("*.md")
        if not p.name.startswith(".")
    }
    db_paths = {
        r["vault_path"]
        for r in db.fetchall(
            "SELECT vault_path FROM notes WHERE status != 'deleted'"
        )
    }

    orphan_files = vault_files - db_paths   # Vault 有，DB 无
    missing_files = db_paths - vault_files  # DB 有，Vault 无

    report = ReconcileReport(orphan_files=orphan_files, missing_files=missing_files)

    if orphan_files or missing_files:
        write_audit_log(db, "reconcile_mismatch", report.to_dict(), level="warn")

    return report
```

### 19.2 触发时机

- Worker 启动时运行一次
- 每天定时运行（默认凌晨 3 点，可配置）
- `snowball status` 输出上次 reconcile 结果

---

## 二十、错误处理

| 类型 | 例子 | 处理 |
|------|------|------|
| Parser drift | transcript 格式变化 | 降低 `source_confidence`，health alert |
| Tool timeout | embedding 计算超时 | 返回 structured error，Agent 感知并调整 |
| Guardrail blocked | 超写入次数 | 返回 blocked result，Agent 感知 |
| Invalid tool input | LLM 参数格式错误 | schema 拦截，返回 validation error |
| Model API failure | 限流/超时 | 指数退避重试 3 次，失败后任务进 FAILED_RETRYABLE |
| Commit validation failed | confidence 不足 | PROPOSED_ACTIONS → FLAGGED，写 audit log |
| Vault write error | 磁盘满/权限问题 | COMMITTING → FAILED_RETRYABLE，等待重试 |
| Loop exhaustion | 超过 max_steps | RUNNING → FLAGGED，terminal_reason = exceeded_max_steps |

---

## 二十一、成本估算

| 参数 | 数值 |
|------|------|
| 日均 Codex turn 数 | 30 条 |
| 入队过滤率（confidence 过低 / 回答过短）| ~35% |
| 实际进入 Agent 的 turn 数 | ~20 条 |
| 平均 tokens/run | 1840（input 1500 + output 340）|
| 模型 | Claude Sonnet（$3/1M input，$15/1M output）|

**日均 LLM 成本：**

```
Input：  20 × 1500 = 30,000 tokens → $0.09/day
Output： 20 × 340  =  6,800 tokens → $0.10/day
日均 ≈ $0.19/day ≈ $5.7/month
```

| 使用强度 | 日均进入 Agent 的 turn | 月均成本 |
|----------|----------------------|---------|
| 轻度 | 7 条 | ~$2 |
| 中度（基准）| 20 条 | ~$6 |
| 重度 | 65 条 | ~$19 |

Embedding 成本（voyage-3）在笔记规模 < 500 条时可忽略不计（< $0.01/month）。

---

## 二十二、目录结构

```text
snowball-notes/
  bin/
    snowball-worker          # 启动 Agent worker
    snowball-review          # 启动审核 Web UI
    snowball-eval            # 运行 eval
    snowball-status          # 查看运行指标

  src/
    intake/
      transcript_poll.py
      transcript_watch.py
      transcript_parser.py
      confidence.py          # compute_source_confidence()
      receiver.py

    queue/
      task_store.py
      task_claim.py

    agent/
      runtime.py             # SnowballAgent：ReAct Loop + 终止合约
      state.py               # AgentState
      state_machine.py       # RunState、transition_state()、VALID_TRANSITIONS
      orchestrator.py        # Worker 主循环，负责 claim 和调用 runtime
      adapter.py             # AgentModelAdapter Protocol
      guardrails.py          # check_guardrail()
      result.py              # AgentResult
      trace.py               # AgentTrace、TraceStep
      commit.py              # Committer：两阶段提交
      proposals.py           # ActionProposal 数据结构
      replay.py              # ReplayBundle、ReplayRunner

      tools/
        __init__.py          # validated_tool_execute()、工具注册
        assess.py
        extract.py
        search.py            # SearchSimilarNotesTool（含 knowledge snapshot 写入）
        read.py
        propose_create.py
        propose_append.py
        propose_archive.py
        propose_link.py
        flag.py              # FlagForReviewTool（含 is_terminated 设置）

      memory/
        session_memory.py    # load / update / flush
        knowledge_index.py   # KnowledgeIndex Protocol

    embedding/
      provider.py            # EmbeddingProvider Protocol
      voyage.py
      local.py               # sentence-transformers 离线方案
      vector_store.py        # VectorStore Protocol
      sqlite_blob.py
      sqlite_vec.py          # 升级路径

    review/
      server.py              # FastAPI app
      static/

    eval/
      runner.py
      scorer.py              # 含安全性指标、Replay 一致性指标计算
      report.py
      fixtures/
        easy/
        medium/
        hard/

    calibrate/
      confidence_feedback.py # analyze_confidence_calibration()

    observability/
      logger.py              # 结构化 JSON Lines 日志
      metrics.py             # snowball status 指标聚合
      health.py              # parser_health、agent_health

    storage/
      sqlite.py
      vault.py
      audit.py               # write_audit_log()
      reconcile.py           # reconcile_vault_and_db()

    models/
      events.py
      tasks.py
      notes.py
      traces.py
      eval.py                # EvalCase（含 unsafe_if_written）
      proposals.py

    prompts/
      agent_system/
        v1.0.md
        CHANGELOG.md

  config.yaml
  config.schema.json
  migrations/
  tests/
    test_parser.py
    test_confidence.py       # source_confidence 规则单测
    test_state_machine.py    # 状态迁移合法性测试
    test_runtime.py          # ReAct Loop 单测（含终止合约）
    test_guardrails.py
    test_committer.py        # 两阶段提交单测（含校验失败场景）
    test_replay.py           # logical replay 和 live replay
    test_tools.py
    test_eval.py
    test_reconcile.py
  README.md
```

---

## 二十三、配置文件

```yaml
vault:
  path: "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/MyVault"
  inbox_dir: "Inbox"
  archive_dir: "Archive/Conversations"
  atomic_dir: "Knowledge/Atomic"

intake:
  mode: "transcript_poll"         # transcript_poll | transcript_watch | cli_wrap
  transcript_dir: "~/.codex/sessions"
  parser_version: "v1"
  poll_interval_seconds: 3

agent:
  model: "claude-sonnet-4-5"
  max_steps: 8
  prompt_version: "agent_system/current"
  min_response_length: 120        # 短于此直接跳过
  min_confidence_to_run: 0.5      # 低于此在入队前过滤
  max_writes_per_run: 1
  max_appends_per_run: 1

embedding:
  provider: "voyage"              # voyage | local
  voyage_model: "voyage-3"
  local_model: "all-MiniLM-L6-v2"
  vector_store: "sqlite_blob"     # sqlite_blob | sqlite_vec
  index_text_strategy: "title_plus_summary"

retrieval:
  title_match_threshold: 0.85
  tag_min_overlap: 2
  embedding_top_k: 5
  embedding_threshold: 0.80

guardrails:
  min_confidence_for_note: 0.70
  min_confidence_for_append: 0.85

worker:
  poll_interval_seconds: 10
  max_retries: 3
  claim_timeout_seconds: 300

reconcile:
  enabled: true
  run_on_startup: true
  schedule_cron: "0 3 * * *"

sanitize:
  enabled: true
  archive_notes_exempt: true
```

---

## 二十四、MVP 实现顺序

### Phase 1：跑通 Agent 闭环（2-3 周）

- [ ] Intake：transcript 重建 + `compute_source_confidence()`
- [ ] `AgentState` 定义
- [ ] `RunState` 枚举 + `transition_state()`（只实现核心迁移）
- [ ] ReAct Loop（含 `is_terminated` 终止合约）
- [ ] 5 个核心 Tool：`search / read / propose_create / propose_append / flag`
- [ ] Committer（先实现核心校验和 create/append 提交，link 后补）
- [ ] Session Memory（注入 initial messages）
- [ ] AgentTrace 记录
- [ ] ReplayBundle 收集（tool_results + knowledge_snapshot_refs）

**验收标准：** 给定一条真实 turn，Agent 完成 create / append / flag 三种动作之一，DB 中有完整 trace 和 ReplayBundle，Obsidian 中能看到实际产物。

### Phase 2：安全与可观测（2-3 周）

- [ ] Guardrails（check_guardrail，基于 AgentState）
- [ ] Tool 参数 Schema 校验
- [ ] Embedding 模块（先 voyage，再加 local fallback）
- [ ] 结构化日志
- [ ] `snowball status` 输出（含 commit 拒绝率）
- [ ] Parser Health + Agent Health
- [ ] Vault-DB reconciliation + audit_logs

**验收标准：** 连续运行 3 天，能通过 `snowball status` 和 trace 定位任意一次异常 run。

### Phase 3：Eval 体系（2-3 周）

- [ ] 构建 50-100 条 ground truth（含 `unsafe_if_written` 标注）
- [ ] Eval Runner（沙箱模式）
- [ ] 安全性指标（`false_write_rate`、`unsafe_merge_rate`、`proposal_rejection_rate`）
- [ ] Replay 一致性指标（`logical_replay_match_rate`、`live_replay_drift_rate`）
- [ ] 跑 baseline eval，做一次 prompt 迭代
- [ ] 生成可对比报告

**验收标准：** 能回答"这次 prompt 改动后，哪些维度变好了，哪些没变"，且有数字支撑。

### Phase 4：Review UI 与开源包装（1-2 周）

- [ ] FastAPI Review UI（含 trace replay 和 proposals 展示页面）
- [ ] confidence_feedback 写入入口
- [ ] README（Problem / Why Agent / How It Is Controlled / Results 四段）
- [ ] mock fixtures（让没有 Codex 的人也能跑 eval）
- [ ] 示例 trace、ReplayBundle 和 eval report

**验收标准：** 没有 Codex 环境的贡献者可以本地跑通 eval 和 logical replay。

---

## 二十五、面试叙事

### 你在做什么

> 我做了一个 knowledge curation agent。
> 它从 Codex transcript 中异步消费已完成 turn，动态决定是跳过、归档、
> 创建新笔记、更新旧笔记还是升级人工审核。
> 整个过程有状态机管理生命周期、Proposal/Commit 控制副作用、
> Trace 和 ReplayBundle 支持任意历史 run 的离线复现。

### 你解决的核心工程问题

- 如何把 transcript 重建成置信度可知的输入（source_confidence 启发式规则 + feedback 校准）
- 如何约束 Agent 的写入边界（AgentState + Guardrails 前置拦截，Committer 二次校验）
- 如何保证 Agent 在不确定时能正确停止（is_terminated 终止合约 + 状态机）
- 如何让推理与副作用解耦（Proposal/Commit 两阶段，推理阶段不产生最终写入）
- 如何让任意历史问题可复现（ReplayBundle 冻结输入，logical/live 两种 replay）
- 如何量化 Agent 决策质量（Eval 体系：决策准确率、安全性、成本、Replay 一致性）

### 最应该强调的差异化

> 大多数 Agent 项目的迭代依赖人工感知——"感觉好像变好了"。
> 我做了 eval 体系，让每次改动都有数字依据。
> 更重要的是，我的 Agent 有明确的运行状态机、副作用隔离层和可复现的 replay 机制。
> 它不是一个 while-loop 调 LLM 的脚本，而是一个有 runtime 约束的工程系统。
> 这三件事——状态机、副作用控制、replayability——
> 正好把我的服务端工程背景和 Agent 工程结合在了一起。

---

## 二十六、各版本演进对照

| 维度 | v6 | v7 | v8 | 最终版 |
|------|----|----|----|----|
| AgentState | 未定义 | 完整定义 | 继承 | 继承 + tool_results_for_replay |
| Session Memory 注入 | 无 | 注入 initial messages | 继承 | 继承 |
| 终止合约 | 仅描述 | is_terminated + break | 继承 | 继承 + proposals.clear() |
| source_confidence | 黑盒 | 启发式规则 | + feedback 校准 | 完整：规则 + 校准表 + 周期分析 |
| Embedding | 遗漏 | voyage/local 双 Provider | 继承 | 继承 |
| Vault-DB 一致性 | 遗漏 | 完整恢复 | 继承 | 继承 |
| 安全性 Eval 标注 | 无 | unsafe_if_written | 继承 | 继承 |
| 成本估算 | 无 | 补充 | 继承 | 更新（含过滤率） |
| 状态机 | 无 | 无 | RunState 枚举 + 迁移图 | + transition_state() 实现 |
| 副作用控制 | Tool 直接写入 | Tool 直接写入 | Proposal/Commit 两阶段 | + Committer 完整实现 |
| Replay | 无 | 无 | ReplayBundle 设计 | + tool_results 填充路径 + 两种 replay 实现 |
| audit_logs | 引用但无 schema | 引用但无 schema | 补充 schema | 完整 + write_audit_log() |
| metrics Tool 矛盾 | 存在 | 删除 | 删除 | 删除 |
