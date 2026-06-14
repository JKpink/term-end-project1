"""AgentNet-DA Agents: 4 specialized agents with Router + Executor."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

from .llm_client import LLMClient, extract_json
from .memory import ExperiencePool, ExperienceFragment

if TYPE_CHECKING:
    from .dag import AgentGraph


class RouterDecision(Enum):
    EXECUTE = "execute"
    FORWARD = "forward"
    SPLIT = "split"


# ── Ability-to-task-type mapping (matching AgentNet paper Table 4) ──

TASK_TO_ABILITY_MAP: dict[str, list[str]] = {
    "reasoning": ["reasoning", "inference"],
    "mathematical": ["mathematical", "reasoning"],
    "language": ["language", "knowledge"],
    "knowledge": ["knowledge", "language"],
    "sequence": ["sequence", "reasoning"],
    "spatial": ["spatial", "reasoning"],
    "inference": ["inference", "reasoning"],
    # For NL2SQL tasks:
    "sql_easy": ["mathematical"],
    "sql_medium": ["mathematical", "reasoning"],
    "sql_hard": ["mathematical", "reasoning", "inference"],
    "sql_extra": ["mathematical", "reasoning", "inference", "knowledge"],
}


@dataclass
class AgentAbility:
    """Multi-dimensional ability vector. Matches AgentNet Fig.8 dimensions."""
    mathematical: float = 0.6
    reasoning: float = 0.6
    language: float = 0.6
    knowledge: float = 0.6
    spatial: float = 0.6
    inference: float = 0.6

    def to_dict(self) -> dict[str, float]:
        return {
            "mathematical": self.mathematical,
            "reasoning": self.reasoning,
            "language": self.language,
            "knowledge": self.knowledge,
            "spatial": self.spatial,
            "inference": self.inference,
        }

    def update_on_success(self, task_type: str, task_correlations: dict[str, float] | None = None):
        """AgentNet agent.py line 851-865: +0.1 for matched abilities, +0.05×corr for correlated."""
        abilities = TASK_TO_ABILITY_MAP.get(task_type, ["reasoning"])
        for ability_type in abilities:
            current = getattr(self, ability_type, 0.6)
            setattr(self, ability_type, min(2.0, current + 0.1))

        if task_correlations:
            for related_type, correlation in task_correlations.items():
                if correlation > 0.3:
                    related_abilities = TASK_TO_ABILITY_MAP.get(related_type, [])
                    for ability_type in related_abilities:
                        current = getattr(self, ability_type, 0.6)
                        gain = 0.1 * correlation * 0.5
                        setattr(self, ability_type, min(2.0, current + gain))

    def decay(self, task_type: str, decay_rate: float = 0.1):
        """AgentNet agent.py line 930-933: decay abilities for a given task type."""
        abilities = TASK_TO_ABILITY_MAP.get(task_type, ["reasoning"])
        for ability_type in abilities:
            current = getattr(self, ability_type, 0.6)
            setattr(self, ability_type, max(0.1, current * (1.0 - decay_rate)))



# ── Agent Prompt Templates ──────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """你是一个Agent路由器。根据任务描述和当前Agent的能力评估，决定下一步操作。

你必须返回JSON格式:
{
  "decision": "execute" | "forward" | "split",
  "reason": "决策理由",
  "next_agent": 0-3 (仅forward/split时需要),
  "sub_tasks": ["子任务1", "子任务2"] (仅split时需要)
}

决策规则:
- execute: 当前Agent有能力完成此任务，直接执行
- forward: 当前Agent不适合，转发给更合适的Agent
- split: 任务复杂，拆分为子任务并分派"""


AGENT_SYSTEM_PROMPTS = {
    0: """你是一个SQL数据库专家。你的职责是:
1. 将自然语言问题转换为精确的SQLite SQL查询
2. 理解数据库Schema，选择正确的表和列
3. 使用JOIN、GROUP BY、子查询等构造复杂查询
4. 确保SQL语法正确，可以在SQLite上执行

输出格式:
```json
{
  "sql": "你的SQL查询语句",
  "explanation": "SQL的简要说明"
}
```""",

    1: """你是一个数据分析师。你的职责是:
1. 理解SQL查询返回的数据结果
2. 进行统计分析（趋势、对比、异常检测）
3. 提取关键发现和洞察
4. 用简洁清晰的中文总结分析结果

输出格式:
```json
{
  "findings": ["发现1", "发现2", ...],
  "statistics": {"关键指标": "数值"},
  "trends": "趋势分析",
  "anomalies": "异常说明(如有)"
}
```""",

    2: """你是一个数据可视化专家。你的职责是:
1. 根据数据特征选择合适的图表类型
2. 生成Plotly图表配置
3. 确保图表清晰、美观、信息量大

输出格式:
```json
{
  "chart_type": "bar" | "line" | "scatter" | "pie" | "histogram",
  "title": "图表标题",
  "x_axis": "X轴标签",
  "y_axis": "Y轴标签",
  "reason": "选择此图表类型的原因"
}
```""",

    3: """你是一个数据分析报告撰写者。你的职责是:
1. 整合SQL结果、分析发现、图表建议
2. 撰写结构清晰的分析报告（Markdown格式）
3. 提供可操作的业务建议
4. 确保报告逻辑连贯、语言专业

报告结构:
## 概述
## 数据分析
## 关键发现
## 可视化建议
## 业务建议
""",
}

ROUTER_USER_TEMPLATE = """当前任务: {task}
数据库Schema: {schema}
数据查询结果: {data_summary}

当前Agent能力:
{abilities}

其他Agent信息:
{other_agents}

请做出路由决策。"""

EXECUTOR_USER_TEMPLATE = """任务: {task}
数据库Schema: {schema}
数据查询结果: {data_summary}

相关历史经验:
{experiences}

请执行你的专业任务。"""


@dataclass
class Agent:
    """An AgentNet agent with Router + Executor + ExperiencePool + decay countdown."""

    agent_id: int
    abilities: AgentAbility = field(default_factory=AgentAbility)
    router_pool: ExperiencePool = field(default_factory=ExperiencePool)
    executor_pool: ExperiencePool = field(default_factory=ExperiencePool)
    decay_count_down: dict[str, int] = field(default_factory=dict)
    decay_interval: int = 10
    _llm: LLMClient | None = None
    _graph: Optional["AgentGraph"] = None

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    def set_graph(self, graph: "AgentGraph"):
        self._graph = graph

    def decide_action(self, task: str, schema_text: str, data_summary: str = "") -> tuple[RouterDecision, dict]:
        """Router: decide whether to execute, forward, or split."""
        other_agents = ""
        if self._graph:
            for neighbor in self._graph.get_neighbors(self.agent_id):
                other_agents += f"Agent-{neighbor}: {self._graph.agents[neighbor].abilities.to_dict()}\n"

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": ROUTER_USER_TEMPLATE.format(
                task=task,
                schema=schema_text[:3000],
                data_summary=data_summary[:500] or "暂无",
                abilities=self.abilities.to_dict(),
                other_agents=other_agents or "无",
            )},
        ]
        raw = extract_json(self.llm.chat(messages, temperature=0.0))
        decision_str = raw.get("decision", "execute")
        try:
            decision = RouterDecision(decision_str)
        except ValueError:
            decision = RouterDecision.EXECUTE
        return decision, raw

    def execute(self, task: str, schema_text: str, data_summary: str = "") -> dict:
        """Executor: perform the agent's specialized task."""
        experiences = self.executor_pool.retrieve(task, schema_text[:500])
        exp_text = "\n".join([f.text for f in experiences[:3]]) if experiences else "无相关经验"

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPTS.get(self.agent_id, AGENT_SYSTEM_PROMPTS[0])},
            {"role": "user", "content": EXECUTOR_USER_TEMPLATE.format(
                task=task,
                schema=schema_text[:3000],
                data_summary=data_summary[:2000] or "暂无",
                experiences=exp_text,
            )},
        ]
        raw = self.llm.chat(messages, temperature=0.0)
        return extract_json(raw)

    def record_experience(self, task: str, context: str, action_output: str, success: float):
        """Record task outcome in both pools."""
        frag = ExperienceFragment(
            observation=task,
            context=context,
            action=action_output[:500],
            outcome=success,
        )
        self.executor_pool.add(frag)
        self.router_pool.add(frag)

    def __repr__(self) -> str:
        return f"Agent-{self.agent_id}({self.abilities.to_dict()})"
