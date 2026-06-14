"""Main pipeline: B1-B5 baselines + AgentNet DAG orchestration."""
from __future__ import annotations

import time
from typing import Any, Optional

from .config import get_config
from .llm_client import LLMClient
from .nl2sql import NL2SQLEngine
from .datasource import SQLiteDataSource
from .agents import Agent, AgentAbility
from .dag import AgentGraph, DAGConfig, run_dag_task


# ── B1: Single Direct ──────────────────────────────────────────────

def run_single_direct(question: str, schema_text: str, db: SQLiteDataSource) -> dict:
    """Single model generates SQL and returns result."""
    llm = LLMClient()
    start = time.time()

    nl2sql = NL2SQLEngine(llm)
    sql = nl2sql.generate_sql(question, schema_text)

    rows, error = db.execute_query(sql)
    elapsed = time.time() - start

    return {
        "method": "B1_Single_Direct",
        "sql": sql,
        "rows": rows[:20] if rows else [],
        "error": error,
        "elapsed": elapsed,
        "row_count": len(rows) if rows else 0,
    }


# ── B2: Single CoT ─────────────────────────────────────────────────

COT_SYSTEM_PROMPT = """你是一个SQL专家。请逐步推理然后给出SQL查询。

按以下格式输出:
思考: (逐步分析问题和Schema)
SQL: (最终的SQLite查询语句)"""


def run_single_cot(question: str, schema_text: str, db: SQLiteDataSource) -> dict:
    """Single model with Chain-of-Thought reasoning."""
    llm = LLMClient()
    start = time.time()

    messages = [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{schema_text}\n\n问题: {question}"},
    ]
    raw = llm.chat(messages, temperature=0.0)

    # Extract SQL from CoT output
    sql = ""
    for line in raw.split("\n"):
        line_stripped = line.strip()
        if line_stripped.upper().startswith("SQL:") or line_stripped.upper().startswith("SELECT"):
            sql = line_stripped.split(":", 1)[-1].strip() if ":" in line_stripped else line_stripped
            break
    if not sql:
        sql = raw  # fallback: use whole output

    rows, error = db.execute_query(sql)
    elapsed = time.time() - start

    return {
        "method": "B2_Single_CoT",
        "sql": sql,
        "rows": rows[:20] if rows else [],
        "error": error,
        "elapsed": elapsed,
        "reasoning": raw[:500],
        "row_count": len(rows) if rows else 0,
    }


# ── B3: NL2SQL Only (Agent-0 standalone) ───────────────────────────

def run_nl2sql_only(question: str, schema_text: str, db: SQLiteDataSource) -> dict:
    """Only Agent-0: NL2SQL without downstream analysis."""
    start = time.time()

    agent0 = Agent(agent_id=0, abilities=AgentAbility(mathematical=0.7, reasoning=0.7))
    exec_output = agent0.execute(question, schema_text)
    sql = exec_output.get("sql", "")

    if sql:
        rows, error = db.execute_query(sql)
    else:
        rows, error = [], "No SQL generated"

    elapsed = time.time() - start

    return {
        "method": "B3_NL2SQL_Only",
        "sql": sql,
        "rows": rows[:20] if rows else [],
        "error": error,
        "elapsed": elapsed,
        "row_count": len(rows) if rows else 0,
    }


# ── B4: 2-Agent Sequential (Agent-0 → Agent-3) ─────────────────────

def run_2agent_sequential(question: str, schema_text: str, db: SQLiteDataSource) -> dict:
    """Agent-0 (SQL) → Agent-3 (Report), skipping analysis and visualization."""
    start = time.time()

    # Agent-0: SQL
    agent0 = Agent(agent_id=0, abilities=AgentAbility(mathematical=0.7, reasoning=0.7))
    sql_output = agent0.execute(question, schema_text)
    sql = sql_output.get("sql", "")

    if sql:
        rows, error = db.execute_query(sql)
    else:
        rows, error = [], "No SQL generated"

    data_str = str(rows[:10])[:2000] if rows else "无数据"

    # Agent-3: Report
    agent3 = Agent(agent_id=3, abilities=AgentAbility(language=0.8, knowledge=0.6))
    report_output = agent3.execute(
        f"问题: {question}\n查询结果: {data_str}",
        schema_text,
        data_str,
    )
    report = report_output if isinstance(report_output, str) else str(report_output)

    elapsed = time.time() - start

    return {
        "method": "B4_2Agent_Sequential",
        "sql": sql,
        "rows": rows[:20] if rows else [],
        "error": error,
        "elapsed": elapsed,
        "report": report[:1000],
        "row_count": len(rows) if rows else 0,
    }


# ── Ours: AgentNet 4-Agent DAG ─────────────────────────────────────

def run_agentnet_dag(
    question: str,
    schema_text: str,
    db: SQLiteDataSource,
    allow_split: bool = True,
    allow_evolution: bool = True,
    allow_experience: bool = True,
) -> dict:
    """Full AgentNet DAG with 4 agents, evolution, and experience pool."""
    start = time.time()

    # Create 4 agents
    agents = {
        0: Agent(agent_id=0, abilities=AgentAbility(mathematical=0.7, reasoning=0.7)),
        1: Agent(agent_id=1, abilities=AgentAbility(reasoning=0.7, inference=0.7)),
        2: Agent(agent_id=2, abilities=AgentAbility(spatial=0.7, knowledge=0.6)),
        3: Agent(agent_id=3, abilities=AgentAbility(language=0.8, knowledge=0.6)),
    }

    # Create DAG
    graph = AgentGraph(agents=agents, config=DAGConfig())

    # Run task through DAG
    dag_output = run_dag_task(
        graph=graph,
        task=question,
        schema_text=schema_text,
        allow_split=allow_split,
        allow_evolution=allow_evolution,
        allow_experience=allow_experience,
    )

    sql = dag_output.get("sql", "")
    rows, error = [], None
    if sql:
        rows, error = db.execute_query(sql)

    elapsed = time.time() - start

    return {
        "method": "Ours_AgentNet_DAG",
        "sql": sql,
        "rows": rows[:20] if rows else [],
        "error": error,
        "elapsed": elapsed,
        "row_count": len(rows) if rows else 0,
        "findings": dag_output.get("findings", []),
        "chart_type": dag_output.get("chart_type", ""),
        "chart_title": dag_output.get("chart_title", ""),
        "report": dag_output.get("report", ""),
        "trace": dag_output.get("trace", []),
        "agent_count": 4,
    }


# ── Pipeline Runner ─────────────────────────────────────────────────

METHOD_NAMES = {
    "B1": "B1_Single_Direct",
    "B2": "B2_Single_CoT",
    "B3": "B3_NL2SQL_Only",
    "B4": "B4_2Agent_Sequential",
    "Ours": "Ours_AgentNet_DAG",
    "Ours_no_evolution": "Ours_AgentNet_DAG",
    "Ours_no_experience": "Ours_AgentNet_DAG",
    "Ours_no_split": "Ours_AgentNet_DAG",
}


def run_single_question(
    question: str,
    db: SQLiteDataSource,
    methods: list[str] | None = None,
) -> dict[str, dict]:
    """Run a single question through specified methods."""
    schema_text = db.get_schema_text()

    if methods is None:
        methods = ["B1", "B2", "B3", "B4", "Ours", "Ours_no_evolution", "Ours_no_experience", "Ours_no_split"]

    results = {}

    for method in methods:
        try:
            if method == "B1":
                results[method] = run_single_direct(question, schema_text, db)
            elif method == "B2":
                results[method] = run_single_cot(question, schema_text, db)
            elif method == "B3":
                results[method] = run_nl2sql_only(question, schema_text, db)
            elif method == "B4":
                results[method] = run_2agent_sequential(question, schema_text, db)
            elif method == "Ours":
                results[method] = run_agentnet_dag(question, schema_text, db)
            elif method == "Ours_no_evolution":
                results[method] = run_agentnet_dag(question, schema_text, db, allow_evolution=False)
            elif method == "Ours_no_experience":
                results[method] = run_agentnet_dag(question, schema_text, db, allow_experience=False)
            elif method == "Ours_no_split":
                results[method] = run_agentnet_dag(question, schema_text, db, allow_split=False)
        except Exception as e:
            results[method] = {
                "method": METHOD_NAMES.get(method, method),
                "sql": "",
                "rows": [],
                "error": str(e),
                "elapsed": 0,
            }

    return results
