"""AgentNet-DA Gradio Web UI."""
from __future__ import annotations

import sys
import os
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr

from src.config import get_config
from src.datasource import SQLiteDataSource
from src.pipeline import (
    run_single_direct,
    run_single_cot,
    run_nl2sql_only,
    run_2agent_sequential,
    run_agentnet_dag,
)

DEMO_DB = os.path.join(os.path.dirname(__file__), "..", "data", "campus_trade.db")


def process_question(question: str, db_path: str, method_choice: str):
    """Process a question through the selected method(s), yield progress updates."""
    if not question.strip():
        yield "请输入问题", "", "", "", ""
        return

    if not os.path.exists(db_path):
        yield f"数据库文件不存在: {db_path}", "", "", "", ""
        return

    db = SQLiteDataSource(db_path)
    schema_text = db.get_schema_text()

    methods = [m.strip() for m in method_choice.split(",")]
    all_results: dict[str, dict] = {}

    for i, method in enumerate(methods):
        progress = f"({i+1}/{len(methods)}) 运行 {method}..."

        try:
            if method == "B1":
                result = run_single_direct(question, schema_text, db)
            elif method == "B2":
                result = run_single_cot(question, schema_text, db)
            elif method == "B3":
                result = run_nl2sql_only(question, schema_text, db)
            elif method == "B4":
                result = run_2agent_sequential(question, schema_text, db)
            elif method == "Ours":
                result = run_agentnet_dag(question, schema_text, db)
            else:
                continue

            all_results[method] = result
        except Exception as e:
            all_results[method] = {"error": str(e), "elapsed": 0}

    # Format output
    progress_text = _format_progress(all_results)
    sql_text = _format_sql(all_results)
    data_text = _format_data(all_results)
    analysis_text = _format_analysis(all_results)
    report_text = _format_report(all_results)

    yield progress_text, sql_text, data_text, analysis_text, report_text


def _format_progress(results: dict) -> str:
    lines = ["## 执行进度\n"]
    for method, r in results.items():
        if r.get("error"):
            lines.append(f"- **{method}**: ❌ {r['error'][:100]}")
        else:
            elapsed = r.get("elapsed", 0)
            row_count = r.get("row_count", 0)
            lines.append(f"- **{method}**: ✅ {elapsed:.1f}s | {row_count}行")
    return "\n".join(lines)


def _format_sql(results: dict) -> str:
    lines = ["## SQL\n"]
    for method, r in results.items():
        sql = r.get("sql", "")
        if sql:
            lines.append(f"### {method}\n```sql\n{sql}\n```\n")
    return "\n".join(lines)


def _format_data(results: dict) -> str:
    lines = ["## 数据预览\n"]
    for method, r in results.items():
        rows = r.get("rows", [])
        if rows:
            lines.append(f"### {method} ({len(rows)}行)\n")
            # Table header
            cols = list(rows[0].keys())
            lines.append("| " + " | ".join(cols[:8]) + " |")
            lines.append("|" + "|".join(["---"] * min(len(cols), 8)) + "|")
            for row in rows[:10]:
                vals = [str(row.get(c, ""))[:50] for c in cols[:8]]
                lines.append("| " + " | ".join(vals) + " |")
            if len(rows) > 10:
                lines.append(f"\n*...共{len(rows)}行，仅显示前10行*")
            lines.append("")
    if all(not r.get("rows") for r in results.values()):
        lines.append("*无数据返回*")
    return "\n".join(lines)


def _format_analysis(results: dict) -> str:
    lines = ["## 分析结果\n"]
    for method, r in results.items():
        findings = r.get("findings", [])
        trends = r.get("trends", "")
        chart = r.get("chart_type", "")
        if findings:
            lines.append(f"### {method}\n")
            for f in findings:
                lines.append(f"- {f}")
            if trends:
                lines.append(f"\n**趋势**: {trends}")
            if chart:
                lines.append(f"\n**图表建议**: {chart} - {r.get('chart_title', '')}")
            lines.append("")
    if not any(r.get("findings") for r in results.values()):
        lines.append("*仅SQL模式无分析结果*")
    return "\n".join(lines)


def _format_report(results: dict) -> str:
    for r in results.values():
        report = r.get("report", "")
        if report:
            return f"## 分析报告\n\n{report}"
    return "## 分析报告\n\n*仅全流程方法(B4/Ours)生成报告*"


def create_demo():
    """Create and return the Gradio Blocks app."""
    with gr.Blocks(
        title="AgentNet-DA: 多智能体数据分析",
        theme=gr.themes.Soft(primary_hue="blue"),
    ) as demo:
        gr.Markdown("""
        # 🤖 AgentNet-DA: 多智能体数据分析系统
        基于 AgentNet (NeurIPS 2025) 的 DAG 去中心化多智能体框架，实现零训练的 NL2SQL 数据分析。
        """)

        with gr.Row():
            with gr.Column(scale=1):
                question_input = gr.Textbox(
                    label="📝 输入问题",
                    placeholder="例: 计算机学院有多少用户？各学院交易总额排名？",
                    lines=3,
                )
                db_input = gr.Textbox(
                    label="🗄️ 数据库路径",
                    value=DEMO_DB,
                )
                method_choice = gr.Dropdown(
                    label="⚙️ 方法",
                    choices=["B1", "B2", "B3", "B4", "Ours", "B1,B2,B3,B4,Ours"],
                    value="B1,B2,B3,B4,Ours",
                    multiselect=False,
                )
                run_btn = gr.Button("🚀 执行分析", variant="primary")

                gr.Markdown("""
                ### 方法说明
                - **B1**: 单模型直接生成SQL
                - **B2**: 单模型+思维链推理
                - **B3**: 仅Agent-0 NL2SQL
                - **B4**: Agent-0→Agent-3 两Agent顺序
                - **Ours**: 完整4-Agent DAG
                """)

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("📊 进度"):
                        progress_output = gr.Markdown("等待输入...")
                    with gr.TabItem("💻 SQL"):
                        sql_output = gr.Markdown("")
                    with gr.TabItem("📋 数据"):
                        data_output = gr.Markdown("")
                    with gr.TabItem("📈 分析"):
                        analysis_output = gr.Markdown("")
                    with gr.TabItem("📄 报告"):
                        report_output = gr.Markdown("")

        run_btn.click(
            fn=process_question,
            inputs=[question_input, db_input, method_choice],
            outputs=[progress_output, sql_output, data_output, analysis_output, report_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
