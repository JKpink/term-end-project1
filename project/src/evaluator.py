"""Evaluation engine: EX/EM metrics, analysis scoring, summary generation."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .datasource import SQLiteDataSource


# ── Result set comparison ───────────────────────────────────────────

def compare_result_sets(pred_rows: list[dict], gold_rows: list[dict]) -> bool:
    """Compare two result sets for exact match (EX metric)."""
    pred_tuples = sorted([tuple(r.values()) for r in pred_rows])
    gold_tuples = sorted([tuple(r.values()) for r in gold_rows])

    if len(pred_tuples) != len(gold_tuples):
        return False

    for p_row, g_row in zip(pred_tuples, gold_tuples):
        if len(p_row) != len(g_row):
            return False
        for p_val, g_val in zip(p_row, g_row):
            if not _values_equal(p_val, g_val):
                return False
    return True


def _values_equal(a, b) -> bool:
    """NULL-safe value comparison with float tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 0.01
    return str(a).strip() == str(b).strip()


# ── SQL normalization (for EM) ──────────────────────────────────────

def normalize_sql(sql: str) -> str:
    """Normalize SQL for exact match comparison."""
    import re
    s = sql.strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.replace('"', "'")
    s = re.sub(r';\s*$', '', s)
    return s.lower()


# ── Analysis question scoring ───────────────────────────────────────

def score_analysis_question(result: dict, ground_truth: dict) -> dict:
    """Auto-score an analysis question (20% each dimension)."""
    scores = {}

    # 1. SQL executable (20%)
    sql = result.get("sql", "")
    if not sql:
        scores["sql_exec"] = 0.0
    elif result.get("error"):
        scores["sql_exec"] = 0.5
    else:
        scores["sql_exec"] = 1.0

    # 2. Data correctness - required elements coverage (30%)
    required = ground_truth.get("required_elements", [])
    output_text = str(result.get("rows", "")) + str(result.get("report", "")) + str(result.get("findings", ""))
    if required:
        hits = sum(1 for elem in required if elem.lower() in output_text.lower())
        scores["data_correct"] = hits / len(required)
    else:
        scores["data_correct"] = 0.5

    # 3. Chart appropriateness (20%)
    expected_chart = ground_truth.get("expected_chart_type", "")
    actual_chart = result.get("chart_type", "")
    scores["chart_ok"] = 1.0 if expected_chart and actual_chart and expected_chart in actual_chart else 0.5

    # 4. Report completeness (20%)
    min_sections = ground_truth.get("min_report_sections", 2)
    report = result.get("report", "")
    # Count markdown headers
    import re
    headers = re.findall(r'^#{1,3}\s', report, re.MULTILINE)
    scores["report_ok"] = min(1.0, len(headers) / min_sections)

    # 5. Analysis depth (10%) - check for numerical analysis indicators
    depth_indicators = ["增长", "下降", "趋势", "占比", "平均", "总计", "排名", "%", "倍"]
    depth_hits = sum(1 for w in depth_indicators if w in output_text)
    scores["depth"] = min(1.0, depth_hits / 3)

    # Weighted total
    weighted = (
        0.2 * scores["sql_exec"]
        + 0.3 * scores["data_correct"]
        + 0.2 * scores["chart_ok"]
        + 0.2 * scores["report_ok"]
        + 0.1 * scores["depth"]
    )
    scores["weighted_total"] = round(weighted, 3)
    return scores


# ── Per-question evaluation ─────────────────────────────────────────

def evaluate_question(
    q: dict,
    method_results: dict[str, dict],
) -> dict[str, dict]:
    """Evaluate a single question across all methods."""
    db = SQLiteDataSource(q["db_path"])
    gold_rows = db.execute_gold(q["gold_sql"])

    eval_results = {}
    for method_name, result in method_results.items():
        pred_rows = result.get("rows", [])
        ex = compare_result_sets(pred_rows, gold_rows)
        em = normalize_sql(result.get("sql", "")) == normalize_sql(q["gold_sql"])

        entry = {
            "EX": ex,
            "EM": em,
            "VE": result.get("error") is None,
            "elapsed": result.get("elapsed", 0),
            "row_count": result.get("row_count", 0),
        }

        # Analysis scoring
        if q.get("type") == "analysis" and q.get("ground_truth"):
            analysis_scores = score_analysis_question(result, q["ground_truth"])
            entry["analysis"] = analysis_scores

        eval_results[method_name] = entry

    return eval_results


# ── Summary generation ──────────────────────────────────────────────

def generate_summary(all_eval_results: list[dict], method_order: list[str]) -> dict:
    """Generate aggregate summary statistics across all questions."""
    summary = {}

    for method in method_order:
        ex_values = [r[method]["EX"] for r in all_eval_results if method in r]
        em_values = [r[method]["EM"] for r in all_eval_results if method in r]
        ve_values = [r[method]["VE"] for r in all_eval_results if method in r]
        elapsed_values = [r[method]["elapsed"] for r in all_eval_results if method in r]

        n = len(ex_values)
        summary[method] = {
            "EX": {
                "mean": statistics.mean(ex_values) if ex_values else 0,
                "std": statistics.stdev(ex_values) if n > 1 else 0,
            },
            "EM": {
                "mean": statistics.mean(em_values) if em_values else 0,
                "std": statistics.stdev(em_values) if n > 1 else 0,
            },
            "VE": {
                "mean": statistics.mean(ve_values) if ve_values else 0,
                "std": statistics.stdev(ve_values) if n > 1 else 0,
            },
            "elapsed_mean": statistics.mean(elapsed_values) if elapsed_values else 0,
            "count": n,
        }

    return summary


def format_summary_markdown(summary: dict, source_name: str = "全部") -> str:
    """Format summary as a Markdown table."""
    method_order = ["B1", "B2", "B3", "B4", "Ours", "Ours_no_evolution", "Ours_no_experience", "Ours_no_split"]
    display_names = {
        "B1": "B1: Single Direct",
        "B2": "B2: Single CoT",
        "B3": "B3: NL2SQL Only",
        "B4": "B4: 2-Agent Sequential",
        "Ours": "Ours: AgentNet 4-Agent",
        "Ours_no_evolution": "  w/o Evolution",
        "Ours_no_experience": "  w/o Experience",
        "Ours_no_split": "  w/o Split",
    }

    lines = [f"## {source_name} ({summary.get('total_questions', '?')}题)\n"]
    lines.append("| 方法 | EX(%) | EM(%) | VE(%) | 耗时(s) |")
    lines.append("|------|:---:|:---:|:---:|:---:|")

    ours_ex = 0
    for method in method_order:
        if method not in summary:
            continue
        s = summary[method]
        name = display_names.get(method, method)
        ex_str = f"{s['EX']['mean']*100:.1f}±{s['EX']['std']*100:.1f}"
        em_str = f"{s['EM']['mean']*100:.1f}±{s['EM']['std']*100:.1f}"
        ve_str = f"{s['VE']['mean']*100:.1f}±{s['VE']['std']*100:.1f}"
        time_str = f"{s['elapsed_mean']:.1f}"
        lines.append(f"| {name} | {ex_str} | {em_str} | {ve_str} | {time_str} |")
        if method == "Ours":
            ours_ex = s['EX']['mean'] * 100

    # Add delta row
    if "B4" in summary and "Ours" in summary:
        delta = (summary["Ours"]["EX"]["mean"] - summary["B4"]["EX"]["mean"]) * 100
        lines.append(f"\n**Ours vs B4 提升: +{delta:.1f}%**")
    if "B1" in summary and "Ours" in summary:
        delta_total = (summary["Ours"]["EX"]["mean"] - summary["B1"]["EX"]["mean"]) * 100
        lines.append(f"**Ours vs B1 提升: +{delta_total:.1f}%**")

    return "\n".join(lines)


def save_eval_results(all_question_results: list[dict], all_eval_results: list[dict], output_dir: Path):
    """Save evaluation results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw results
    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "question_results": all_question_results,
            "eval_results": all_eval_results,
        }, f, ensure_ascii=False, indent=2)

    # Summary
    method_order = ["B1", "B2", "B3", "B4", "Ours", "Ours_no_evolution", "Ours_no_experience", "Ours_no_split"]
    summary = generate_summary(all_eval_results, method_order)
    summary["total_questions"] = len(all_eval_results)
    md = format_summary_markdown(summary)

    with open(output_dir / "eval_summary.md", "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Results saved to {output_dir}")
    print(md)
