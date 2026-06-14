"""AgentNet-DA unified entry point."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from src.config import get_config
from src.datasource import SQLiteDataSource
from src.pipeline import run_single_question
from src.evaluator import evaluate_question, generate_summary, format_summary_markdown, save_eval_results


def cmd_demo(args):
    """Launch Gradio demo."""
    from app.gradio_app import create_demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


def cmd_eval(args):
    """Run evaluation on test suite."""
    test_suite_path = PROJECT_DIR / "data" / "test_questions.json"
    if not test_suite_path.exists():
        print(f"Test suite not found: {test_suite_path}")
        print("Run: python data/build_dataset.py first")
        return

    with open(test_suite_path, "r", encoding="utf-8") as f:
        test_suite = json.load(f)

    methods = args.methods.split(",") if args.methods else [
        "B1", "B2", "B3", "B4", "Ours",
        "Ours_no_evolution", "Ours_no_experience", "Ours_no_split",
    ]
    rounds = args.rounds
    sources = args.source.split(",") if args.source else None

    # Filter by source
    if sources:
        test_suite = [q for q in test_suite if q.get("source") in sources or q.get("database") in sources]

    print(f"Evaluating {len(test_suite)} questions × {len(methods)} methods × {rounds} rounds")
    print(f"Methods: {methods}")
    print(f"Source filter: {sources or 'all'}")
    print("=" * 60)

    all_round_summaries = []

    for round_num in range(1, rounds + 1):
        print(f"\n{'='*60}")
        print(f"Round {round_num}/{rounds}")
        print(f"{'='*60}")

        all_question_results = []
        all_eval_results = []

        for i, q in enumerate(test_suite):
            db_path = q.get("db_path", "")
            # Resolve path: try absolute, then relative to project/data
            if not os.path.exists(db_path):
                db_path = os.path.join(PROJECT_DIR, "data", db_path)
            if not os.path.exists(db_path):
                # Try project/data/{database}.db
                db_name = q.get("database", "")
                db_path = os.path.join(PROJECT_DIR, "data", f"{db_name}.db")

            if not os.path.exists(db_path):
                print(f"  [{i+1}/{len(test_suite)}] SKIP {q['id']}: db not found ({db_path})")
                continue

            print(f"  [{i+1}/{len(test_suite)}] {q['id']} ({q.get('difficulty','?')})", end="", flush=True)

            try:
                db = SQLiteDataSource(db_path)
                method_results = run_single_question(q["question"], db, methods=methods)
                all_question_results.append({
                    "id": q["id"],
                    "question": q["question"],
                    "difficulty": q.get("difficulty"),
                    "source": q.get("source", q.get("database")),
                    "methods": method_results,
                })

                if q.get("type") != "analysis":
                    eval_result = evaluate_question(q, method_results)
                    all_eval_results.append(eval_result)

                print(f" ✅")
            except Exception as e:
                print(f" ❌ {e}")

        # Save round results
        round_dir = PROJECT_DIR / "outputs" / f"round_{round_num:03d}"
        summary = generate_summary(all_eval_results, methods)
        summary["total_questions"] = len(all_eval_results)
        all_round_summaries.append(summary)
        save_eval_results(all_question_results, all_eval_results, round_dir)

    # Generate final summary
    _print_final_summary(all_round_summaries, methods, rounds)


def _print_final_summary(round_summaries: list[dict], methods: list[str], rounds: int):
    """Print final aggregated summary across all rounds."""
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY ({rounds} rounds)")
    print(f"{'='*60}")

    # Average EX across rounds for each method
    print(f"\n| 方法 | EX(%) | 标准差 |")
    print("|------|:---:|:---:|")
    for method in methods:
        ex_values = [s[method]["EX"]["mean"] * 100 for s in round_summaries if method in s]
        if ex_values:
            avg = sum(ex_values) / len(ex_values)
            std = (sum((x - avg) ** 2 for x in ex_values) / len(ex_values)) ** 0.5 if len(ex_values) > 1 else 0
            print(f"| {method} | {avg:.1f} | {std:.1f} |")


def main():
    parser = argparse.ArgumentParser(description="AgentNet-DA")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # demo
    p_demo = subparsers.add_parser("demo", help="Launch Gradio demo")
    p_demo.add_argument("--port", type=int, default=7860)
    p_demo.add_argument("--share", action="store_true")

    # eval
    p_eval = subparsers.add_parser("eval", help="Run evaluation")
    p_eval.add_argument("--methods", type=str, default=None,
                        help="Comma-separated methods (B1,B2,B3,B4,Ours)")
    p_eval.add_argument("--rounds", type=int, default=3,
                        help="Number of evaluation rounds")
    p_eval.add_argument("--source", type=str, default=None,
                        help="Filter by source (self_built,spider,bird,spider2)")

    args = parser.parse_args()

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
