from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


import json
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return recall score between 0.0 and 1.0 depending on expected facts present in answer."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matched = sum(1 for exp in expected if exp.lower() in ans_lower)
    return matched / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight response quality score for offline mode."""
    if not answer or not answer.strip():
        return 0.0
    rec = recall_points(answer, expected)
    # Give high score if recall is satisfied and answer is well-formed
    length_bonus = 0.2 if len(answer.strip()) > 10 else 0.0
    return min(1.0, rec * 0.8 + length_bonus)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations."""
    all_threads = []
    user_ids = set()

    recall_scores = []
    quality_scores = []

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        user_ids.add(user_id)

        main_thread_id = f"{conv_id}_main"
        all_threads.append(main_thread_id)

        # 1. Feed all turns to agent in main thread
        for turn in conv["turns"]:
            agent.reply(user_id, main_thread_id, turn)

        # 2. Ask recall questions in fresh threads (simulating new sessions)
        for q_idx, q_item in enumerate(conv.get("recall_questions", [])):
            recall_thread_id = f"{conv_id}_recall_{q_idx}"
            all_threads.append(recall_thread_id)

            res = agent.reply(user_id, recall_thread_id, q_item["question"])
            answer = res.get("response", "")
            expected = q_item.get("expected_contains", [])

            rec = recall_points(answer, expected)
            qual = heuristic_quality(answer, expected)

            recall_scores.append(rec)
            quality_scores.append(qual)

    # Calculate metrics across all threads & users
    agent_tokens_only = sum(agent.token_usage(t) for t in all_threads)
    prompt_tokens_processed = sum(agent.prompt_token_usage(t) for t in all_threads)
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    compactions = sum(agent.compaction_count(t) for t in all_threads)

    memory_growth_bytes = 0
    if hasattr(agent, "memory_file_size"):
        memory_growth_bytes = sum(agent.memory_file_size(u) for u in user_ids)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens_only,
        prompt_tokens_processed=prompt_tokens_processed,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth_bytes,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a markdown table or tabulated output."""
    headers = [
        "Agent",
        "Agent Tokens Only",
        "Prompt Tokens Processed",
        "Recall Score",
        "Response Quality",
        "Memory Growth (bytes)",
        "Compactions",
    ]
    data = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score * 100:.1f}%",
            f"{r.response_quality * 100:.1f}%",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]

    if tabulate is not None:
        return tabulate(data, headers=headers, tablefmt="github")

    # Fallback Markdown table generator
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(str(val) for val in row) + " |" for row in data]
    return "\n".join([header_line, sep_line] + row_lines)


def main() -> None:
    """Run both benchmark suites (Standard & Long-Context Stress)."""
    config = load_config(Path(__file__).resolve().parent.parent)

    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    standard_convs = load_conversations(standard_path)
    stress_convs = load_conversations(stress_path)

    # --- 1. Standard Benchmark ---
    print("\n=======================================================")
    print(" STANDARD BENCHMARK (data/conversations.json)")
    print("=======================================================")
    baseline_std = BaselineAgent(config=config, force_offline=True)
    advanced_std = AdvancedAgent(config=config, force_offline=True)

    row_base_std = run_agent_benchmark("Baseline", baseline_std, standard_convs, config)
    row_adv_std = run_agent_benchmark("Advanced", advanced_std, standard_convs, config)
    print(format_rows([row_base_std, row_adv_std]))

    # --- 2. Long-Context Stress Benchmark ---
    print("\n=======================================================")
    print(" LONG-CONTEXT STRESS BENCHMARK (data/advanced_long_context.json)")
    print("=======================================================")
    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)

    row_base_stress = run_agent_benchmark("Baseline", baseline_stress, stress_convs, config)
    row_adv_stress = run_agent_benchmark("Advanced", advanced_stress, stress_convs, config)
    print(format_rows([row_base_stress, row_adv_stress]))
    print()


if __name__ == "__main__":
    main()
