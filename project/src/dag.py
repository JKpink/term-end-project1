"""AgentNet DAG Engine — decentralized graph with adaptive edge weights."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .agents import Agent, AgentAbility, RouterDecision, TASK_TO_ABILITY_MAP
from .config import get_config


@dataclass
class DAGConfig:
    """Configuration for the DAG engine."""
    edge_prune_threshold: float = 0.3
    max_forward_hops: int = 3
    max_split_depth: int = 5


@dataclass
class AgentGraph:
    """Complete directed graph of agents with evolving edge weights and success rates.

    Matches AgentNet agentgraph.py + agent.py update_abilities/decay logic.
    """

    agents: dict[int, Agent] = field(default_factory=dict)
    edge_weights: dict[tuple[int, int], float] = field(default_factory=dict)
    edge_success_rate: dict[tuple[int, int], float] = field(default_factory=dict)
    config: DAGConfig = field(default_factory=DAGConfig)

    def __post_init__(self):
        for a_id in self.agents:
            self.agents[a_id].set_graph(self)
        for i in self.agents:
            for j in self.agents:
                if i != j:
                    self.edge_weights[(i, j)] = 1.0
                    self.edge_success_rate[(i, j)] = 0.0

    def get_neighbors(self, agent_id: int) -> list[int]:
        neighbors = []
        for j in self.agents:
            if j != agent_id and self.edge_weights.get((agent_id, j), 0) > self.config.edge_prune_threshold:
                neighbors.append(j)
        return neighbors

    def select_best_neighbor(self, agent_id: int, task_abilities: dict[str, float]) -> Optional[int]:
        best_id, best_score = None, -1.0
        for j in self.get_neighbors(agent_id):
            ability = self.agents[j].abilities
            score = sum(
                task_abilities.get(k, 0) * getattr(ability, k, 0.6)
                for k in task_abilities
            ) * self.edge_weights.get((agent_id, j), 1.0)
            if score > best_score:
                best_score = score
                best_id = j
        return best_id

    # ── Edge weight update (agentgraph.py line 130-148) ────────────

    def update_edge(self, from_id: int, to_id: int, execution_time: float, success: bool):
        """Multiplicative edge weight update with time factor. Matches agentgraph.py."""
        # Edge success rate (EMA)
        old_rate = self.edge_success_rate.get((from_id, to_id), 0.0)
        self.edge_success_rate[(from_id, to_id)] = old_rate * 0.9 + (1.0 if success else 0.0) * 0.1

        # Edge weight: multiplicative with time factor
        current_weight = self.edge_weights.get((from_id, to_id), 1.0)
        success_factor = 1.1 if success else 0.9
        time_factor = min(1.0, 1.0 / (execution_time * 0.1)) if execution_time > 0 else 1.0

        new_weight = current_weight * success_factor * time_factor
        new_weight = max(0.1, min(2.0, new_weight))
        self.edge_weights[(from_id, to_id)] = new_weight

        # Auto-prune on update
        if new_weight <= self.config.edge_prune_threshold:
            self.edge_weights.pop((from_id, to_id), None)
            self.edge_success_rate.pop((from_id, to_id), None)

    # ── Ability + decay (agent.py line 851-865, 904-914, 930-933) ─

    def update_abilities_on_success(self, agent_id: int, task_type: str,
                                     task_correlations: dict[str, float] | None = None):
        """AgentNet agent.py line 851-865: +0.1 main, +0.05×corr related."""
        self.agents[agent_id].abilities.update_on_success(task_type, task_correlations)

    def tick_decay(self, agent_id: int, task_type: str, success: bool):
        """AgentNet agent.py line 904-914: decay countdown management."""
        agent = self.agents[agent_id]
        cfg = get_config()

        # Decrement all countdowns
        for key in list(agent.decay_count_down.keys()):
            agent.decay_count_down[key] -= 1

        # Reset on success, init on first encounter
        if success:
            agent.decay_count_down[task_type] = agent.decay_interval
        elif task_type not in agent.decay_count_down:
            agent.decay_count_down[task_type] = agent.decay_interval

        # Execute decay for any type that reached zero
        for key, count in list(agent.decay_count_down.items()):
            if count <= 0:
                agent.abilities.decay(key, cfg.pool_decay_rate)
                agent.decay_count_down[key] = agent.decay_interval


# ── Task execution loop ─────────────────────────────────────────────

def run_dag_task(
    graph: AgentGraph,
    task: str,
    schema_text: str,
    difficulty: str = "sql_medium",
    allow_split: bool = True,
    allow_evolution: bool = True,
    allow_experience: bool = True,
) -> dict:
    """Execute a task through the AgentNet DAG."""
    trace: list[dict] = []
    current_id = 0
    visited: set[int] = set()
    forward_hops = 0
    depth = 0

    data_result: list = []
    final_output: dict = {}
    current_task = task

    while depth < graph.config.max_split_depth:
        if current_id in visited or current_id not in graph.agents:
            break
        visited.add(current_id)

        agent = graph.agents[current_id]
        data_summary = str(data_result)[:2000] if data_result else ""

        decision, decision_info = agent.decide_action(current_task, schema_text, data_summary)
        trace.append({
            "agent_id": current_id,
            "decision": decision.value,
            "decision_info": decision_info,
            "task": current_task[:300],
        })

        if decision == RouterDecision.SPLIT and allow_split:
            sub_tasks = decision_info.get("sub_tasks", [current_task])
            for sub_task in sub_tasks:
                sub_output = agent.execute(sub_task, schema_text, data_summary)
                trace.append({
                    "agent_id": current_id, "action": "split_execute",
                    "sub_task": sub_task[:200],
                })
                if current_id == 0:
                    final_output.setdefault("sql", sub_output.get("sql", ""))
                elif current_id == 1:
                    final_output.setdefault("findings", []).extend(sub_output.get("findings", []))

            task_abilities = {"reasoning": 0.5, "inference": 0.5}
            next_id = graph.select_best_neighbor(current_id, task_abilities)
            if next_id is not None:
                current_id = next_id
                forward_hops = 0
            depth += 1
            continue

        elif decision == RouterDecision.FORWARD:
            task_abilities = decision_info.get("task_abilities", {"reasoning": 0.5})
            next_id = graph.select_best_neighbor(current_id, task_abilities)
            if next_id is None or forward_hops >= graph.config.max_forward_hops:
                decision = RouterDecision.EXECUTE
            else:
                trace.append({"agent_id": current_id, "action": "forward", "to": next_id})
                current_id = next_id
                forward_hops += 1
                continue

        # EXECUTE
        t_start = time.time()
        exec_output = agent.execute(current_task, schema_text, data_summary)
        t_elapsed = time.time() - t_start

        trace.append({
            "agent_id": current_id, "action": "execute",
            "output_keys": list(exec_output.keys()),
            "elapsed": t_elapsed,
        })

        if current_id == 0:
            final_output["sql"] = exec_output.get("sql", "")
            final_output["sql_explanation"] = exec_output.get("explanation", "")
        elif current_id == 1:
            final_output["findings"] = exec_output.get("findings", [])
            final_output["statistics"] = exec_output.get("statistics", {})
            final_output["trends"] = exec_output.get("trends", "")
        elif current_id == 2:
            final_output["chart_type"] = exec_output.get("chart_type", "bar")
            final_output["chart_title"] = exec_output.get("title", "")
        elif current_id == 3:
            final_output["report"] = exec_output.get("report", "")

        # Success detection
        success = bool(final_output.get("sql") and "error" not in str(exec_output).lower())

        # Record experience
        if allow_experience:
            agent.record_experience(
                current_task, schema_text[:500], str(exec_output)[:500],
                success=1.0 if success else 0.0,
            )

        # Evolution updates
        next_in_pipeline = current_id + 1
        if next_in_pipeline < len(graph.agents):
            if allow_evolution:
                graph.update_edge(current_id, next_in_pipeline, t_elapsed, success)
                graph.update_abilities_on_success(current_id, difficulty)
                graph.tick_decay(current_id, difficulty, success)
            current_id = next_in_pipeline
            forward_hops = 0
        else:
            break

        depth += 1

    final_output["trace"] = trace
    return final_output
