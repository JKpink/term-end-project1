"""Tests for AgentGraph DAG engine (v2: AgentNet-aligned edge weight + decay)."""
import pytest
from src.agents import Agent, AgentAbility
from src.dag import AgentGraph, DAGConfig


@pytest.fixture
def graph():
    agents = {
        0: Agent(agent_id=0, abilities=AgentAbility(mathematical=0.8, reasoning=0.7)),
        1: Agent(agent_id=1, abilities=AgentAbility(reasoning=0.7, inference=0.6)),
        2: Agent(agent_id=2, abilities=AgentAbility(spatial=0.6, language=0.5)),
    }
    return AgentGraph(agents=agents, config=DAGConfig())


def test_graph_initialization(graph):
    assert len(graph.agents) == 3
    # Fully connected: 3*2 = 6 edges
    assert len(graph.edge_weights) == 6
    assert len(graph.edge_success_rate) == 6
    assert all(w == 1.0 for w in graph.edge_weights.values())
    assert all(s == 0.0 for s in graph.edge_success_rate.values())


def test_get_neighbors_all_connected(graph):
    neighbors = graph.get_neighbors(0)
    assert 1 in neighbors
    assert 2 in neighbors


def test_select_best_neighbor(graph):
    """Agent-0 should select the neighbor with best matching abilities."""
    task_abilities = {"reasoning": 0.8, "inference": 0.5}
    best = graph.select_best_neighbor(0, task_abilities)
    assert best is not None
    # Agent-1 has reasoning=0.7, inference=0.6 → higher score than Agent-2
    assert best == 1


def test_update_edge_success(graph):
    """AgentNet agentgraph.py line 130-148: multiplicative update."""
    graph.update_edge(0, 1, execution_time=2.0, success=True)
    # success_factor=1.1, time_factor=min(1, 1/(2.0*0.1))=min(1,5)=1
    # new = 1.0 * 1.1 * 1.0 = 1.1
    expected = 1.0 * 1.1 * 1.0
    assert graph.edge_weights[(0, 1)] == pytest.approx(expected)
    assert graph.edge_success_rate[(0, 1)] == pytest.approx(0.1)  # 0.9*0 + 0.1*1


def test_update_edge_failure(graph):
    graph.update_edge(0, 1, execution_time=1.0, success=False)
    # success_factor=0.9, time_factor=min(1, 1/(1.0*0.1))=min(1,10)=1
    # new = 1.0 * 0.9 * 1.0 = 0.9
    assert graph.edge_weights[(0, 1)] == pytest.approx(0.9)
    assert graph.edge_success_rate[(0, 1)] == pytest.approx(0.0)


def test_update_edge_with_time_factor(graph):
    """Slow execution penalizes edge weight."""
    graph.update_edge(0, 1, execution_time=20.0, success=True)
    # time_factor = min(1, 1/(20*0.1)) = min(1, 0.5) = 0.5
    # new = 1.0 * 1.1 * 0.5 = 0.55
    expected = 1.0 * 1.1 * 0.5
    assert graph.edge_weights[(0, 1)] == pytest.approx(expected)


def test_update_edge_auto_prune(graph):
    """Edge below threshold should be auto-removed."""
    graph.update_edge(0, 1, execution_time=100.0, success=False)
    # Edge should be pruned (removed entirely) because weight dropped below 0.3
    assert (0, 1) not in graph.edge_weights


def test_update_abilities_on_success(graph):
    a0_before = graph.agents[0].abilities.mathematical
    graph.update_abilities_on_success(0, "sql_medium")
    assert graph.agents[0].abilities.mathematical == pytest.approx(a0_before + 0.1)


def test_tick_decay_success(graph):
    """Success resets countdown, prevents decay."""
    agent = graph.agents[0]
    agent.decay_count_down["sql_medium"] = 1
    graph.tick_decay(0, "sql_medium", success=True)
    assert agent.decay_count_down["sql_medium"] == agent.decay_interval  # reset


def test_tick_decay_triggers_decay(graph):
    """Countdown reaching zero triggers ability decay."""
    agent = graph.agents[0]
    agent.decay_count_down["sql_medium"] = 1
    a_before = agent.abilities.mathematical
    graph.tick_decay(0, "sql_medium", success=False)
    # countdown: 1→0, decays, resets to interval
    assert agent.abilities.mathematical < a_before  # decayed
    assert agent.decay_count_down["sql_medium"] == agent.decay_interval


def test_neighbors_respect_pruned_edges(graph):
    graph.edge_weights[(0, 1)] = 0.1  # below threshold
    neighbors = graph.get_neighbors(0)
    assert 1 not in neighbors  # pruned
    assert 2 in neighbors      # still connected
