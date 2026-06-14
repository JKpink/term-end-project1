"""Tests for Agent and AgentAbility (v2: AgentNet-aligned)."""
import pytest
from unittest.mock import MagicMock, patch
from src.agents import Agent, AgentAbility, RouterDecision, TASK_TO_ABILITY_MAP


def test_ability_initial():
    a = AgentAbility()
    assert a.mathematical == 0.6
    assert a.reasoning == 0.6
    assert a.language == 0.6


def test_ability_to_dict():
    a = AgentAbility(mathematical=0.8, language=0.5)
    d = a.to_dict()
    assert d["mathematical"] == 0.8
    assert d["language"] == 0.5


def test_ability_update_on_success():
    """AgentNet agent.py line 859-860: +0.1 for matched abilities."""
    a = AgentAbility(mathematical=0.6)
    a.update_on_success("sql_medium")  # maps to [mathematical, reasoning]
    assert a.mathematical == 0.7  # +0.1
    assert a.reasoning == 0.7    # +0.1


def test_ability_update_with_correlation():
    """AgentNet agent.py line 862-865: correlated abilities get +0.05×corr."""
    a = AgentAbility(mathematical=0.6, inference=0.6)
    a.update_on_success("sql_hard", {"sql_extra": 0.8})  # corr>0.3 triggers
    assert a.mathematical == pytest.approx(0.74)  # 0.6 + 0.1 + 0.1×0.8×0.5
    assert a.reasoning == pytest.approx(0.74)
    assert a.inference == pytest.approx(0.74)


def test_ability_decay():
    """AgentNet agent.py line 930-933: ×(1-decay_rate)."""
    a = AgentAbility(mathematical=0.6, reasoning=0.6)
    a.decay("sql_medium", decay_rate=0.1)
    assert a.mathematical == 0.54  # 0.6 * 0.9
    assert a.reasoning == 0.54


def test_ability_decay_clamp():
    """Decay should not go below 0.1."""
    a = AgentAbility(mathematical=0.1)
    a.decay("sql_easy", decay_rate=0.5)
    assert a.mathematical == 0.1  # clamped


def test_ability_success_clamp():
    """Success should not exceed 2.0."""
    a = AgentAbility(mathematical=2.0)
    a.update_on_success("sql_easy")  # mathematical + 0.1 but capped
    assert a.mathematical == 2.0


def test_agent_creation():
    agent = Agent(agent_id=0, abilities=AgentAbility(mathematical=0.7))
    assert agent.agent_id == 0
    assert agent.abilities.mathematical == 0.7
    assert len(agent.router_pool) == 0
    assert len(agent.executor_pool) == 0
    assert agent.decay_interval == 10
    assert agent.decay_count_down == {}


def test_agent_record_experience():
    agent = Agent(agent_id=1)
    agent.record_experience("task", "context", "output", success=1.0)
    assert len(agent.executor_pool) == 1
    assert len(agent.router_pool) == 1
    assert agent.executor_pool._fragments[0].outcome == 1.0


@pytest.mark.parametrize("decision_str,expected", [
    ("execute", RouterDecision.EXECUTE),
    ("forward", RouterDecision.FORWARD),
    ("split", RouterDecision.SPLIT),
])
def test_router_decision_enum(decision_str, expected):
    assert RouterDecision(decision_str) == expected


def test_agent_repr():
    agent = Agent(agent_id=2)
    r = repr(agent)
    assert "Agent-2" in r
    assert "mathematical" in r


def test_task_to_ability_map():
    """Verify ability mapping is correctly defined."""
    assert "sql_easy" in TASK_TO_ABILITY_MAP
    assert "mathematical" in TASK_TO_ABILITY_MAP["sql_easy"]
    assert "reasoning" in TASK_TO_ABILITY_MAP["sql_hard"]
    assert "inference" in TASK_TO_ABILITY_MAP["sql_extra"]
