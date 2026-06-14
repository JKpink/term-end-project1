"""Tests for ExperiencePool and ExperienceFragment."""
import pytest
from src.memory import ExperiencePool, ExperienceFragment


@pytest.fixture
def pool():
    return ExperiencePool(max_size=10, retrieval_count=3)


@pytest.fixture
def fragment():
    return ExperienceFragment("question", "context", "action", outcome=0.8)


def test_fragment_creation(fragment):
    assert fragment.observation == "question"
    assert fragment.context == "context"
    assert fragment.outcome == 0.8
    assert len(fragment.text) > 0


def test_fragment_hash(fragment):
    f2 = ExperienceFragment("question", "context", "action", outcome=0.5)
    assert hash(fragment) == hash(f2)  # same content → same hash


def test_add_fragment(pool):
    f = ExperienceFragment("q1", "c1", "a1", 0.5)
    pool.add(f)
    assert len(pool) == 1


def test_retrieve_empty(pool):
    results = pool.retrieve("query")
    assert results == []


def test_retrieve_keyword_fallback(pool):
    """When embedding is unavailable, keyword matching is used."""
    pool.add(ExperienceFragment("How many users", "schema", "SELECT COUNT(*)", 1.0))
    pool.add(ExperienceFragment("What products sold", "schema", "SELECT * FROM products", 0.8))
    pool.add(ExperienceFragment("Delivery time stats", "schema", "SELECT AVG(time)", 0.5))

    results = pool.retrieve("how many users in database")
    assert len(results) > 0
    # First result should be most relevant
    assert "How many users" in results[0].observation or "users" in results[0].observation


def test_eviction_when_full(pool):
    pool.max_size = 3
    pool.add(ExperienceFragment("q1", "c1", "a1", 0.1))  # lowest outcome
    pool.add(ExperienceFragment("q2", "c2", "a2", 0.9))
    pool.add(ExperienceFragment("q3", "c3", "a3", 0.5))
    assert len(pool) == 3

    # Add 4th, should evict q1 (outcome=0.1)
    pool.add(ExperienceFragment("q4", "c4", "a4", 0.7))
    assert len(pool) == 3
    observations = [f.observation for f in pool._fragments]
    assert "q1" not in observations  # evicted


def test_get_successful():
    pool = ExperiencePool(max_size=10)
    pool.add(ExperienceFragment("q1", "c", "a", 1.0))
    pool.add(ExperienceFragment("q2", "c", "a", 0.0))   # failure
    pool.add(ExperienceFragment("q3", "c", "a", -0.5))   # failure
    pool.add(ExperienceFragment("q4", "c", "a", 0.5))

    successful = pool.get_successful()
    assert len(successful) == 2  # q1 and q4

    failures = pool.get_failures()
    assert len(failures) == 2  # q2 and q3


def test_repr(pool):
    pool.add(ExperienceFragment("q1", "c1", "a1", 1.0))
    assert "ExperiencePool" in repr(pool)
    assert "1/10" in repr(pool)
