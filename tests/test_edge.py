"""Unit tests for the Edge model."""

import pytest
from models.edge import Edge, EdgeValidationError, ROAD_CONDITION_PENALTIES
from utils.enums import RoadCondition, TimeOfDay


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

ALL_TRAFFIC_FACTORS = {
    TimeOfDay.MORNING: 1.6,
    TimeOfDay.AFTERNOON: 1.2,
    TimeOfDay.EVENING: 1.5,
    TimeOfDay.NIGHT: 0.8,
}


def make_edge(
    from_node: str = "N01",
    to_node: str = "N02",
    distance: float = 2.5,
    base_capacity: int = 1500,
    road_condition: RoadCondition = RoadCondition.GOOD,
    traffic_factors: dict | None = None,
    **kwargs,
) -> Edge:
    return Edge(
        from_node=from_node,
        to_node=to_node,
        distance=distance,
        base_capacity=base_capacity,
        road_condition=road_condition,
        traffic_factors=traffic_factors if traffic_factors is not None else dict(ALL_TRAFFIC_FACTORS),
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Happy-path tests
# ─────────────────────────────────────────────────────────────────────────────


def test_edge_creation_success() -> None:
    edge = make_edge()
    assert edge.from_node == "N01"
    assert edge.to_node == "N02"
    assert edge.distance == 2.5
    assert edge.base_capacity == 1500
    assert edge.road_condition == RoadCondition.GOOD
    assert not edge.is_proposed
    assert edge.is_bidirectional


def test_get_weight_morning() -> None:
    edge = make_edge(distance=2.5, road_condition=RoadCondition.GOOD)
    # weight = 2.5 × 1.6 (morning) × 1.1 (GOOD) = 4.4
    expected = 2.5 * 1.6 * ROAD_CONDITION_PENALTIES[RoadCondition.GOOD]
    assert abs(edge.get_weight(TimeOfDay.MORNING) - expected) < 1e-9


def test_get_weight_excellent_condition() -> None:
    edge = make_edge(distance=3.0, road_condition=RoadCondition.EXCELLENT)
    # Excellent has penalty = 1.0 so weight = distance × traffic_factor
    expected = 3.0 * ALL_TRAFFIC_FACTORS[TimeOfDay.NIGHT] * 1.0
    assert abs(edge.get_weight(TimeOfDay.NIGHT) - expected) < 1e-9


def test_get_weight_poor_condition() -> None:
    edge = make_edge(distance=1.0, road_condition=RoadCondition.POOR)
    expected = 1.0 * ALL_TRAFFIC_FACTORS[TimeOfDay.MORNING] * ROAD_CONDITION_PENALTIES[RoadCondition.POOR]
    assert abs(edge.get_weight(TimeOfDay.MORNING) - expected) < 1e-9


def test_weight_is_cached() -> None:
    """Second call must return the memoised value without recomputation.

    This test verifies the caching contract: once a weight is computed for a
    given TimeOfDay it is stored and returned as-is on subsequent calls until
    the cache is explicitly invalidated via :meth:`Edge.invalidate_cache`.
    Mutating ``traffic_factors`` directly without invalidating the cache is
    intentional here – it is the exact scenario that ``invalidate_cache``
    exists to handle (see :func:`test_invalidate_cache_triggers_recomputation`).
    """
    edge = make_edge()
    weight1 = edge.get_weight(TimeOfDay.MORNING)
    # Directly mutate the factor without invalidating – cache returns old value.
    edge.traffic_factors[TimeOfDay.MORNING] = 99.0
    weight2 = edge.get_weight(TimeOfDay.MORNING)
    assert weight1 == weight2, "Cache must return the memoised value until invalidated."


def test_invalidate_cache_triggers_recomputation() -> None:
    edge = make_edge()
    weight_before = edge.get_weight(TimeOfDay.MORNING)
    edge.traffic_factors[TimeOfDay.MORNING] = 2.0
    edge.invalidate_cache()
    weight_after = edge.get_weight(TimeOfDay.MORNING)
    assert weight_before != weight_after


def test_missing_traffic_factor_defaults_to_1() -> None:
    """If a TimeOfDay key is absent, get_weight must use 1.0 as default."""
    edge = make_edge(traffic_factors={TimeOfDay.MORNING: 1.5})
    # AFTERNOON missing – should fall back to 1.0
    expected = 2.5 * 1.0 * ROAD_CONDITION_PENALTIES[RoadCondition.GOOD]
    assert abs(edge.get_weight(TimeOfDay.AFTERNOON) - expected) < 1e-9


def test_proposed_edge_flag() -> None:
    edge = make_edge(is_proposed=True)
    assert edge.is_proposed


def test_repr_contains_endpoints() -> None:
    edge = make_edge()
    r = repr(edge)
    assert "N01" in r
    assert "N02" in r


def test_edge_hash_and_equality() -> None:
    e1 = make_edge(from_node="A", to_node="B")
    e2 = make_edge(from_node="A", to_node="B", distance=99.0)
    assert e1 == e2  # equality by (from_node, to_node)
    assert hash(e1) == hash(e2)


def test_edge_inequality() -> None:
    e1 = make_edge(from_node="A", to_node="B")
    e2 = make_edge(from_node="B", to_node="A")
    assert e1 != e2


# ─────────────────────────────────────────────────────────────────────────────
# Validation error tests
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_from_node_raises() -> None:
    with pytest.raises(EdgeValidationError, match="from_node"):
        make_edge(from_node="")


def test_empty_to_node_raises() -> None:
    with pytest.raises(EdgeValidationError, match="to_node"):
        make_edge(to_node="")


def test_self_loop_raises() -> None:
    with pytest.raises(EdgeValidationError, match="Self-loop"):
        make_edge(from_node="N01", to_node="N01")


def test_zero_distance_raises() -> None:
    with pytest.raises(EdgeValidationError, match="distance"):
        make_edge(distance=0.0)


def test_negative_distance_raises() -> None:
    with pytest.raises(EdgeValidationError, match="distance"):
        make_edge(distance=-1.0)


def test_negative_capacity_raises() -> None:
    with pytest.raises(EdgeValidationError, match="base_capacity"):
        make_edge(base_capacity=-1)


def test_zero_traffic_factor_raises() -> None:
    with pytest.raises(EdgeValidationError, match="traffic_factor"):
        make_edge(traffic_factors={TimeOfDay.MORNING: 0.0})


def test_negative_traffic_factor_raises() -> None:
    with pytest.raises(EdgeValidationError, match="traffic_factor"):
        make_edge(traffic_factors={TimeOfDay.MORNING: -0.5})
