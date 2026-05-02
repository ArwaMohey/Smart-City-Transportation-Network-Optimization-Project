"""Unit tests for the Graph class."""

import pytest
from core.graph import (
    DuplicateEdgeError,
    DuplicateNodeError,
    Graph,
    GraphError,
    NodeNotFoundError,
)
from models.edge import Edge
from models.node import Node
from utils.enums import NodeType, RoadCondition, TimeOfDay


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / Fixtures
# ─────────────────────────────────────────────────────────────────────────────

_TF = {
    TimeOfDay.MORNING: 1.6,
    TimeOfDay.AFTERNOON: 1.2,
    TimeOfDay.EVENING: 1.5,
    TimeOfDay.NIGHT: 0.8,
}


def _node(node_id: str, node_type: NodeType = NodeType.NEIGHBORHOOD) -> Node:
    return Node(
        id=node_id,
        name=f"Node {node_id}",
        node_type=node_type,
        population=1000,
        x_coordinate=100.0,
        y_coordinate=100.0,
    )


def _edge(from_node: str, to_node: str, distance: float = 1.0) -> Edge:
    return Edge(
        from_node=from_node,
        to_node=to_node,
        distance=distance,
        base_capacity=1000,
        road_condition=RoadCondition.GOOD,
        traffic_factors=dict(_TF),
    )


@pytest.fixture()
def empty_graph() -> Graph:
    return Graph()


@pytest.fixture()
def simple_graph() -> Graph:
    """Graph with nodes N01..N04 and edges N01-N02, N02-N03, N03-N04."""
    g = Graph()
    for nid in ("N01", "N02", "N03", "N04"):
        g.add_node(_node(nid))
    g.add_edge(_edge("N01", "N02"))
    g.add_edge(_edge("N02", "N03"))
    g.add_edge(_edge("N03", "N04"))
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_graph_starts_with_zero_nodes_and_edges(empty_graph: Graph) -> None:
    assert empty_graph.node_count == 0
    assert empty_graph.edge_count == 0


def test_directed_flag_stored(empty_graph: Graph) -> None:
    assert not empty_graph.directed
    dg = Graph(directed=True)
    assert dg.directed


# ─────────────────────────────────────────────────────────────────────────────
# Node operations
# ─────────────────────────────────────────────────────────────────────────────


def test_add_node_increases_count(empty_graph: Graph) -> None:
    empty_graph.add_node(_node("N01"))
    assert empty_graph.node_count == 1


def test_add_duplicate_node_raises(empty_graph: Graph) -> None:
    empty_graph.add_node(_node("N01"))
    with pytest.raises(DuplicateNodeError):
        empty_graph.add_node(_node("N01"))


def test_get_node_returns_correct_node(empty_graph: Graph) -> None:
    n = _node("N01")
    empty_graph.add_node(n)
    assert empty_graph.get_node("N01") is n


def test_get_nonexistent_node_raises(empty_graph: Graph) -> None:
    with pytest.raises(NodeNotFoundError):
        empty_graph.get_node("XXXX")


def test_remove_node_decreases_count(simple_graph: Graph) -> None:
    simple_graph.remove_node("N04")
    assert simple_graph.node_count == 3


def test_remove_node_removes_connected_edges(simple_graph: Graph) -> None:
    """Removing N02 must also remove edges N01-N02 and N02-N03."""
    simple_graph.remove_node("N02")
    assert simple_graph.get_neighbors("N01") == []
    assert simple_graph.get_neighbors("N03") == [_edge("N03", "N04")]


def test_remove_nonexistent_node_raises(empty_graph: Graph) -> None:
    with pytest.raises(NodeNotFoundError):
        empty_graph.remove_node("XXXX")


def test_get_all_nodes_count(simple_graph: Graph) -> None:
    assert len(simple_graph.get_all_nodes()) == 4


# ─────────────────────────────────────────────────────────────────────────────
# Edge operations
# ─────────────────────────────────────────────────────────────────────────────


def test_add_edge_increases_count(empty_graph: Graph) -> None:
    empty_graph.add_node(_node("A"))
    empty_graph.add_node(_node("B"))
    empty_graph.add_edge(_edge("A", "B"))
    assert empty_graph.edge_count == 1


def test_undirected_graph_registers_reverse_edge(empty_graph: Graph) -> None:
    empty_graph.add_node(_node("A"))
    empty_graph.add_node(_node("B"))
    empty_graph.add_edge(_edge("A", "B"))
    # Both directions must be traversable.
    assert len(empty_graph.get_neighbors("A")) == 1
    assert len(empty_graph.get_neighbors("B")) == 1


def test_directed_graph_does_not_register_reverse(empty_graph: Graph) -> None:
    dg = Graph(directed=True)
    dg.add_node(_node("A"))
    dg.add_node(_node("B"))
    dg.add_edge(_edge("A", "B"))
    assert len(dg.get_neighbors("A")) == 1
    assert len(dg.get_neighbors("B")) == 0


def test_add_duplicate_edge_raises(simple_graph: Graph) -> None:
    with pytest.raises(DuplicateEdgeError):
        simple_graph.add_edge(_edge("N01", "N02"))


def test_add_edge_with_missing_node_raises(empty_graph: Graph) -> None:
    empty_graph.add_node(_node("A"))
    with pytest.raises(NodeNotFoundError):
        empty_graph.add_edge(_edge("A", "MISSING"))


def test_remove_edge(simple_graph: Graph) -> None:
    simple_graph.remove_edge("N01", "N02")
    assert simple_graph.get_neighbors("N01") == []


def test_remove_nonexistent_edge_raises(simple_graph: Graph) -> None:
    with pytest.raises(GraphError):
        simple_graph.remove_edge("N01", "N04")


def test_undirected_edge_count_not_doubled(simple_graph: Graph) -> None:
    """Each bidirectional edge must be counted once."""
    assert simple_graph.edge_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# Query API
# ─────────────────────────────────────────────────────────────────────────────


def test_get_hospitals() -> None:
    g = Graph()
    g.add_node(_node("H01", NodeType.HOSPITAL))
    g.add_node(_node("N01", NodeType.NEIGHBORHOOD))
    hospitals = g.get_hospitals()
    assert len(hospitals) == 1
    assert hospitals[0].id == "H01"


def test_get_facilities_by_type() -> None:
    g = Graph()
    g.add_node(_node("F01", NodeType.FIRE_STATION))
    g.add_node(_node("F02", NodeType.FIRE_STATION))
    g.add_node(_node("N01", NodeType.NEIGHBORHOOD))
    stations = g.get_facilities_by_type(NodeType.FIRE_STATION)
    assert len(stations) == 2


def test_get_dynamic_edge_weight(simple_graph: Graph) -> None:
    w = simple_graph.get_dynamic_edge_weight("N01", "N02", TimeOfDay.MORNING)
    # distance=1.0, morning=1.6, GOOD penalty=1.1
    from models.edge import ROAD_CONDITION_PENALTIES
    expected = 1.0 * 1.6 * ROAD_CONDITION_PENALTIES[RoadCondition.GOOD]
    assert abs(w - expected) < 1e-9


def test_get_dynamic_edge_weight_missing_edge_raises(simple_graph: Graph) -> None:
    with pytest.raises(GraphError):
        simple_graph.get_dynamic_edge_weight("N01", "N04", TimeOfDay.MORNING)


# ─────────────────────────────────────────────────────────────────────────────
# Real-time traffic update
# ─────────────────────────────────────────────────────────────────────────────


def test_update_traffic_factor_changes_weight(simple_graph: Graph) -> None:
    old_weight = simple_graph.get_dynamic_edge_weight("N01", "N02", TimeOfDay.MORNING)
    simple_graph.update_traffic_factor("N01", "N02", TimeOfDay.MORNING, 3.0)
    new_weight = simple_graph.get_dynamic_edge_weight("N01", "N02", TimeOfDay.MORNING)
    assert new_weight > old_weight


def test_update_traffic_factor_invalid_raises(simple_graph: Graph) -> None:
    with pytest.raises(ValueError):
        simple_graph.update_traffic_factor("N01", "N02", TimeOfDay.MORNING, -1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


def test_validate_valid_graph_passes(simple_graph: Graph) -> None:
    """Should not raise for a structurally sound graph."""
    simple_graph.validate()


def test_validate_isolated_node_warns(empty_graph: Graph) -> None:
    """An isolated node should produce a warning but NOT raise an error."""
    empty_graph.add_node(_node("ALONE"))
    # validate() must not raise for isolated nodes (warning only).
    empty_graph.validate()


def test_repr_contains_counts(simple_graph: Graph) -> None:
    r = repr(simple_graph)
    assert "4" in r   # node count
    assert "3" in r   # edge count
