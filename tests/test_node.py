"""Unit tests for the Node model."""

import pytest
from models.node import Node, NodeValidationError
from utils.enums import NodeType


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def downtown_node() -> Node:
    return Node(
        id="N01",
        name="Downtown",
        node_type=NodeType.NEIGHBORHOOD,
        population=50_000,
        x_coordinate=500.0,
        y_coordinate=500.0,
    )


@pytest.fixture()
def hospital_node() -> Node:
    return Node(
        id="H01",
        name="City Central Hospital",
        node_type=NodeType.HOSPITAL,
        population=0,
        x_coordinate=490.0,
        y_coordinate=510.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Happy-path tests
# ─────────────────────────────────────────────────────────────────────────────


def test_node_creation_success(downtown_node: Node) -> None:
    assert downtown_node.id == "N01"
    assert downtown_node.name == "Downtown"
    assert downtown_node.node_type == NodeType.NEIGHBORHOOD
    assert downtown_node.population == 50_000
    assert downtown_node.x_coordinate == 500.0
    assert downtown_node.y_coordinate == 500.0


def test_node_is_immutable(downtown_node: Node) -> None:
    """Frozen dataclass must not allow attribute mutation."""
    with pytest.raises((AttributeError, TypeError)):
        downtown_node.name = "Changed"  # type: ignore[misc]


def test_node_is_not_critical_facility(downtown_node: Node) -> None:
    assert not downtown_node.is_critical_facility()


def test_hospital_is_critical_facility(hospital_node: Node) -> None:
    assert hospital_node.is_critical_facility()


@pytest.mark.parametrize(
    "node_type",
    [NodeType.FIRE_STATION, NodeType.POLICE, NodeType.SCHOOL],
)
def test_critical_facility_types(node_type: NodeType) -> None:
    node = Node(
        id="X01",
        name="Test Facility",
        node_type=node_type,
        population=0,
        x_coordinate=100.0,
        y_coordinate=100.0,
    )
    assert node.is_critical_facility()


def test_euclidean_distance(downtown_node: Node, hospital_node: Node) -> None:
    """Distance from (500,500) to (490,510) should be sqrt(200) ≈ 14.14."""
    dist = downtown_node.euclidean_distance_to(hospital_node)
    assert abs(dist - (200 ** 0.5)) < 1e-9


def test_zero_population_allowed() -> None:
    node = Node(
        id="F01",
        name="Fire Station",
        node_type=NodeType.FIRE_STATION,
        population=0,
        x_coordinate=0.0,
        y_coordinate=0.0,
    )
    assert node.population == 0


def test_boundary_coordinates() -> None:
    """Nodes at the exact boundary (0 and 1000) must be valid."""
    node = Node(
        id="B01",
        name="Boundary Node",
        node_type=NodeType.PARK,
        population=100,
        x_coordinate=0.0,
        y_coordinate=1000.0,
    )
    assert node.x_coordinate == 0.0
    assert node.y_coordinate == 1000.0


def test_repr_contains_id(downtown_node: Node) -> None:
    assert "N01" in repr(downtown_node)


# ─────────────────────────────────────────────────────────────────────────────
# Validation error tests
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_id_raises() -> None:
    with pytest.raises(NodeValidationError, match="id"):
        Node(
            id="",
            name="Test",
            node_type=NodeType.NEIGHBORHOOD,
            population=1000,
            x_coordinate=10.0,
            y_coordinate=10.0,
        )


def test_whitespace_id_raises() -> None:
    with pytest.raises(NodeValidationError):
        Node(
            id="   ",
            name="Test",
            node_type=NodeType.NEIGHBORHOOD,
            population=0,
            x_coordinate=10.0,
            y_coordinate=10.0,
        )


def test_empty_name_raises() -> None:
    with pytest.raises(NodeValidationError, match="name"):
        Node(
            id="N99",
            name="",
            node_type=NodeType.NEIGHBORHOOD,
            population=0,
            x_coordinate=10.0,
            y_coordinate=10.0,
        )


def test_negative_population_raises() -> None:
    with pytest.raises(NodeValidationError, match="population"):
        Node(
            id="N99",
            name="Test",
            node_type=NodeType.NEIGHBORHOOD,
            population=-1,
            x_coordinate=10.0,
            y_coordinate=10.0,
        )


def test_negative_x_coordinate_raises() -> None:
    with pytest.raises(NodeValidationError, match="x_coordinate"):
        Node(
            id="N99",
            name="Test",
            node_type=NodeType.NEIGHBORHOOD,
            population=0,
            x_coordinate=-1.0,
            y_coordinate=10.0,
        )


def test_out_of_bounds_y_coordinate_raises() -> None:
    with pytest.raises(NodeValidationError, match="y_coordinate"):
        Node(
            id="N99",
            name="Test",
            node_type=NodeType.NEIGHBORHOOD,
            population=0,
            x_coordinate=100.0,
            y_coordinate=1001.0,
        )


def test_invalid_node_type_raises() -> None:
    with pytest.raises(NodeValidationError):
        Node(
            id="N99",
            name="Test",
            node_type="hospital",  # type: ignore[arg-type]
            population=0,
            x_coordinate=100.0,
            y_coordinate=100.0,
        )
