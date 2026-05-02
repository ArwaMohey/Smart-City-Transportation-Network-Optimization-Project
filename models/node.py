"""Node model for the Smart City Transportation Network System.

A :class:`Node` represents any location in the city that can be a
source, destination, or intermediate stop in the transport network.
Locations include residential neighbourhoods as well as critical
facilities such as hospitals, fire stations, and schools.

The class is *frozen* (immutable after construction) to allow safe use
as a dictionary key or inside sets, and to prevent accidental mutation
across algorithm implementations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from utils.enums import NodeType

logger = logging.getLogger("smart_city.models.node")

# ── Coordinate validation bounds (city grid, arbitrary units) ─────────────
_COORD_MIN: float = 0.0
_COORD_MAX: float = 1000.0

# ── Critical facility types (used for is_critical_facility helper) ────────
_CRITICAL_TYPES: frozenset[NodeType] = frozenset(
    {
        NodeType.HOSPITAL,
        NodeType.FIRE_STATION,
        NodeType.POLICE,
        NodeType.SCHOOL,
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────


class NodeValidationError(ValueError):
    """Raised when a :class:`Node` is constructed with invalid attribute values."""


# ─────────────────────────────────────────────────────────────────────────────
# Node dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Node:
    """Immutable representation of a location in the city transport network.

    Attributes:
        id:           Unique identifier string (e.g. ``"N01"``, ``"H01"``).
        name:         Human-readable label (e.g. ``"Downtown"``).
        node_type:    Category of the location; one of :class:`~utils.enums.NodeType`.
        population:   Resident or user population count (≥ 0).
        x_coordinate: Horizontal position on the city grid in [0, 1000].
        y_coordinate: Vertical position on the city grid in [0, 1000].

    Raises:
        NodeValidationError: If any attribute fails validation.

    Example::

        from utils.enums import NodeType
        node = Node(
            id="N01",
            name="Downtown",
            node_type=NodeType.NEIGHBORHOOD,
            population=50_000,
            x_coordinate=500.0,
            y_coordinate=500.0,
        )
    """

    id: str
    name: str
    node_type: NodeType
    population: int
    x_coordinate: float
    y_coordinate: float

    # ------------------------------------------------------------------
    # Post-construction validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate all attributes immediately after dataclass construction."""
        self._validate_id()
        self._validate_name()
        self._validate_node_type()
        self._validate_population()
        self._validate_coordinates()
        logger.debug("Node created: id=%r name=%r type=%s", self.id, self.name, self.node_type.value)

    def _validate_id(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise NodeValidationError(
                f"Node 'id' must be a non-empty string; got {self.id!r}."
            )

    def _validate_name(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise NodeValidationError(
                f"Node 'name' must be a non-empty string; got {self.name!r}."
            )

    def _validate_node_type(self) -> None:
        if not isinstance(self.node_type, NodeType):
            raise NodeValidationError(
                f"Node 'node_type' must be a NodeType instance; got {self.node_type!r}."
            )

    def _validate_population(self) -> None:
        if not isinstance(self.population, int) or self.population < 0:
            raise NodeValidationError(
                f"Node 'population' must be a non-negative integer; got {self.population!r}."
            )

    def _validate_coordinates(self) -> None:
        for attr, value in (
            ("x_coordinate", self.x_coordinate),
            ("y_coordinate", self.y_coordinate),
        ):
            if not isinstance(value, (int, float)):
                raise NodeValidationError(
                    f"Node '{attr}' must be a number; got {value!r}."
                )
            if not (_COORD_MIN <= float(value) <= _COORD_MAX):
                raise NodeValidationError(
                    f"Node '{attr}'={value} is outside the valid range "
                    f"[{_COORD_MIN}, {_COORD_MAX}]."
                )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def is_critical_facility(self) -> bool:
        """Return *True* if this node represents a critical emergency facility.

        Critical types are: HOSPITAL, FIRE_STATION, POLICE, SCHOOL.

        Returns:
            bool: ``True`` when the node type is a critical facility type.
        """
        return self.node_type in _CRITICAL_TYPES

    def euclidean_distance_to(self, other: Node) -> float:
        """Compute the straight-line (Euclidean) distance to another node.

        Useful as a heuristic in A* path-finding algorithms.

        Args:
            other: Another :class:`Node` instance.

        Returns:
            float: Euclidean distance in grid units.
        """
        dx = self.x_coordinate - other.x_coordinate
        dy = self.y_coordinate - other.y_coordinate
        return (dx * dx + dy * dy) ** 0.5

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Node(id={self.id!r}, name={self.name!r}, "
            f"type={self.node_type.value!r}, population={self.population:,}, "
            f"coords=({self.x_coordinate}, {self.y_coordinate}))"
        )
