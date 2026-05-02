"""Edge model for the Smart City Transportation Network System.

An :class:`Edge` represents a road segment connecting two nodes.  Its
effective travel cost (*weight*) is computed dynamically by combining:

1. **Physical distance** – the road length in kilometres.
2. **Road-condition penalty** – a fixed multiplier reflecting surface quality.
3. **Traffic congestion factor** – a time-of-day multiplier sourced from
   real (or simulated) traffic data.

Computed weights are cached inside the instance to avoid redundant
arithmetic on repeated look-ups.  The cache is invalidated whenever
traffic factors or road condition change (e.g. after a real-time update).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

from utils.enums import RoadCondition, TimeOfDay

logger = logging.getLogger("smart_city.models.edge")

# ── Road-condition penalty multipliers ───────────────────────────────────────
ROAD_CONDITION_PENALTIES: Dict[RoadCondition, float] = {
    RoadCondition.EXCELLENT: 1.0,
    RoadCondition.GOOD: 1.1,
    RoadCondition.FAIR: 1.3,
    RoadCondition.POOR: 1.6,
}

# ── Validation thresholds ─────────────────────────────────────────────────────
_MIN_DISTANCE: float = 0.0
_MIN_CAPACITY: int = 0
_MIN_TRAFFIC_FACTOR: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────


class EdgeValidationError(ValueError):
    """Raised when an :class:`Edge` is constructed with invalid attribute values."""


# ─────────────────────────────────────────────────────────────────────────────
# Edge dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Edge:
    """Mutable representation of a road segment in the transport network.

    Attributes:
        from_node:       Identifier of the originating node.
        to_node:         Identifier of the destination node.
        distance:        Road length in kilometres (> 0).
        base_capacity:   Maximum throughput in vehicles per hour (≥ 0).
        road_condition:  Surface quality; drives the condition-penalty multiplier.
        traffic_factors: Mapping of :class:`~utils.enums.TimeOfDay` → congestion
                         multiplier (all values must be > 0).
        is_proposed:     *True* for roads not yet constructed.
        is_bidirectional: *True* if the road is traversable in both directions.

    Raises:
        EdgeValidationError: If any attribute fails validation.

    Example::

        from utils.enums import RoadCondition, TimeOfDay
        edge = Edge(
            from_node="N01",
            to_node="N02",
            distance=2.5,
            base_capacity=1500,
            road_condition=RoadCondition.GOOD,
            traffic_factors={
                TimeOfDay.MORNING: 1.6,
                TimeOfDay.AFTERNOON: 1.2,
                TimeOfDay.EVENING: 1.5,
                TimeOfDay.NIGHT: 0.8,
            },
        )
        weight = edge.get_weight(TimeOfDay.MORNING)
    """

    from_node: str
    to_node: str
    distance: float
    base_capacity: int
    road_condition: RoadCondition
    traffic_factors: Dict[TimeOfDay, float]
    is_proposed: bool = False
    is_bidirectional: bool = True

    # Internal weight cache – excluded from __init__, __repr__, and equality.
    _weight_cache: Dict[TimeOfDay, float] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    # ------------------------------------------------------------------
    # Post-construction validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate all edge attributes immediately after construction."""
        self._validate_nodes()
        self._validate_distance()
        self._validate_capacity()
        self._validate_traffic_factors()
        logger.debug(
            "Edge created: %s -> %s (%.2f km, %s)",
            self.from_node,
            self.to_node,
            self.distance,
            self.road_condition.value,
        )

    def _validate_nodes(self) -> None:
        for attr, value in (("from_node", self.from_node), ("to_node", self.to_node)):
            if not isinstance(value, str) or not value.strip():
                raise EdgeValidationError(
                    f"Edge '{attr}' must be a non-empty string; got {value!r}."
                )
        if self.from_node == self.to_node:
            raise EdgeValidationError(
                f"Self-loops are not allowed (from_node == to_node == {self.from_node!r})."
            )

    def _validate_distance(self) -> None:
        if not isinstance(self.distance, (int, float)) or self.distance <= _MIN_DISTANCE:
            raise EdgeValidationError(
                f"Edge 'distance' must be a positive number; got {self.distance!r}."
            )

    def _validate_capacity(self) -> None:
        if not isinstance(self.base_capacity, int) or self.base_capacity < _MIN_CAPACITY:
            raise EdgeValidationError(
                f"Edge 'base_capacity' must be a non-negative integer; "
                f"got {self.base_capacity!r}."
            )

    def _validate_traffic_factors(self) -> None:
        for tod, factor in self.traffic_factors.items():
            if not isinstance(factor, (int, float)) or factor <= _MIN_TRAFFIC_FACTOR:
                raise EdgeValidationError(
                    f"traffic_factor for {tod!r} must be a positive number; "
                    f"got {factor!r}."
                )

    # ------------------------------------------------------------------
    # Dynamic weight calculation
    # ------------------------------------------------------------------

    def get_weight(self, time_of_day: TimeOfDay) -> float:
        """Return the effective travel cost for this edge at a given time.

        The weight is defined as::

            weight = distance × traffic_factor(time) × condition_penalty

        Results are memoised per ``time_of_day`` and returned from cache on
        subsequent calls.  Call :meth:`invalidate_cache` after updating
        ``traffic_factors`` or ``road_condition`` to discard stale values.

        Args:
            time_of_day: The :class:`~utils.enums.TimeOfDay` period to
                         evaluate the weight for.

        Returns:
            float: Positive effective travel cost.
        """
        if time_of_day in self._weight_cache:
            return self._weight_cache[time_of_day]

        traffic_factor: float = self.traffic_factors.get(time_of_day, 1.0)
        condition_penalty: float = ROAD_CONDITION_PENALTIES[self.road_condition]
        weight: float = self.distance * traffic_factor * condition_penalty

        self._weight_cache[time_of_day] = weight
        logger.debug(
            "Weight(%s->%s, %s) = distance=%.2f × traffic=%.2f × condition=%.2f → %.4f",
            self.from_node,
            self.to_node,
            time_of_day.value,
            self.distance,
            traffic_factor,
            condition_penalty,
            weight,
        )
        return weight

    def invalidate_cache(self) -> None:
        """Discard all cached weight values.

        Must be called after :attr:`traffic_factors` or :attr:`road_condition`
        are mutated so that subsequent :meth:`get_weight` calls recompute
        fresh values.
        """
        self._weight_cache.clear()
        logger.debug(
            "Weight cache invalidated for edge %s -> %s.", self.from_node, self.to_node
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Edge(from={self.from_node!r}, to={self.to_node!r}, "
            f"distance={self.distance:.2f} km, "
            f"condition={self.road_condition.value!r}, "
            f"bidirectional={self.is_bidirectional}, "
            f"proposed={self.is_proposed})"
        )

    def __hash__(self) -> int:
        return hash((self.from_node, self.to_node))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self.from_node == other.from_node and self.to_node == other.to_node
