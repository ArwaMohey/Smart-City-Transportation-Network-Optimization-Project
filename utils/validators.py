"""Graph validation module for the Smart City Transportation Network System.

:class:`GraphValidator` performs a battery of structural integrity checks
on a :class:`~core.graph.Graph` instance.  It is intentionally decoupled
from the graph itself (Single Responsibility Principle) so that new
validation rules can be added without modifying ``Graph``.

Usage::

    from utils.validators import GraphValidator
    GraphValidator(graph).validate()

Or via the convenience method on the graph itself::

    graph.validate()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Set

if TYPE_CHECKING:
    # Imported only during static analysis to avoid circular imports.
    from core.graph import Graph

logger = logging.getLogger("smart_city.utils.validators")

# ── Coordinate bounds (must match models/node.py) ─────────────────────────
_COORD_MIN: float = 0.0
_COORD_MAX: float = 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Custom exception
# ─────────────────────────────────────────────────────────────────────────────


class GraphValidationError(Exception):
    """Raised when the graph contains one or more critical structural errors."""


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────


class GraphValidator:
    """Validates the structural integrity of a :class:`~core.graph.Graph`.

    Checks performed
    ----------------
    * Every edge references nodes that actually exist in the graph.
    * Node coordinates lie within the valid grid bounds.
    * No isolated nodes (nodes with no adjacent edges in either direction).

    Warnings vs errors
    ------------------
    * **Errors** (critical) – will cause :meth:`validate` to raise
      :class:`GraphValidationError` after processing all checks.
    * **Warnings** (non-critical) – logged but do not abort execution.

    Args:
        graph: The :class:`~core.graph.Graph` instance to validate.

    Example::

        validator = GraphValidator(graph)
        validator.validate()   # raises GraphValidationError on failure
    """

    def __init__(self, graph: "Graph") -> None:
        self._graph = graph
        self._errors: List[str] = []
        self._warnings: List[str] = []

    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Run all validation checks and report results via the logger.

        Raises:
            GraphValidationError: If any critical error is detected.
        """
        self._errors.clear()
        self._warnings.clear()

        logger.info(
            "Graph validation started: %d node(s), %d edge(s).",
            self._graph.node_count,
            self._graph.edge_count,
        )

        self._check_edge_references()
        self._check_node_coordinates()
        self._check_isolated_nodes()

        # ── Emit accumulated messages ──────────────────────────────────
        for msg in self._warnings:
            logger.warning("VALIDATION WARNING – %s", msg)

        for msg in self._errors:
            logger.error("VALIDATION ERROR – %s", msg)

        if self._errors:
            raise GraphValidationError(
                f"Graph validation failed with {len(self._errors)} error(s) "
                f"and {len(self._warnings)} warning(s). See logs for details."
            )

        logger.info(
            "Graph validation passed (%d warning(s)).", len(self._warnings)
        )

    # ──────────────────────────────────────────────────────────────────────
    # Individual checks
    # ──────────────────────────────────────────────────────────────────────

    def _check_edge_references(self) -> None:
        """Verify that both endpoints of every edge exist in the graph."""
        node_ids: Set[str] = {n.id for n in self._graph.get_all_nodes()}
        for edge in self._graph.get_all_edges():
            if edge.from_node not in node_ids:
                self._errors.append(
                    f"Edge references non-existent from_node={edge.from_node!r}."
                )
            if edge.to_node not in node_ids:
                self._errors.append(
                    f"Edge references non-existent to_node={edge.to_node!r}."
                )

    def _check_node_coordinates(self) -> None:
        """Verify that all node coordinates are within [_COORD_MIN, _COORD_MAX]."""
        for node in self._graph.get_all_nodes():
            for attr, value in (
                ("x_coordinate", node.x_coordinate),
                ("y_coordinate", node.y_coordinate),
            ):
                if not (_COORD_MIN <= value <= _COORD_MAX):
                    self._errors.append(
                        f"Node {node.id!r}: {attr}={value} is outside the valid "
                        f"range [{_COORD_MIN}, {_COORD_MAX}]."
                    )

    def _check_isolated_nodes(self) -> None:
        """Warn about nodes that have no outgoing *and* no incoming edges.

        In a directed graph a node may have only incoming edges and no outgoing
        ones (a pure sink), or only outgoing edges and no incoming ones (a pure
        source) – neither case is flagged as fully isolated.  Only nodes with
        zero edges in *both* directions receive a warning.

        Isolated nodes are not necessarily invalid (e.g. a proposed station
        that has no roads yet), but they are suspicious and should be flagged.
        """
        # Build the set of node ids that appear as *to_node* in at least one edge.
        nodes_with_incoming: Set[str] = {
            edge.to_node for edge in self._graph.get_all_edges()
        }
        if not self._graph.directed:
            # In an undirected graph every edge is traversable both ways, so
            # both endpoints always carry "incoming" coverage.
            nodes_with_incoming.update(
                edge.from_node for edge in self._graph.get_all_edges()
            )

        for node in self._graph.get_all_nodes():
            has_outgoing: bool = len(self._graph.get_neighbors(node.id)) > 0
            has_incoming: bool = node.id in nodes_with_incoming
            if not has_outgoing and not has_incoming:
                self._warnings.append(
                    f"Node {node.id!r} ({node.name!r}) is isolated "
                    f"(no outgoing and no incoming edges)."
                )
