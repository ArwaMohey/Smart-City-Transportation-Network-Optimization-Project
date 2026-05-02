"""Core Graph engine for the Smart City Transportation Network System.

The :class:`Graph` class is the central data structure used by every
algorithm team (Dijkstra, A*, MST, Emergency Routing, etc.).  It exposes
a clean, stable public API so that consuming code never needs to touch
internal storage details.

Design decisions
----------------
* **Adjacency-list** representation – ``O(1)`` neighbour look-up, compact
  storage for sparse city networks.
* **Directed / undirected** – controlled by the ``directed`` flag at
  construction time.  Undirected graphs register both (u→v) and (v→u) in
  the adjacency structure but share the same :class:`~models.edge.Edge`
  object to avoid duplication.
* **No tight coupling** to validation – graph validation is delegated to
  :class:`~utils.validators.GraphValidator` and invoked lazily via
  :meth:`validate`, keeping the graph class focused on structural
  operations only.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set

from models.edge import Edge
from models.node import Node
from utils.enums import NodeType, TimeOfDay

logger = logging.getLogger("smart_city.core.graph")


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────


class GraphError(Exception):
    """Base exception for graph structural errors."""


class NodeNotFoundError(GraphError):
    """Raised when a referenced node does not exist in the graph."""


class DuplicateNodeError(GraphError):
    """Raised when a node with the same ``id`` is added a second time."""


class DuplicateEdgeError(GraphError):
    """Raised when an edge with the same ``(from_node, to_node)`` already exists."""


# ─────────────────────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────────────────────


class Graph:
    """Adjacency-list graph modelling a smart city's transportation network.

    Args:
        directed: When *True* the graph is directed; edges are only
                  traversable from ``from_node`` to ``to_node``.
                  When *False* (default) the graph is undirected; both
                  directions are registered automatically.

    Attributes:
        directed (bool): Read-only flag indicating the graph's directionality.

    Example::

        from core.graph import Graph
        from models.node import Node
        from models.edge import Edge
        from utils.enums import NodeType, RoadCondition, TimeOfDay

        g = Graph()
        g.add_node(Node("N01", "Downtown", NodeType.NEIGHBORHOOD, 50000, 500.0, 500.0))
        g.add_node(Node("N02", "Riverside", NodeType.NEIGHBORHOOD, 35000, 300.0, 400.0))
        g.add_edge(Edge(
            from_node="N01", to_node="N02",
            distance=2.5, base_capacity=1500,
            road_condition=RoadCondition.GOOD,
            traffic_factors={TimeOfDay.MORNING: 1.6, TimeOfDay.AFTERNOON: 1.2,
                             TimeOfDay.EVENING: 1.5, TimeOfDay.NIGHT: 0.8},
        ))
        weight = g.get_dynamic_edge_weight("N01", "N02", TimeOfDay.MORNING)
    """

    def __init__(self, directed: bool = False) -> None:
        self._directed: bool = directed
        self._nodes: Dict[str, Node] = {}
        # adjacency[from_id][to_id] = Edge
        self._adjacency: Dict[str, Dict[str, Edge]] = defaultdict(dict)
        logger.info("Graph initialised (%s).", "directed" if directed else "undirected")

    # ──────────────────────────────────────────────────────────────────────
    # Node operations
    # ──────────────────────────────────────────────────────────────────────

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Args:
            node: :class:`~models.node.Node` instance to register.

        Raises:
            DuplicateNodeError: If a node with the same ``id`` already exists.
        """
        if node.id in self._nodes:
            raise DuplicateNodeError(
                f"Node id={node.id!r} already exists in the graph."
            )
        self._nodes[node.id] = node
        # Ensure adjacency entry exists even for initially-isolated nodes.
        if node.id not in self._adjacency:
            self._adjacency[node.id] = {}
        logger.info("Node added: %s (%s | %s).", node.id, node.name, node.node_type.value)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all edges that reference it.

        Args:
            node_id: Identifier of the node to remove.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        self._require_node(node_id)
        del self._nodes[node_id]
        del self._adjacency[node_id]
        # Remove incoming edges from all other adjacency lists.
        for neighbours in self._adjacency.values():
            neighbours.pop(node_id, None)
        logger.info("Node removed: %s.", node_id)

    # ──────────────────────────────────────────────────────────────────────
    # Edge operations
    # ──────────────────────────────────────────────────────────────────────

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph.

        For undirected graphs both (u→v) and (v→u) are registered, sharing
        the same :class:`~models.edge.Edge` object.

        Args:
            edge: :class:`~models.edge.Edge` instance to register.

        Raises:
            NodeNotFoundError: If ``from_node`` or ``to_node`` does not exist.
            DuplicateEdgeError: If the edge ``(from_node, to_node)`` already exists.
        """
        self._require_node(edge.from_node)
        self._require_node(edge.to_node)

        if edge.to_node in self._adjacency[edge.from_node]:
            raise DuplicateEdgeError(
                f"Edge {edge.from_node!r} -> {edge.to_node!r} already exists."
            )

        self._adjacency[edge.from_node][edge.to_node] = edge
        logger.info(
            "Edge added: %s -> %s (%.2f km, %s).",
            edge.from_node,
            edge.to_node,
            edge.distance,
            edge.road_condition.value,
        )

        if not self._directed and edge.is_bidirectional:
            if edge.from_node not in self._adjacency[edge.to_node]:
                self._adjacency[edge.to_node][edge.from_node] = edge
                logger.debug(
                    "Reverse edge auto-registered: %s -> %s.", edge.to_node, edge.from_node
                )

    def remove_edge(self, from_node: str, to_node: str) -> None:
        """Remove the edge from *from_node* to *to_node*.

        For undirected graphs the reverse edge is also removed.

        Args:
            from_node: Source node identifier.
            to_node:   Destination node identifier.

        Raises:
            NodeNotFoundError: If either endpoint does not exist.
            GraphError:        If the edge does not exist.
        """
        self._require_node(from_node)
        self._require_node(to_node)

        if to_node not in self._adjacency.get(from_node, {}):
            raise GraphError(f"Edge {from_node!r} -> {to_node!r} does not exist.")

        del self._adjacency[from_node][to_node]
        logger.info("Edge removed: %s -> %s.", from_node, to_node)

        if not self._directed:
            self._adjacency[to_node].pop(from_node, None)

    # ──────────────────────────────────────────────────────────────────────
    # Public query API
    # ──────────────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Node:
        """Return the :class:`~models.node.Node` with the given identifier.

        Args:
            node_id: Unique node identifier.

        Returns:
            The :class:`~models.node.Node` instance.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        self._require_node(node_id)
        return self._nodes[node_id]

    def get_neighbors(self, node_id: str) -> List[Edge]:
        """Return all edges originating from *node_id*.

        Args:
            node_id: Source node identifier.

        Returns:
            List of :class:`~models.edge.Edge` objects reachable from this node.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        self._require_node(node_id)
        return list(self._adjacency[node_id].values())

    def get_all_nodes(self) -> List[Node]:
        """Return every node currently registered in the graph.

        Returns:
            List of :class:`~models.node.Node` objects.
        """
        return list(self._nodes.values())

    def get_all_edges(self) -> List[Edge]:
        """Return every unique edge currently registered in the graph.

        In undirected graphs each road segment is returned **once** even
        though both directions are stored in the adjacency structure.

        Returns:
            Deduplicated list of :class:`~models.edge.Edge` objects.
        """
        seen: Set[int] = set()
        edges: List[Edge] = []
        for neighbours in self._adjacency.values():
            for edge in neighbours.values():
                eid = id(edge)
                if eid not in seen:
                    seen.add(eid)
                    edges.append(edge)
        return edges

    def get_hospitals(self) -> List[Node]:
        """Return all nodes of type :attr:`~utils.enums.NodeType.HOSPITAL`.

        Returns:
            List of hospital :class:`~models.node.Node` objects.
        """
        return self.get_facilities_by_type(NodeType.HOSPITAL)

    def get_facilities_by_type(self, node_type: NodeType) -> List[Node]:
        """Return all nodes that match the given *node_type*.

        Args:
            node_type: The :class:`~utils.enums.NodeType` to filter by.

        Returns:
            List of matching :class:`~models.node.Node` objects.
        """
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def get_dynamic_edge_weight(self, u: str, v: str, time: TimeOfDay) -> float:
        """Return the dynamic travel cost for the edge (u → v) at *time*.

        Args:
            u:    Source node identifier.
            v:    Destination node identifier.
            time: :class:`~utils.enums.TimeOfDay` period.

        Returns:
            float: Effective edge weight (distance × traffic × condition).

        Raises:
            NodeNotFoundError: If either node does not exist.
            GraphError:        If no edge connects *u* to *v*.
        """
        self._require_node(u)
        self._require_node(v)

        edge: Optional[Edge] = self._adjacency[u].get(v)
        if edge is None:
            raise GraphError(f"No edge exists from {u!r} to {v!r}.")
        return edge.get_weight(time)

    def update_traffic_factor(
        self,
        from_node: str,
        to_node: str,
        time: TimeOfDay,
        factor: float,
    ) -> None:
        """Update the traffic congestion factor for an edge at a given time.

        Supports real-time traffic updates.  The weight cache of the
        affected edge is automatically invalidated.

        Args:
            from_node: Source node identifier.
            to_node:   Destination node identifier.
            time:      The :class:`~utils.enums.TimeOfDay` period to update.
            factor:    New congestion multiplier (must be > 0).

        Raises:
            NodeNotFoundError: If either node does not exist.
            GraphError:        If the edge does not exist.
            ValueError:        If *factor* is not positive.
        """
        if not isinstance(factor, (int, float)) or factor <= 0:
            raise ValueError(f"Traffic factor must be a positive number; got {factor!r}.")

        self._require_node(from_node)
        self._require_node(to_node)

        edge: Optional[Edge] = self._adjacency[from_node].get(to_node)
        if edge is None:
            raise GraphError(f"No edge exists from {from_node!r} to {to_node!r}.")

        edge.traffic_factors[time] = factor
        edge.invalidate_cache()
        logger.info(
            "Traffic factor updated: %s -> %s at %s = %.2f.",
            from_node,
            to_node,
            time.value,
            factor,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Validation entry-point
    # ──────────────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Validate the structural integrity of the graph.

        Delegates all checks to :class:`~utils.validators.GraphValidator`.
        All warnings are logged at ``WARNING`` level; errors at ``ERROR``
        level.

        Raises:
            GraphValidationError: If one or more critical structural errors
                                  are detected.
        """
        # Late import to avoid a circular dependency at module load time.
        from utils.validators import GraphValidator  # noqa: PLC0415

        GraphValidator(self).validate()

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _require_node(self, node_id: str) -> None:
        """Raise :class:`NodeNotFoundError` if *node_id* is not in the graph."""
        if node_id not in self._nodes:
            raise NodeNotFoundError(
                f"Node {node_id!r} does not exist in the graph."
            )

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def directed(self) -> bool:
        """*True* if the graph is directed, *False* if undirected."""
        return self._directed

    @property
    def node_count(self) -> int:
        """Total number of nodes registered in the graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Total number of unique edges registered in the graph."""
        return len(self.get_all_edges())

    def __repr__(self) -> str:
        return (
            f"Graph(nodes={self.node_count}, edges={self.edge_count}, "
            f"directed={self._directed})"
        )
