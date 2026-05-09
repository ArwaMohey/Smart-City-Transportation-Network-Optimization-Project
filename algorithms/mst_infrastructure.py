"""MST Infrastructure Module for the Smart City Transportation Network System.

Implements **Kruskal's algorithm** with a custom composite edge weight to find
the optimal Minimum Spanning Tree (MST) for the city's road network.

Composite weight formula
------------------------
Each edge is scored with::

    composite = (dist_norm + COST_COEFF × cost_norm)
                × population_factor
                × facility_multiplier

Where:

* ``dist_norm``          – distance / max_distance  ∈ (0, 1]
* ``cost_norm``          – construction_cost / max_cost  ∈ [0, 1]
  (0 for already-built roads)
* ``population_factor``  – 1 / (1 + avg_population / max_population)  ∈ (0, 1]
  (inverse relationship: higher-population connections get *lower* weight and
  are therefore preferred by Kruskal)
* ``facility_multiplier``– 0.3 when either endpoint is a hospital or other
  important facility; 1.0 otherwise (hospitals are strongly prioritised)

Hospital / facility guarantee
------------------------------
After the standard Kruskal pass the algorithm verifies that every hospital
and important facility is reachable inside the MST.  Any isolated facility
node has its cheapest connecting edge force-added.

Usage::

    from data.data_loader import DataLoader
    from algorithms.mst_infrastructure import MSTOptimizer

    loader = DataLoader("data/sample_data", include_proposed=True)
    graph  = loader.load()

    optimizer  = MSTOptimizer(graph, data_dir="data/sample_data")
    mst_edges  = optimizer.run()
    analysis   = optimizer.get_cost_analysis()
    optimizer.visualize()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from core.graph import Graph
from models.edge import Edge
from models.node import Node
from utils.enums import NodeType

logger = logging.getLogger("smart_city.algorithms.mst_infrastructure")

# ── Priority constants ────────────────────────────────────────────────────────

# Multiplier applied to the composite weight of edges that touch a hospital or
# other important facility.  Values below 1.0 reduce the weight and cause
# Kruskal's to select these edges earlier (higher priority).
_CRITICAL_FACILITY_MULTIPLIER: float = 0.3

# Coefficient for the normalised construction-cost term in the composite weight.
_COST_WEIGHT_COEFFICIENT: float = 0.5

# Node types treated as "important facilities" for priority and connectivity.
_IMPORTANT_TYPES: frozenset[NodeType] = frozenset(
    {
        NodeType.HOSPITAL,
        NodeType.FIRE_STATION,
        NodeType.POLICE,
        NodeType.TRANSIT_HUB,
        NodeType.AIRPORT,
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# MSTOptimizer
# ─────────────────────────────────────────────────────────────────────────────


class MSTOptimizer:
    """Minimum Spanning Tree optimizer for the city infrastructure network.

    Applies Kruskal's algorithm with a custom composite edge weight that
    accounts for distance, construction cost, population-density priority,
    and critical-facility priority.

    Args:
        graph:    Fully loaded :class:`~core.graph.Graph` instance.
        data_dir: Path to the directory containing ``edges.json``, used to
                  read ``construction_cost_million_egp`` for proposed roads.
                  Defaults to ``"data/sample_data"``.

    Attributes:
        mst_edges (List[Edge]):       Edges selected for the MST.
                                      Populated only after :meth:`run`.
        total_distance_km (float):    Sum of MST edge distances (km).
        total_construction_cost (float): Sum of proposed-road costs (M EGP).

    Example::

        optimizer = MSTOptimizer(graph, data_dir="data/sample_data")
        optimizer.run()
        optimizer.visualize()
        analysis = optimizer.get_cost_analysis()
    """

    def __init__(
        self,
        graph: Graph,
        data_dir: Optional[Path | str] = None,
    ) -> None:
        self._graph: Graph = graph
        self._data_dir: Path = (
            Path(data_dir) if data_dir is not None else Path("data/sample_data")
        )
        # (from_node, to_node) → construction cost in million EGP.
        # Both orderings are stored so look-ups are direction-agnostic.
        self._construction_costs: Dict[Tuple[str, str], float] = {}

        self.mst_edges: List[Edge] = []
        self.total_distance_km: float = 0.0
        self.total_construction_cost: float = 0.0
        self._ran: bool = False

        self._load_construction_costs()

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> List[Edge]:
        """Execute Kruskal's MST algorithm on the loaded graph.

        Steps:

        1. Compute a composite weight for every edge (see :meth:`_composite_weight`).
        2. Sort all edges by composite weight (ascending = preferred first).
        3. Use a Union-Find structure to greedily add non-cycle-forming edges.
        4. Validate that every hospital / important-facility node is covered;
           force the cheapest connecting edge into the MST for any that are not.

        Returns:
            List of :class:`~models.edge.Edge` objects forming the MST.
        """
        nodes: List[Node] = self._graph.get_all_nodes()
        edges: List[Edge] = self._graph.get_all_edges()

        if not nodes:
            logger.warning("Graph has no nodes; MST is empty.")
            self.mst_edges = []
            self._ran = True
            return self.mst_edges

        # Precompute normalisation constants from actual data.
        max_distance: float = max((e.distance for e in edges), default=1.0)
        max_cost: float = max(self._construction_costs.values(), default=1.0)
        max_population: int = max((n.population for n in nodes), default=1)

        # Sort edges by composite weight (ascending → cheapest/most important first).
        sorted_edges: List[Edge] = sorted(
            edges,
            key=lambda e: self._composite_weight(
                e, max_distance, max_cost, max_population
            ),
        )

        # Kruskal's algorithm using Union-Find.
        parent: Dict[str, str] = {n.id: n.id for n in nodes}
        rank: Dict[str, int] = {n.id: 0 for n in nodes}
        mst: List[Edge] = []

        for edge in sorted_edges:
            u_root = self._find(parent, edge.from_node)
            v_root = self._find(parent, edge.to_node)
            if u_root != v_root:
                self._union(parent, rank, u_root, v_root)
                mst.append(edge)
                if len(mst) == len(nodes) - 1:
                    break  # MST complete (graph is connected).

        # Guarantee hospitals and important facilities are part of the MST.
        mst = self._ensure_facility_connectivity(
            mst, edges, parent, rank, max_distance, max_cost, max_population
        )

        self.mst_edges = mst
        self.total_distance_km = sum(e.distance for e in mst)
        self.total_construction_cost = sum(
            self._edge_construction_cost(e) for e in mst
        )

        self._ran = True
        logger.info(
            "MST computed: %d edge(s), total distance=%.2f km, "
            "total construction cost=%.1f M EGP.",
            len(self.mst_edges),
            self.total_distance_km,
            self.total_construction_cost,
        )
        return self.mst_edges

    def get_cost_analysis(self) -> Dict[str, Any]:
        """Return a detailed cost and connectivity breakdown of the MST.

        Must be called after :meth:`run`.

        Returns:
            Dictionary containing:

            - ``mst_edge_count`` – number of edges in the MST.
            - ``total_distance_km`` – combined road length (km).
            - ``total_construction_cost_million_egp`` – sum of proposed-road
              costs (M EGP).
            - ``existing_road_count`` – already-built roads in the MST.
            - ``proposed_road_count`` – proposed roads selected for the MST.
            - ``hospital_nodes_covered`` – hospital node IDs present in the MST.
            - ``critical_facility_nodes_covered`` – all important-facility IDs
              present in the MST.
            - ``edges`` – per-edge detail list (from/to, distance, condition,
              is_proposed, construction cost).

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        if not self._ran:
            raise RuntimeError("Call run() before get_cost_analysis().")

        mst_node_ids: Set[str] = self._mst_node_ids()

        hospitals = [
            n for n in self._graph.get_hospitals() if n.id in mst_node_ids
        ]
        critical = [
            n
            for n in self._graph.get_all_nodes()
            if n.node_type in _IMPORTANT_TYPES and n.id in mst_node_ids
        ]

        edge_details = [
            {
                "from_node": e.from_node,
                "to_node": e.to_node,
                "distance_km": e.distance,
                "road_condition": e.road_condition.value,
                "is_proposed": e.is_proposed,
                "construction_cost_million_egp": self._edge_construction_cost(e),
            }
            for e in self.mst_edges
        ]

        return {
            "mst_edge_count": len(self.mst_edges),
            "total_distance_km": round(self.total_distance_km, 3),
            "total_construction_cost_million_egp": round(
                self.total_construction_cost, 3
            ),
            "existing_road_count": sum(
                1 for e in self.mst_edges if not e.is_proposed
            ),
            "proposed_road_count": sum(
                1 for e in self.mst_edges if e.is_proposed
            ),
            "hospital_nodes_covered": [n.id for n in hospitals],
            "critical_facility_nodes_covered": [n.id for n in critical],
            "edges": edge_details,
        }

    def visualize(self) -> None:
        """Log a human-readable representation of the MST to the application logger.

        Outputs:

        * Overall statistics (edge count, distance, cost).
        * Every MST edge with endpoints, distance, road condition, and (for
          proposed roads) construction cost.
        * An adjacency map showing which nodes are connected in the MST.
        * Confirmed hospital and critical-facility coverage.

        Must be called after :meth:`run`.

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        if not self._ran:
            raise RuntimeError("Call run() before visualize().")

        analysis = self.get_cost_analysis()

        logger.info("=" * 68)
        logger.info("MST Infrastructure Network – Visualization")
        logger.info("=" * 68)
        logger.info("  Edges in MST            : %d", analysis["mst_edge_count"])
        logger.info(
            "  Total road distance     : %.2f km", analysis["total_distance_km"]
        )
        logger.info(
            "  Total construction cost : %.1f M EGP",
            analysis["total_construction_cost_million_egp"],
        )
        logger.info("  Existing roads selected : %d", analysis["existing_road_count"])
        logger.info("  Proposed roads selected : %d", analysis["proposed_road_count"])

        logger.info("-" * 68)
        logger.info("  MST Edges (sorted by distance):")
        for detail in sorted(analysis["edges"], key=lambda d: d["distance_km"]):
            proposed_tag = "[PROPOSED]" if detail["is_proposed"] else "[EXISTING]"
            cost_str = (
                f"  cost={detail['construction_cost_million_egp']:.0f}M EGP"
                if detail["is_proposed"]
                else ""
            )
            from_label = self._node_label(detail["from_node"])
            to_label = self._node_label(detail["to_node"])
            logger.info(
                "  %s %-26s → %-26s  dist=%5.1f km  cond=%-9s%s",
                proposed_tag,
                from_label,
                to_label,
                detail["distance_km"],
                detail["road_condition"],
                cost_str,
            )

        logger.info("-" * 68)
        logger.info("  MST Adjacency (connectivity map):")
        adjacency: Dict[str, List[str]] = {}
        for e in self.mst_edges:
            adjacency.setdefault(e.from_node, []).append(e.to_node)
            adjacency.setdefault(e.to_node, []).append(e.from_node)
        for node_id in sorted(adjacency):
            label = self._node_label(node_id)
            neighbours = ", ".join(sorted(adjacency[node_id]))
            logger.info("    %-28s ↔  %s", label, neighbours)

        logger.info("-" * 68)
        logger.info(
            "  Hospitals confirmed in MST (%d): %s",
            len(analysis["hospital_nodes_covered"]),
            ", ".join(analysis["hospital_nodes_covered"]) or "none",
        )
        logger.info(
            "  Critical facilities in MST (%d): %s",
            len(analysis["critical_facility_nodes_covered"]),
            ", ".join(analysis["critical_facility_nodes_covered"]) or "none",
        )
        logger.info("=" * 68)

    # ──────────────────────────────────────────────────────────────────────
    # Weight computation
    # ──────────────────────────────────────────────────────────────────────

    def _composite_weight(
        self,
        edge: Edge,
        max_distance: float,
        max_cost: float,
        max_population: int,
    ) -> float:
        """Compute the composite MST weight for *edge*.

        The formula combines three factors:

        1. **Distance** (normalised to [0, 1]).
        2. **Construction cost** (normalised to [0, 1]; 0 for existing roads).
        3. **Population priority** (inverse: higher population → lower weight).
        4. **Facility priority** (0.3× multiplier for critical facilities).

        Args:
            edge:            The edge to score.
            max_distance:    Largest distance across all edges (denominator).
            max_cost:        Largest construction cost across proposed roads.
            max_population:  Largest node population in the graph.

        Returns:
            float: A positive composite weight (smaller = preferred by Kruskal).
        """
        # ── Distance component (normalised) ───────────────────────────────
        dist_norm: float = edge.distance / max(max_distance, 1e-9)

        # ── Construction-cost component (normalised; 0 for existing roads) ─
        cost: float = self._edge_construction_cost(edge)
        cost_norm: float = (
            cost / max(max_cost, 1e-9) if edge.is_proposed else 0.0
        )

        # ── Population factor (inverse relationship) ──────────────────────
        try:
            node_u: Node = self._graph.get_node(edge.from_node)
            node_v: Node = self._graph.get_node(edge.to_node)
            avg_pop: float = (node_u.population + node_v.population) / 2.0
            is_critical: bool = (
                node_u.node_type in _IMPORTANT_TYPES
                or node_v.node_type in _IMPORTANT_TYPES
            )
        except Exception:
            avg_pop = 0.0
            is_critical = False

        # Higher average population → lower pop_factor → lower composite weight.
        pop_factor: float = 1.0 / (1.0 + avg_pop / max(max_population, 1))

        # ── Facility priority multiplier ──────────────────────────────────
        facility_multiplier: float = (
            _CRITICAL_FACILITY_MULTIPLIER if is_critical else 1.0
        )

        return (
            (dist_norm + _COST_WEIGHT_COEFFICIENT * cost_norm)
            * pop_factor
            * facility_multiplier
        )

    # ──────────────────────────────────────────────────────────────────────
    # Union-Find helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _find(parent: Dict[str, str], x: str) -> str:
        """Return the representative (root) of *x*'s component.

        Uses path-halving to keep the tree flat.
        """
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    @staticmethod
    def _union(
        parent: Dict[str, str], rank: Dict[str, int], x: str, y: str
    ) -> None:
        """Merge the components of *x* and *y* (union by rank)."""
        if rank[x] < rank[y]:
            x, y = y, x
        parent[y] = x
        if rank[x] == rank[y]:
            rank[x] += 1

    # ──────────────────────────────────────────────────────────────────────
    # Hospital / facility connectivity guarantee
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_facility_connectivity(
        self,
        mst: List[Edge],
        all_edges: List[Edge],
        parent: Dict[str, str],
        rank: Dict[str, int],
        max_distance: float,
        max_cost: float,
        max_population: int,
    ) -> List[Edge]:
        """Guarantee every hospital and important-facility node is in the MST.

        Called after the standard Kruskal pass.  If the graph is disconnected
        some facility nodes might not have been reached.  This method finds the
        cheapest connecting edge for each such isolated node and forces it into
        the MST, logging each forced insertion at INFO level.

        Args:
            mst:            Current MST edge list (modified in place if needed).
            all_edges:      All edges in the graph.
            parent, rank:   Union-Find structures from the Kruskal pass.
            max_distance, max_cost, max_population: Normalisation constants.

        Returns:
            The (potentially extended) MST edge list.
        """
        mst_node_ids: Set[str] = self._mst_node_ids_from(mst)

        important_nodes: List[Node] = [
            n
            for n in self._graph.get_all_nodes()
            if n.node_type in _IMPORTANT_TYPES
        ]

        for imp_node in important_nodes:
            if imp_node.id in mst_node_ids:
                continue  # already covered

            # Candidate edges that touch this node.
            candidates = [
                e
                for e in all_edges
                if e.from_node == imp_node.id or e.to_node == imp_node.id
            ]
            candidates.sort(
                key=lambda e: self._composite_weight(
                    e, max_distance, max_cost, max_population
                )
            )

            for edge in candidates:
                u_root = self._find(parent, edge.from_node)
                v_root = self._find(parent, edge.to_node)
                if u_root != v_root:
                    self._union(parent, rank, u_root, v_root)
                    mst.append(edge)
                    mst_node_ids.add(edge.from_node)
                    mst_node_ids.add(edge.to_node)
                    logger.info(
                        "Forced facility edge into MST: %s → %s "
                        "(ensuring %s (%s) is connected).",
                        edge.from_node,
                        edge.to_node,
                        imp_node.id,
                        imp_node.name,
                    )
                    break

        return mst

    # ──────────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────────

    def _load_construction_costs(self) -> None:
        """Parse ``edges.json`` and populate the construction-cost lookup table.

        Reads ``construction_cost_million_egp`` from each proposed-road record.
        Both ``(from, to)`` and ``(to, from)`` orderings are stored so that
        look-ups in :meth:`_edge_construction_cost` are direction-agnostic.
        """
        edges_file: Path = self._data_dir / "edges.json"
        if not edges_file.is_file():
            logger.warning(
                "edges.json not found at %s; construction costs will default to 0.",
                edges_file,
            )
            return

        try:
            with edges_file.open(encoding="utf-8") as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not read edges.json for construction costs: %s", exc
            )
            return

        count = 0
        for record in raw.get("proposed_roads", []):
            from_id = record.get("from_node")
            to_id = record.get("to_node")
            cost = record.get("construction_cost_million_egp")
            if from_id and to_id and cost is not None:
                self._construction_costs[(from_id, to_id)] = float(cost)
                self._construction_costs[(to_id, from_id)] = float(cost)
                count += 1

        logger.info(
            "Construction costs loaded: %d proposed road(s) with cost data.", count
        )

    # ──────────────────────────────────────────────────────────────────────
    # Utility helpers
    # ──────────────────────────────────────────────────────────────────────

    def _edge_construction_cost(self, edge: Edge) -> float:
        """Return the construction cost (M EGP) for *edge*, or 0 if none."""
        return self._construction_costs.get(
            (edge.from_node, edge.to_node),
            self._construction_costs.get((edge.to_node, edge.from_node), 0.0),
        )

    def _mst_node_ids(self) -> Set[str]:
        """Return the set of node IDs present in the current MST."""
        return self._mst_node_ids_from(self.mst_edges)

    @staticmethod
    def _mst_node_ids_from(mst: List[Edge]) -> Set[str]:
        """Return the set of node IDs reachable from a list of edges."""
        ids: Set[str] = set()
        for e in mst:
            ids.add(e.from_node)
            ids.add(e.to_node)
        return ids

    def _node_label(self, node_id: str) -> str:
        """Return a display string ``'ID (Name)'`` for *node_id*."""
        try:
            node = self._graph.get_node(node_id)
            return f"{node_id} ({node.name})"
        except Exception:
            return node_id

    def __repr__(self) -> str:
        status = (
            f"mst_edges={len(self.mst_edges)}" if self._ran else "not yet run"
        )
        return f"MSTOptimizer(graph={self._graph!r}, {status})"
