"""Routing & Traffic Module for Smart City Transportation Network.

Provides a Dijkstra-based router that computes congestion-aware shortest paths
using dynamic edge weights for the requested time of day, with optional road
closures for incident simulation.
"""

from __future__ import annotations

import heapq
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from core.graph import Graph
from models.edge import Edge
from utils.enums import TimeOfDay

logger = logging.getLogger("smart_city.algorithms.dijkstra_routing")

# Default multiplier used when a specific time bucket is missing on an edge.
DEFAULT_TRAFFIC_FACTOR: float = 1.0
# Traffic-factor thresholds used to classify congestion severity labels.
HEAVY_CONGESTION_THRESHOLD: float = 3.5
MODERATE_CONGESTION_THRESHOLD: float = 2.0
LIGHT_CONGESTION_THRESHOLD: float = 1.2


class TrafficRouter:
    """Dijkstra routing engine with time-dependent edge costs and closures."""

    def __init__(self, graph: Graph) -> None:
        self._graph = graph

    def find_best_route(
        self,
        source_id: str,
        dest_id: str,
        time_of_day: TimeOfDay | str,
        closed_roads: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Compute the best route from *source_id* to *dest_id*.

        Args:
            source_id: Source node id.
            dest_id: Destination node id.
            time_of_day: Time window enum or string (e.g., "morning", "Morning Peak").
            closed_roads: Optional list of closed road tuples ``(from_node, to_node)``.

        Returns:
            Structured dictionary with path, total travel cost, and turn-by-turn log.
        """
        tod = self._normalize_time_of_day(time_of_day)

        # Validate endpoints early using graph public API.
        self._graph.get_node(source_id)
        self._graph.get_node(dest_id)

        if source_id == dest_id:
            return {
                "found": True,
                "source": source_id,
                "destination": dest_id,
                "time_of_day": tod.value,
                "best_route": [source_id],
                "total_travel_cost": 0.0,
                "turn_by_turn": [],
                "routing_decisions": {
                    "algorithm": "dijkstra",
                    "closed_roads_applied": self._format_closed_roads(closed_roads),
                    "notes": ["Source and destination are the same node."],
                },
            }

        blocked_edges = self._build_blocked_edge_set(closed_roads)
        # Tracks only closures encountered during search (subset of blocked_edges).
        skipped_closed_edges: Set[Tuple[str, str]] = set()

        distances: Dict[str, float] = {source_id: 0.0}
        previous: Dict[str, str] = {}
        visited: Set[str] = set()
        heap: List[Tuple[float, str]] = [(0.0, source_id)]

        while heap:
            current_cost, current_node = heapq.heappop(heap)
            if current_node in visited:
                continue
            visited.add(current_node)

            if current_node == dest_id:
                break

            for edge in self._graph.get_neighbors(current_node):
                neighbor = self._neighbor_from_edge(current_node, edge)

                if (current_node, neighbor) in blocked_edges:
                    skipped_closed_edges.add((current_node, neighbor))
                    continue

                if neighbor in visited:
                    continue

                edge_cost = self._graph.get_dynamic_edge_weight(current_node, neighbor, tod)
                new_cost = current_cost + edge_cost

                if new_cost < distances.get(neighbor, float("inf")):
                    distances[neighbor] = new_cost
                    previous[neighbor] = current_node
                    heapq.heappush(heap, (new_cost, neighbor))

        if dest_id not in distances:
            return {
                "found": False,
                "source": source_id,
                "destination": dest_id,
                "time_of_day": tod.value,
                "best_route": [],
                "total_travel_cost": None,
                "turn_by_turn": [],
                "routing_decisions": {
                    "algorithm": "dijkstra",
                    "closed_roads_applied": sorted(skipped_closed_edges),
                    "notes": [
                        "No route available with current road closures and graph connectivity."
                    ],
                },
            }

        best_route = self._reconstruct_path(previous, source_id, dest_id)
        turn_by_turn = self._build_turn_by_turn(best_route, tod)
        total_cost = distances[dest_id]

        return {
            "found": True,
            "source": source_id,
            "destination": dest_id,
            "time_of_day": tod.value,
            "best_route": best_route,
            "total_travel_cost": round(total_cost, 4),
            "turn_by_turn": turn_by_turn,
            "routing_decisions": {
                "algorithm": "dijkstra",
                "closed_roads_applied": self._format_closed_roads(closed_roads),
                "closed_edges_skipped_during_search": sorted(skipped_closed_edges),
                "visited_node_count": len(visited),
                "total_steps": max(0, len(best_route) - 1),
            },
        }

    @staticmethod
    def _normalize_time_of_day(time_of_day: TimeOfDay | str) -> TimeOfDay:
        if isinstance(time_of_day, TimeOfDay):
            return time_of_day

        if not isinstance(time_of_day, str):
            raise ValueError("time_of_day must be a TimeOfDay value or string.")

        normalized = time_of_day.strip().lower().replace("_", " ")
        aliases = {
            "morning": TimeOfDay.MORNING,
            "morning peak": TimeOfDay.MORNING,
            "afternoon": TimeOfDay.AFTERNOON,
            "evening": TimeOfDay.EVENING,
            "evening peak": TimeOfDay.EVENING,
            "night": TimeOfDay.NIGHT,
        }
        tod = aliases.get(normalized)
        if tod is None:
            valid = ", ".join(t.value for t in TimeOfDay)
            raise ValueError(f"Unsupported time_of_day={time_of_day!r}. Valid: {valid}.")
        return tod

    def _build_turn_by_turn(
        self, route: List[str], time_of_day: TimeOfDay
    ) -> List[Dict[str, Any]]:
        turns: List[Dict[str, Any]] = []

        for step_number, (from_node, to_node) in enumerate(zip(route, route[1:]), start=1):
            edge = self._find_edge(from_node, to_node)
            if edge is None:
                continue

            dynamic_cost = self._graph.get_dynamic_edge_weight(from_node, to_node, time_of_day)
            traffic_factor = float(
                edge.traffic_factors.get(time_of_day, DEFAULT_TRAFFIC_FACTOR)
            )

            turns.append(
                {
                    "step": step_number,
                    "from": from_node,
                    "to": to_node,
                    "distance_km": round(edge.distance, 3),
                    "road_condition": edge.road_condition.value,
                    "traffic_factor": round(traffic_factor, 3),
                    "dynamic_cost": round(dynamic_cost, 4),
                    "congestion_level": self._congestion_label(traffic_factor),
                }
            )

        return turns

    def _build_blocked_edge_set(
        self, closed_roads: Optional[List[Tuple[str, str]]]
    ) -> Set[Tuple[str, str]]:
        blocked: Set[Tuple[str, str]] = set()
        if not closed_roads:
            return blocked

        for from_node, to_node in closed_roads:
            blocked.add((from_node, to_node))
            # For undirected graphs, a closure blocks the whole segment.
            if not self._graph.directed:
                blocked.add((to_node, from_node))
        return blocked

    @staticmethod
    def _reconstruct_path(
        previous: Dict[str, str], source_id: str, dest_id: str
    ) -> List[str]:
        path = [dest_id]
        current = dest_id

        while current != source_id:
            current = previous[current]
            path.append(current)

        path.reverse()
        return path

    @staticmethod
    def _neighbor_from_edge(current_node: str, edge: Edge) -> str:
        if edge.from_node == current_node:
            return edge.to_node
        return edge.from_node

    def _find_edge(self, from_node: str, to_node: str) -> Optional[Edge]:
        for edge in self._graph.get_neighbors(from_node):
            if self._neighbor_from_edge(from_node, edge) == to_node:
                return edge
        return None

    @staticmethod
    def _congestion_label(traffic_factor: float) -> str:
        if traffic_factor >= HEAVY_CONGESTION_THRESHOLD:
            return "heavy"
        if traffic_factor >= MODERATE_CONGESTION_THRESHOLD:
            return "moderate"
        if traffic_factor >= LIGHT_CONGESTION_THRESHOLD:
            return "light"
        return "free flow"

    @staticmethod
    def _format_closed_roads(
        closed_roads: Optional[List[Tuple[str, str]]]
    ) -> List[Tuple[str, str]]:
        return [] if not closed_roads else sorted({(u, v) for u, v in closed_roads})
