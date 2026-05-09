"""Emergency Routing Module (A*) for Smart City Transportation Network.

Provides an A*-based emergency router that prioritizes fast response by
dramatically reducing congestion impact while preserving distance and road
condition effects.
"""

from __future__ import annotations

import heapq
import logging
from typing import Any, Dict, List, Optional, Tuple

from algorithms.dijkstra_routing import TrafficRouter
from core.graph import Graph
from models.edge import Edge, ROAD_CONDITION_PENALTIES
from models.node import Node
from utils.enums import TimeOfDay

logger = logging.getLogger("smart_city.algorithms.a_star_emergency")


class EmergencyRouter:
    """A* routing engine specialized for emergency vehicle dispatch."""

    def __init__(
        self,
        graph: Graph,
        *,
        congestion_impact: float = 0.15,
        heuristic_mode: str = "euclidean",
    ) -> None:
        self._graph = graph
        self._congestion_impact = max(0.0, min(1.0, congestion_impact))
        self._heuristic_mode = heuristic_mode.strip().lower()
        if self._heuristic_mode not in {"euclidean", "manhattan"}:
            raise ValueError("heuristic_mode must be either 'euclidean' or 'manhattan'.")
        self._heuristic_scale = self._build_heuristic_scale()

    def find_emergency_route(
        self,
        source_id: str,
        dest_id: str,
        time_of_day: TimeOfDay | str,
    ) -> Dict[str, Any]:
        """Find a congestion-preempted emergency route using A* search."""
        tod = self._normalize_time_of_day(time_of_day)

        source_node = self._graph.get_node(source_id)
        dest_node = self._graph.get_node(dest_id)

        normal_router = TrafficRouter(self._graph)
        normal_result = normal_router.find_best_route(source_id, dest_id, tod)

        if source_id == dest_id:
            comparison = self._build_comparison(0.0, normal_result.get("total_travel_cost"))
            return {
                "found": True,
                "source": source_id,
                "destination": dest_id,
                "time_of_day": tod.value,
                "fast_emergency_route": [source_id],
                "response_time_cost": 0.0,
                "turn_by_turn": [],
                "comparison": comparison,
                "normal_dijkstra": normal_result,
                "routing_decisions": {
                    "algorithm": "a_star_emergency",
                    "heuristic_mode": self._heuristic_mode,
                    "notes": ["Source and destination are the same node."],
                },
            }

        open_heap: List[Tuple[float, str]] = []
        heapq.heappush(open_heap, (self._heuristic(source_node, dest_node), source_id))

        came_from: Dict[str, str] = {}
        g_score: Dict[str, float] = {source_id: 0.0}
        visited_count = 0

        while open_heap:
            _, current = heapq.heappop(open_heap)
            visited_count += 1

            if current == dest_id:
                break

            current_cost = g_score[current]
            for edge in self._graph.get_neighbors(current):
                neighbor = self._neighbor_from_edge(current, edge)
                emergency_cost = self._emergency_edge_weight(edge, tod)
                tentative_cost = current_cost + emergency_cost

                if tentative_cost < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_cost
                    estimated_total = tentative_cost + self._heuristic(
                        self._graph.get_node(neighbor), dest_node
                    )
                    heapq.heappush(open_heap, (estimated_total, neighbor))

        if dest_id not in g_score:
            return {
                "found": False,
                "source": source_id,
                "destination": dest_id,
                "time_of_day": tod.value,
                "fast_emergency_route": [],
                "response_time_cost": None,
                "turn_by_turn": [],
                "comparison": self._build_comparison(None, normal_result.get("total_travel_cost")),
                "normal_dijkstra": normal_result,
                "routing_decisions": {
                    "algorithm": "a_star_emergency",
                    "heuristic_mode": self._heuristic_mode,
                    "expanded_nodes": visited_count,
                    "notes": ["No route found with emergency-priority A* search."],
                },
            }

        route = self._reconstruct_path(came_from, source_id, dest_id)
        emergency_total = g_score[dest_id]

        return {
            "found": True,
            "source": source_id,
            "destination": dest_id,
            "time_of_day": tod.value,
            "fast_emergency_route": route,
            "response_time_cost": round(emergency_total, 4),
            "turn_by_turn": self._build_turn_by_turn(route, tod),
            "comparison": self._build_comparison(emergency_total, normal_result.get("total_travel_cost")),
            "normal_dijkstra": normal_result,
            "routing_decisions": {
                "algorithm": "a_star_emergency",
                "heuristic_mode": self._heuristic_mode,
                "congestion_impact": self._congestion_impact,
                "expanded_nodes": visited_count,
                "total_steps": max(0, len(route) - 1),
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

    def _emergency_edge_weight(self, edge: Edge, time_of_day: TimeOfDay) -> float:
        traffic_factor = edge.traffic_factors.get(time_of_day, 1.0)
        adjusted_traffic = 1.0 + (traffic_factor - 1.0) * self._congestion_impact
        condition_penalty = ROAD_CONDITION_PENALTIES[edge.road_condition]
        return edge.distance * adjusted_traffic * condition_penalty

    def _heuristic(self, current: Node, destination: Node) -> float:
        dx = abs(current.x_coordinate - destination.x_coordinate)
        dy = abs(current.y_coordinate - destination.y_coordinate)
        if self._heuristic_mode == "manhattan":
            base_distance = dx + dy
        else:
            base_distance = (dx * dx + dy * dy) ** 0.5
        return base_distance * self._heuristic_scale

    def _build_heuristic_scale(self) -> float:
        minimum_ratio: Optional[float] = None

        for edge in self._graph.get_all_edges():
            from_node = self._graph.get_node(edge.from_node)
            to_node = self._graph.get_node(edge.to_node)
            straight_distance = from_node.euclidean_distance_to(to_node)
            if straight_distance <= 0:
                continue

            condition_penalty = ROAD_CONDITION_PENALTIES[edge.road_condition]
            min_edge_cost = edge.distance * condition_penalty
            ratio = min_edge_cost / straight_distance
            if minimum_ratio is None or ratio < minimum_ratio:
                minimum_ratio = ratio

        if minimum_ratio is None:
            return 0.0
        return minimum_ratio

    def _build_turn_by_turn(self, route: List[str], time_of_day: TimeOfDay) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        for step_number, (from_node, to_node) in enumerate(zip(route, route[1:]), start=1):
            edge = self._find_edge(from_node, to_node)
            if edge is None:
                continue

            normal_cost = self._graph.get_dynamic_edge_weight(from_node, to_node, time_of_day)
            emergency_cost = self._emergency_edge_weight(edge, time_of_day)
            saved = normal_cost - emergency_cost

            steps.append(
                {
                    "step": step_number,
                    "from": from_node,
                    "to": to_node,
                    "distance": round(edge.distance, 3),
                    "normal_cost": round(normal_cost, 4),
                    "emergency_cost": round(emergency_cost, 4),
                    "saved_cost": round(saved, 4),
                }
            )
        return steps

    def _find_edge(self, from_node: str, to_node: str) -> Optional[Edge]:
        for edge in self._graph.get_neighbors(from_node):
            if self._neighbor_from_edge(from_node, edge) == to_node:
                return edge
        return None

    @staticmethod
    def _neighbor_from_edge(current_node: str, edge: Edge) -> str:
        if edge.from_node == current_node:
            return edge.to_node
        return edge.from_node

    @staticmethod
    def _reconstruct_path(previous: Dict[str, str], source_id: str, dest_id: str) -> List[str]:
        path = [dest_id]
        current = dest_id
        while current != source_id:
            current = previous[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _build_comparison(
        emergency_cost: Optional[float], normal_cost: Optional[float]
    ) -> Dict[str, Optional[float]]:
        if emergency_cost is None or normal_cost is None:
            return {
                "normal_dijkstra_cost": normal_cost,
                "emergency_a_star_cost": emergency_cost,
                "time_saved": None,
                "improvement_percent": None,
            }

        time_saved = normal_cost - emergency_cost
        improvement_percent = 0.0 if normal_cost == 0 else (time_saved / normal_cost) * 100.0

        return {
            "normal_dijkstra_cost": round(normal_cost, 4),
            "emergency_a_star_cost": round(emergency_cost, 4),
            "time_saved": round(time_saved, 4),
            "improvement_percent": round(improvement_percent, 2),
        }
