"""Public Transit Optimization Module (Dynamic Programming).

Optimizes bus allocation across routes to maximize a utility/coverage score
based on daily demand and route length, with diminishing returns per extra bus.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from data.data_loader import DataLoader

logger = logging.getLogger("smart_city.algorithms.dp_transit_optimization")


@dataclass(frozen=True)
class _RouteProfile:
    route_id: str
    name: str
    stops_count: int
    daily_passengers: int
    current_buses: int


class TransitOptimizer:
    """Dynamic-programming optimizer for city bus fleet allocation."""

    def __init__(self, transport_data: Dict[str, Any]) -> None:
        bus_routes = transport_data.get("bus_routes", [])
        if not isinstance(bus_routes, list) or not bus_routes:
            raise ValueError("transport_data must include a non-empty 'bus_routes' list.")

        self._transport_data = transport_data
        self._routes: List[_RouteProfile] = [
            _RouteProfile(
                route_id=str(route["route_id"]),
                name=str(route.get("name", route["route_id"])),
                stops_count=max(1, len(route.get("stops", []))),
                daily_passengers=max(0, int(route.get("daily_passengers", 0))),
                current_buses=max(0, int(route.get("buses_assigned", 0))),
            )
            for route in bus_routes
        ]

    @classmethod
    def from_data_dir(
        cls,
        data_dir: Path | str = Path("data/sample_data"),
        *,
        include_proposed: bool = True,
    ) -> "TransitOptimizer":
        """Create optimizer by loading transport metadata from a raw data directory."""
        loader = DataLoader(data_dir=Path(data_dir), include_proposed=include_proposed)
        loader.load()
        return cls(loader.get_public_transport())

    def optimize_bus_allocation(self, extra_buses: int = 0) -> Dict[str, Any]:
        """Return globally optimal bus allocation using bottom-up DP."""
        if extra_buses < 0:
            raise ValueError("extra_buses must be non-negative.")

        route_count = len(self._routes)
        old_total_fleet = sum(route.current_buses for route in self._routes)
        total_bus_fleet = old_total_fleet + extra_buses
        min_bus_per_route = 1
        min_required = route_count * min_bus_per_route
        if total_bus_fleet < min_required:
            raise ValueError("Total fleet is too small to keep all routes active.")

        allocatable = total_bus_fleet - min_required
        utility_lookup = [
            [self._route_utility(route, min_bus_per_route + x) for x in range(allocatable + 1)]
            for route in self._routes
        ]

        dp = [[0.0] * (allocatable + 1) for _ in range(route_count + 1)]
        decision = [[0] * (allocatable + 1) for _ in range(route_count + 1)]

        for i in range(1, route_count + 1):
            for b in range(allocatable + 1):
                best_value = float("-inf")
                best_split = 0
                for x in range(b + 1):
                    candidate = dp[i - 1][b - x] + utility_lookup[i - 1][x]
                    if candidate > best_value:
                        best_value = candidate
                        best_split = x
                dp[i][b] = best_value
                decision[i][b] = best_split

        optimized_allocation: Dict[str, int] = {}
        remaining = allocatable
        for i in range(route_count, 0, -1):
            extra_for_route = decision[i][remaining]
            route = self._routes[i - 1]
            buses = min_bus_per_route + extra_for_route
            optimized_allocation[route.route_id] = buses
            remaining -= extra_for_route

        old_allocation = {route.route_id: route.current_buses for route in self._routes}
        old_utility = self._total_utility(old_allocation)
        new_utility = self._total_utility(optimized_allocation)
        improvement = new_utility - old_utility
        improvement_percent = (improvement / old_utility * 100.0) if old_utility > 0 else None

        route_level_analysis = []
        for route in self._routes:
            before = old_allocation[route.route_id]
            after = optimized_allocation[route.route_id]
            route_level_analysis.append(
                {
                    "route_id": route.route_id,
                    "route_name": route.name,
                    "daily_passengers": route.daily_passengers,
                    "stops_count": route.stops_count,
                    "old_buses": before,
                    "new_buses": after,
                    "delta_buses": after - before,
                    "old_utility": round(self._route_utility(route, before), 3),
                    "new_utility": round(self._route_utility(route, after), 3),
                }
            )
        route_level_analysis.sort(key=lambda x: x["route_id"])

        return {
            "total_bus_fleet": total_bus_fleet,
            "extra_buses": extra_buses,
            "old_allocation": old_allocation,
            "optimized_allocation": dict(sorted(optimized_allocation.items())),
            "old_utility_score": round(old_utility, 3),
            "new_utility_score": round(new_utility, 3),
            "utility_improvement": round(improvement, 3),
            "improvement_percent": None if improvement_percent is None else round(improvement_percent, 2),
            "route_level_analysis": route_level_analysis,
            "metro_transfer_analysis": self._analyze_metro_transfers(),
        }

    def _route_weight(self, route: _RouteProfile) -> float:
        demand_weight = max(1.0, route.daily_passengers / 1000.0)
        length_weight = 1.0 + (route.stops_count - 1) * 0.12
        return demand_weight * length_weight

    def _route_utility(self, route: _RouteProfile, buses: int) -> float:
        if buses <= 0:
            return 0.0
        return self._route_weight(route) * math.log1p(buses)

    def _total_utility(self, allocation: Dict[str, int]) -> float:
        total = 0.0
        route_map = {route.route_id: route for route in self._routes}
        for route_id, buses in allocation.items():
            route = route_map[route_id]
            total += self._route_utility(route, buses)
        return total

    def _analyze_metro_transfers(self) -> List[Dict[str, Any]]:
        metro_lines = self._transport_data.get("metro_lines", [])
        metro_stops = {
            stop
            for line in metro_lines
            for stop in line.get("stops", [])
        }
        transfer_info = []
        for route in self._routes:
            raw_route = next(
                (r for r in self._transport_data.get("bus_routes", []) if r.get("route_id") == route.route_id),
                {},
            )
            stops = raw_route.get("stops", [])
            transfer_points = [stop for stop in stops if stop in metro_stops]
            transfer_info.append(
                {
                    "route_id": route.route_id,
                    "transfer_points": transfer_points,
                    "transfer_count": len(transfer_points),
                }
            )
        return sorted(transfer_info, key=lambda x: x["route_id"])
