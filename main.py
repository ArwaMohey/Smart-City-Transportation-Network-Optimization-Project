"""Smart City Transportation Network – Example Usage.

Demonstrates the complete load → validate → query workflow that algorithm
teams (Dijkstra, A*, MST, etc.) will use as their entry point.

Run from the project root::

    python main.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── Bootstrap logging before any other import ─────────────────────────────
from config.logging_config import setup_logging, get_logger

setup_logging(level=logging.INFO)
logger = get_logger("main")

# ── Ensure project root is on the path (for direct script execution) ──────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.data_loader import DataLoader
from utils.enums import NodeType, TimeOfDay
from algorithms.mst_infrastructure import MSTOptimizer
from algorithms.dijkstra_routing import TrafficRouter
from algorithms.a_star_emergency import EmergencyRouter
from algorithms.dp_transit_optimization import TransitOptimizer


def _format_cost(cost: float | None) -> str:
    """Format route cost safely for logging."""
    return "N/A" if cost is None else f"{cost:.4f}"


def run_mst_demo(graph, data_dir: Path) -> None:
    """Run the MST Infrastructure optimizer on *graph* and log the results.

    Args:
        graph:    The fully loaded and validated :class:`~core.graph.Graph`.
        data_dir: Path to the data directory (needed to resolve construction
                  costs for proposed roads).
    """
    logger.info("=" * 60)
    logger.info("Member 2 – MST Infrastructure Module")
    logger.info("=" * 60)

    optimizer = MSTOptimizer(graph, data_dir=data_dir)
    mst_edges = optimizer.run()

    logger.info("MST edge count : %d", len(mst_edges))
    logger.info(
        "Total distance : %.2f km", optimizer.total_distance_km
    )
    logger.info(
        "Construction cost (proposed roads): %.1f M EGP",
        optimizer.total_construction_cost,
    )

    analysis = optimizer.get_cost_analysis()
    logger.info("-" * 60)
    logger.info(
        "Existing roads in MST : %d", analysis["existing_road_count"]
    )
    logger.info(
        "Proposed roads in MST : %d", analysis["proposed_road_count"]
    )
    logger.info(
        "Hospitals covered     : %s",
        ", ".join(analysis["hospital_nodes_covered"]) or "none",
    )
    logger.info(
        "Critical facilities   : %s",
        ", ".join(analysis["critical_facility_nodes_covered"]) or "none",
    )

    # Full visual breakdown.
    optimizer.visualize()


def run_routing_demo(graph) -> None:
    """Run Member 3 routing scenarios using time-dependent Dijkstra."""
    logger.info("=" * 60)
    logger.info("Member 3 – Routing & Traffic Module (Dijkstra)")
    logger.info("=" * 60)

    router = TrafficRouter(graph)
    source, destination = "N01", "N13"

    # 1) Morning Peak scenario
    morning_result = router.find_best_route(
        source,
        destination,
        time_of_day=TimeOfDay.MORNING,
    )
    logger.info("-" * 60)
    logger.info(
        "Morning Peak route %s -> %s | cost=%.4f | path=%s",
        source,
        destination,
        morning_result["total_travel_cost"],
        " -> ".join(morning_result["best_route"]),
    )
    for step in morning_result["turn_by_turn"]:
        logger.info(
            "  Step %d: %s -> %s | dist=%.2f km | factor=%.3f | cost=%.4f | congestion=%s",
            step["step"],
            step["from"],
            step["to"],
            step["distance"],
            step["traffic_factor"],
            step["dynamic_cost"],
            step["congestion_level"],
        )

    # 2) Night scenario
    night_result = router.find_best_route(source, destination, time_of_day=TimeOfDay.NIGHT)
    logger.info("-" * 60)
    logger.info(
        "Night route %s -> %s | cost=%.4f | path=%s",
        source,
        destination,
        night_result["total_travel_cost"],
        " -> ".join(night_result["best_route"]),
    )
    for step in night_result["turn_by_turn"]:
        logger.info(
            "  Step %d: %s -> %s | dist=%.2f km | factor=%.3f | cost=%.4f | congestion=%s",
            step["step"],
            step["from"],
            step["to"],
            step["distance"],
            step["traffic_factor"],
            step["dynamic_cost"],
            step["congestion_level"],
        )

    # 3) Morning + simulated closure to force alternate route
    closed_roads = []
    if morning_result["found"] and len(morning_result["best_route"]) > 1:
        closed_roads.append(
            (morning_result["best_route"][0], morning_result["best_route"][1])
        )

    closure_result = router.find_best_route(
        source,
        destination,
        time_of_day=TimeOfDay.MORNING,
        closed_roads=closed_roads,
    )
    logger.info("-" * 60)
    logger.info(
        "Morning Peak with closure %s | cost=%s | path=%s",
        closed_roads,
        _format_cost(closure_result["total_travel_cost"]),
        " -> ".join(closure_result["best_route"]) if closure_result["best_route"] else "NO ROUTE",
    )
    logger.info("Routing decisions: %s", closure_result["routing_decisions"])
    for step in closure_result["turn_by_turn"]:
        logger.info(
            "  Step %d: %s -> %s | dist=%.2f km | factor=%.3f | cost=%.4f | congestion=%s",
            step["step"],
            step["from"],
            step["to"],
            step["distance"],
            step["traffic_factor"],
            step["dynamic_cost"],
            step["congestion_level"],
        )


def run_emergency_routing_demo(graph) -> None:
    """Run Member 4 emergency A* routing and compare against normal Dijkstra."""
    logger.info("=" * 60)
    logger.info("Member 4 – Emergency Routing Module (A*)")
    logger.info("=" * 60)

    source, destination = "N07", "F9"
    router = EmergencyRouter(graph, heuristic_mode="euclidean")
    emergency_result = router.find_emergency_route(
        source,
        destination,
        time_of_day=TimeOfDay.MORNING,
    )

    normal_cost = emergency_result["comparison"]["normal_dijkstra_cost"]
    emergency_cost = emergency_result["comparison"]["emergency_a_star_cost"]

    logger.info(
        "Emergency dispatch %s -> %s | A* route=%s",
        source,
        destination,
        " -> ".join(emergency_result["fast_emergency_route"])
        if emergency_result["fast_emergency_route"]
        else "NO ROUTE",
    )
    logger.info("Normal Dijkstra response time : %s", _format_cost(normal_cost))
    logger.info("Emergency A* response time    : %s", _format_cost(emergency_cost))
    logger.info(
        "Time saved                    : %s",
        _format_cost(emergency_result["comparison"]["time_saved"]),
    )
    logger.info(
        "Improvement                   : %s%%",
        "N/A"
        if emergency_result["comparison"]["improvement_percent"] is None
        else f"{emergency_result['comparison']['improvement_percent']:.2f}",
    )
    logger.info("Comparison log                : %s", emergency_result["comparison"])


def run_transit_optimization_demo(transport: dict) -> None:
    """Run Member 5 transit fleet optimization using Dynamic Programming."""
    logger.info("=" * 60)
    logger.info("Member 5 – Public Transit Optimization Module (DP)")
    logger.info("=" * 60)

    optimizer = TransitOptimizer(transport)
    result = optimizer.optimize_bus_allocation(extra_buses=30)

    logger.info("Total available fleet (with +30 buses): %d", result["total_bus_fleet"])
    logger.info("Old utility score: %.3f", result["old_utility_score"])
    logger.info("New utility score: %.3f", result["new_utility_score"])
    logger.info("Utility improvement: %.3f", result["utility_improvement"])
    logger.info(
        "Improvement percent: %s",
        "N/A" if result["improvement_percent"] is None else f"{result['improvement_percent']:.2f}%",
    )

    logger.info("-" * 60)
    logger.info("Old vs New bus assignments by route:")
    for route in result["route_level_analysis"]:
        logger.info(
            "  [%s] old=%d -> new=%d (delta=%+d) | old_u=%.3f new_u=%.3f",
            route["route_id"],
            route["old_buses"],
            route["new_buses"],
            route["delta_buses"],
            route["old_utility"],
            route["new_utility"],
        )

    logger.info("-" * 60)
    logger.info("Metro transfer points per bus route:")
    for item in result["metro_transfer_analysis"]:
        logger.info(
            "  [%s] transfers=%d stops=%s",
            item["route_id"],
            item["transfer_count"],
            ", ".join(item["transfer_points"]) if item["transfer_points"] else "none",
        )


def main() -> None:
    """Load the sample city graph, validate it, and demonstrate the query API."""

    # ── 1. Load ───────────────────────────────────────────────────────────
    data_dir = _PROJECT_ROOT / "data" / "sample_data"
    loader = DataLoader(data_dir=data_dir, include_proposed=True)

    logger.info("=" * 60)
    logger.info("Smart City Transportation Network – System Startup")
    logger.info("=" * 60)

    graph = loader.load()

    # ── 2. Validate ───────────────────────────────────────────────────────
    graph.validate()

    # ── 3. Basic statistics ───────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Graph summary: %s", graph)
    logger.info("Total nodes  : %d", graph.node_count)
    logger.info("Total edges  : %d", graph.edge_count)

    # ── 4. Facility queries ───────────────────────────────────────────────
    hospitals = graph.get_hospitals()
    logger.info("-" * 60)
    logger.info("Hospitals (%d):", len(hospitals))
    for h in hospitals:
        logger.info("  %s", h)

    fire_stations = graph.get_facilities_by_type(NodeType.FIRE_STATION)
    logger.info("Fire stations (%d):", len(fire_stations))
    for fs in fire_stations:
        logger.info("  %s", fs)

    transit_hubs = graph.get_facilities_by_type(NodeType.TRANSIT_HUB)
    logger.info("Transit hubs (%d):", len(transit_hubs))
    for th in transit_hubs:
        logger.info("  %s", th)

    # ── 5. Dynamic edge weight demonstration ─────────────────────────────
    logger.info("-" * 60)
    logger.info("Dynamic edge weights for N01 (Downtown) → N03 (Uptown):")
    for tod in TimeOfDay:
        w = graph.get_dynamic_edge_weight("N01", "N03", tod)
        logger.info("  %-12s → weight = %.4f", tod.value.capitalize(), w)

    # ── 6. Neighbours of Downtown ─────────────────────────────────────────
    logger.info("-" * 60)
    downtown_neighbors = graph.get_neighbors("N01")
    logger.info("Neighbours of N01 (Downtown) [%d total]:", len(downtown_neighbors))
    for edge in sorted(downtown_neighbors, key=lambda e: e.distance):
        morning_w = edge.get_weight(TimeOfDay.MORNING)
        logger.info(
            "  → %-4s  dist=%.1f km  morning_weight=%.3f  condition=%s",
            edge.to_node,
            edge.distance,
            morning_w,
            edge.road_condition.value,
        )

    # ── 7. Real-time traffic update ───────────────────────────────────────
    logger.info("-" * 60)
    logger.info("Simulating traffic incident: N01→N03 morning factor → 3.0")
    w_before = graph.get_dynamic_edge_weight("N01", "N03", TimeOfDay.MORNING)
    graph.update_traffic_factor("N01", "N03", TimeOfDay.MORNING, 3.0)
    w_after = graph.get_dynamic_edge_weight("N01", "N03", TimeOfDay.MORNING)
    logger.info("  Weight before incident : %.4f", w_before)
    logger.info("  Weight after  incident : %.4f", w_after)

    # ── 8. Public transport ───────────────────────────────────────────────
    transport = loader.get_public_transport()
    logger.info("-" * 60)
    logger.info("Metro lines (%d):", len(transport.get("metro_lines", [])))
    for line in transport.get("metro_lines", []):
        logger.info(
            "  [%s] %s – stops: %s  (every %d min)",
            line["line_id"],
            line["name"],
            " → ".join(line["stops"]),
            line["frequency_minutes"],
        )
    logger.info("Bus routes (%d):", len(transport.get("bus_routes", [])))
    for route in transport.get("bus_routes", []):
        logger.info(
            "  [%s] %s – %d stops  (every %d min)",
            route["route_id"],
            route["name"],
            len(route["stops"]),
            route["frequency_minutes"],
        )

    logger.info("=" * 60)
    logger.info("System ready. Algorithm teams can now consume the graph.")
    logger.info("=" * 60)

    # ── 9. MST Infrastructure Demo (Member 2) ────────────────────────────
    run_mst_demo(graph, data_dir)

    # ── 10. Routing & Traffic Demo (Member 3) ────────────────────────────
    run_routing_demo(graph)

    # ── 11. Emergency Routing Demo (Member 4) ────────────────────────────
    run_emergency_routing_demo(graph)

    # ── 12. Public Transit Optimization Demo (Member 5) ──────────────────
    run_transit_optimization_demo(transport)


if __name__ == "__main__":
    main()
