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


if __name__ == "__main__":
    main()
