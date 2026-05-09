"""Streamlit app for Smart City Intelligent Transportation System."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from algorithms.a_star_emergency import EmergencyRouter
from algorithms.dijkstra_routing import TrafficRouter
from algorithms.dp_transit_optimization import TransitOptimizer
from algorithms.ml_traffic_prediction import TrafficPredictor
from algorithms.mst_infrastructure import MSTOptimizer
from data.data_loader import DataLoader
from utils.enums import TimeOfDay

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "sample_data"
EDGES_PATH = DATA_DIR / "edges.json"


@st.cache_resource
def load_system() -> Dict[str, Any]:
    loader = DataLoader(data_dir=DATA_DIR, include_proposed=True)
    graph = loader.load()
    graph.validate()
    transport = loader.get_public_transport()

    with EDGES_PATH.open(encoding="utf-8") as fh:
        edges_payload = json.load(fh)

    return {
        "graph": graph,
        "transport": transport,
        "mst": MSTOptimizer(graph, data_dir=DATA_DIR),
        "dijkstra": TrafficRouter(graph),
        "emergency": EmergencyRouter(graph, heuristic_mode="euclidean"),
        "predictor": TrafficPredictor.from_edges_file(EDGES_PATH),
        "edges_payload": edges_payload,
    }


def node_table(graph) -> pd.DataFrame:
    records = []
    for node in graph.get_all_nodes():
        records.append(
            {
                "node_id": node.id,
                "node_name": node.name,
                "node_type": node.node_type.value,
                "latitude": node.y_coordinate,
                "longitude": node.x_coordinate,
                "population": node.population,
            }
        )
    return pd.DataFrame(records).sort_values("node_id")


def route_map_df(nodes_df: pd.DataFrame, route: List[str]) -> pd.DataFrame:
    route_set = set(route)
    route_df = nodes_df[nodes_df["node_id"].isin(route_set)].copy()
    route_df["route_order"] = route_df["node_id"].apply(route.index)
    return route_df.sort_values("route_order")


def tod_label_map() -> Dict[str, TimeOfDay]:
    return {
        "Morning": TimeOfDay.MORNING,
        "Afternoon": TimeOfDay.AFTERNOON,
        "Evening": TimeOfDay.EVENING,
        "Night": TimeOfDay.NIGHT,
    }


def main() -> None:
    st.set_page_config(page_title="Smart City ITS", page_icon="🚦", layout="wide")
    st.title("🚦 Smart City Intelligent Transportation System")
    st.caption("Integrated infrastructure planning, routing, emergency dispatch, transit optimization, and ML traffic forecasting.")

    system = load_system()
    graph = system["graph"]
    transport = system["transport"]
    nodes_df = node_table(graph)

    node_options = {
        f"{row.node_id} — {row.node_name} ({row.node_type})": row.node_id
        for row in nodes_df.itertuples(index=False)
    }

    st.sidebar.header("Navigation")
    section = st.sidebar.radio(
        "Select Module",
        [
            "Infrastructure Planning",
            "Traffic Routing",
            "Emergency Dispatch",
            "Transit Optimization",
        ],
    )

    st.sidebar.subheader("Global Route Inputs")
    source_label = st.sidebar.selectbox("Source", list(node_options.keys()), index=0)
    destination_label = st.sidebar.selectbox("Destination", list(node_options.keys()), index=min(1, len(node_options) - 1))
    time_label = st.sidebar.selectbox("Time of Day", list(tod_label_map().keys()), index=0)

    source_id = node_options[source_label]
    destination_id = node_options[destination_label]
    selected_tod = tod_label_map()[time_label]

    st.subheader("City Node Map")
    st.map(nodes_df[["latitude", "longitude"]], use_container_width=True)

    if section == "Infrastructure Planning":
        st.header("🏗️ Infrastructure Planning (MST)")
        if st.button("Run MST Optimization", type="primary"):
            mst_edges = system["mst"].run()
            analysis = system["mst"].get_cost_analysis()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MST Edges", analysis["mst_edge_count"])
            c2.metric("Total Distance (km)", analysis["total_distance_km"])
            c3.metric("Construction Cost (M EGP)", analysis["total_construction_cost_million_egp"])
            c4.metric("Proposed Roads Used", analysis["proposed_road_count"])

            st.dataframe(pd.DataFrame(analysis["edges"]), use_container_width=True)
            st.success(f"MST generated successfully with {len(mst_edges)} edges.")

    elif section == "Traffic Routing":
        st.header("🛣️ Traffic Routing (Dijkstra)")
        if st.button("Find Best Route", type="primary"):
            result = system["dijkstra"].find_best_route(source_id, destination_id, selected_tod)

            if not result["found"]:
                st.error("No route found for the selected configuration.")
            else:
                st.metric("Total Travel Cost", result["total_travel_cost"])
                st.write("**Best Route:**", " → ".join(result["best_route"]))
                st.dataframe(pd.DataFrame(result["turn_by_turn"]), use_container_width=True)

                route_df = route_map_df(nodes_df, result["best_route"])
                fig = px.line_mapbox(
                    route_df,
                    lat="latitude",
                    lon="longitude",
                    hover_name="node_name",
                    text="node_id",
                    zoom=9,
                    height=450,
                )
                fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)

    elif section == "Emergency Dispatch":
        st.header("🚑 Emergency Dispatch (A*)")
        if st.button("Run Emergency Dispatch", type="primary"):
            start = time.perf_counter()
            emergency_result = system["emergency"].find_emergency_route(source_id, destination_id, selected_tod)
            emergency_runtime_ms = round((time.perf_counter() - start) * 1000, 2)

            start = time.perf_counter()
            normal_result = system["dijkstra"].find_best_route(source_id, destination_id, selected_tod)
            dijkstra_runtime_ms = round((time.perf_counter() - start) * 1000, 2)

            if not emergency_result["found"]:
                st.error("No emergency route found for the selected configuration.")
            else:
                st.write("**Emergency Route:**", " → ".join(emergency_result["fast_emergency_route"]))
                st.dataframe(pd.DataFrame(emergency_result["turn_by_turn"]), use_container_width=True)

                st.subheader("Side-by-side Algorithm Comparison")
                comparison = emergency_result["comparison"]

                left, right = st.columns(2)
                with left:
                    st.markdown("### Dijkstra")
                    st.metric("Runtime (ms)", dijkstra_runtime_ms)
                    st.metric("Steps", normal_result["routing_decisions"].get("total_steps", 0))
                    st.metric("Cost", normal_result["total_travel_cost"])
                with right:
                    st.markdown("### A* Emergency")
                    st.metric("Runtime (ms)", emergency_runtime_ms)
                    st.metric("Steps", emergency_result["routing_decisions"].get("total_steps", 0))
                    st.metric("Cost", emergency_result["response_time_cost"])

                metric_df = pd.DataFrame(
                    {
                        "Metric": ["Runtime (ms)", "Steps", "Cost"],
                        "Dijkstra": [
                            dijkstra_runtime_ms,
                            normal_result["routing_decisions"].get("total_steps", 0),
                            normal_result["total_travel_cost"] or 0,
                        ],
                        "A* Emergency": [
                            emergency_runtime_ms,
                            emergency_result["routing_decisions"].get("total_steps", 0),
                            emergency_result["response_time_cost"] or 0,
                        ],
                    }
                )

                fig = px.bar(
                    metric_df.melt(id_vars="Metric", var_name="Algorithm", value_name="Value"),
                    x="Metric",
                    y="Value",
                    color="Algorithm",
                    barmode="group",
                    height=420,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.info(
                    f"Estimated time saved: {comparison.get('time_saved')} | "
                    f"Improvement: {comparison.get('improvement_percent')}%"
                )

    elif section == "Transit Optimization":
        st.header("🚌 Transit Optimization (Dynamic Programming)")
        extra_buses = st.slider("Extra Buses to Allocate", min_value=0, max_value=100, value=30, step=5)

        if st.button("Optimize Fleet Allocation", type="primary"):
            optimizer = TransitOptimizer(transport)
            result = optimizer.optimize_bus_allocation(extra_buses=extra_buses)

            c1, c2, c3 = st.columns(3)
            c1.metric("Old Utility", result["old_utility_score"])
            c2.metric("New Utility", result["new_utility_score"])
            c3.metric("Improvement (%)", result["improvement_percent"])

            st.subheader("Route-level Allocation Analysis")
            st.dataframe(pd.DataFrame(result["route_level_analysis"]), use_container_width=True)

            st.subheader("Metro Transfer Analysis")
            st.dataframe(pd.DataFrame(result["metro_transfer_analysis"]), use_container_width=True)

    st.divider()
    st.subheader("🤖 AI Traffic Prediction")

    roads = system["edges_payload"].get("current_roads", []) + system["edges_payload"].get("proposed_roads", [])
    edge_labels = [f"{r['from_node']} → {r['to_node']}" for r in roads]
    selected_edge_label = st.selectbox("Road Segment", edge_labels)
    selected_edge = roads[edge_labels.index(selected_edge_label)]

    predicted_factor = system["predictor"].predict_for_edge(selected_edge, selected_tod)
    st.metric("Predicted Congestion Factor", predicted_factor)

    actual_factor = selected_edge.get("traffic_factors", {}).get(selected_tod.value)
    if actual_factor is not None:
        st.caption(f"Reference factor in dataset for {selected_tod.value}: {actual_factor}")


if __name__ == "__main__":
    main()
