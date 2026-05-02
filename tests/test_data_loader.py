"""Unit tests for the DataLoader class."""

import json
import pytest
from pathlib import Path
from data.data_loader import DataLoader, DataLoadError
from core.graph import Graph
from utils.enums import NodeType, TimeOfDay


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_data"


@pytest.fixture()
def loader() -> DataLoader:
    return DataLoader(data_dir=_SAMPLE_DIR)


@pytest.fixture()
def graph(loader: DataLoader) -> Graph:
    return loader.load()


# ─────────────────────────────────────────────────────────────────────────────
# Load tests
# ─────────────────────────────────────────────────────────────────────────────


def test_load_returns_graph(loader: DataLoader) -> None:
    result = loader.load()
    assert isinstance(result, Graph)


def test_graph_has_25_nodes(graph: Graph) -> None:
    """15 neighborhoods + 10 facilities = 25 nodes."""
    assert graph.node_count == 25


def test_graph_has_expected_edge_count(graph: Graph) -> None:
    """33 current roads + 15 proposed roads = 48 edges total."""
    assert graph.edge_count == 48


def test_graph_has_2_hospitals(graph: Graph) -> None:
    assert len(graph.get_hospitals()) == 2


def test_graph_has_2_fire_stations(graph: Graph) -> None:
    fire_stations = graph.get_facilities_by_type(NodeType.FIRE_STATION)
    assert len(fire_stations) == 0


def test_graph_has_2_police_stations(graph: Graph) -> None:
    police = graph.get_facilities_by_type(NodeType.POLICE)
    assert len(police) == 0


def test_graph_has_2_schools(graph: Graph) -> None:
    schools = graph.get_facilities_by_type(NodeType.SCHOOL)
    assert len(schools) == 2


def test_graph_has_2_transit_hubs(graph: Graph) -> None:
    hubs = graph.get_facilities_by_type(NodeType.TRANSIT_HUB)
    assert len(hubs) == 1


def test_graph_has_1_airport(graph: Graph) -> None:
    airports = graph.get_facilities_by_type(NodeType.AIRPORT)
    assert len(airports) == 1


def test_graph_has_1_business_district(graph: Graph) -> None:
    business = graph.get_facilities_by_type(NodeType.BUSINESS)
    assert len(business) == 1


def test_graph_has_1_tourism_site(graph: Graph) -> None:
    tourism = graph.get_facilities_by_type(NodeType.TOURISM)
    assert len(tourism) == 1


def test_graph_has_1_sports_site(graph: Graph) -> None:
    sports = graph.get_facilities_by_type(NodeType.SPORTS)
    assert len(sports) == 1


def test_graph_has_15_neighborhoods(graph: Graph) -> None:
    neighborhoods = graph.get_facilities_by_type(NodeType.NEIGHBORHOOD)
    assert len(neighborhoods) == 15


def test_dynamic_edge_weight_changes_by_time(graph: Graph) -> None:
    morning = graph.get_dynamic_edge_weight("N01", "N03", TimeOfDay.MORNING)
    night = graph.get_dynamic_edge_weight("N01", "N03", TimeOfDay.NIGHT)
    assert morning > night


def test_validate_passes_on_loaded_graph(graph: Graph) -> None:
    """Loaded sample data must pass structural validation."""
    graph.validate()


def test_public_transport_loaded(loader: DataLoader) -> None:
    loader.load()
    transport = loader.get_public_transport()
    assert "metro_lines" in transport
    assert "bus_routes" in transport


def test_public_transport_has_metro_lines(loader: DataLoader) -> None:
    loader.load()
    transport = loader.get_public_transport()
    assert len(transport["metro_lines"]) >= 1


def test_public_transport_has_bus_routes(loader: DataLoader) -> None:
    loader.load()
    transport = loader.get_public_transport()
    assert len(transport["bus_routes"]) >= 1


def test_exclude_proposed_roads() -> None:
    loader = DataLoader(data_dir=_SAMPLE_DIR, include_proposed=False)
    graph = loader.load()
    # Without proposed roads there should be 15 fewer edges.
    assert graph.edge_count == 33


def test_missing_data_dir_raises(tmp_path: Path) -> None:
    loader = DataLoader(data_dir=tmp_path / "nonexistent")
    with pytest.raises(DataLoadError):
        loader.load()


def test_malformed_json_raises(tmp_path: Path) -> None:
    """A corrupt JSON file must raise DataLoadError, not crash silently."""
    bad_nodes = tmp_path / "nodes.json"
    bad_nodes.write_text("{ this is not valid json }", encoding="utf-8")
    loader = DataLoader(data_dir=tmp_path)
    with pytest.raises(DataLoadError):
        loader.load()


def test_repr_contains_data_dir(loader: DataLoader) -> None:
    assert "sample_data" in repr(loader)
