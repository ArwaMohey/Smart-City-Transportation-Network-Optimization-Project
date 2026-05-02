"""Data loading module for the Smart City Transportation Network System.

:class:`DataLoader` reads structured JSON files from a data directory and
produces a fully-populated :class:`~core.graph.Graph` ready for use by
algorithm teams.

Expected directory layout::

    <data_dir>/
        nodes.json            – 15 neighbourhoods + 10 facilities
        edges.json            – current roads + proposed roads
        public_transport.json – metro lines + bus routes (metadata only)

Usage::

    from data.data_loader import DataLoader
    from config.logging_config import setup_logging
    import logging

    setup_logging(level=logging.INFO)

    loader = DataLoader("data/sample_data")
    graph = loader.load()
    graph.validate()

    transport = loader.get_public_transport()
    print(transport["metro_lines"])
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.graph import Graph
from models.edge import Edge
from models.node import Node
from utils.enums import NodeType, RoadCondition, TimeOfDay

logger = logging.getLogger("smart_city.data.data_loader")

# ── File name constants ────────────────────────────────────────────────────
_NODES_FILE = "nodes.json"
_EDGES_FILE = "edges.json"
_TRANSPORT_FILE = "public_transport.json"

# ── Mapping from raw JSON string keys to enum values ──────────────────────
_NODE_TYPE_MAP: Dict[str, NodeType] = {member.value: member for member in NodeType}
_ROAD_CONDITION_MAP: Dict[str, RoadCondition] = {
    member.value: member for member in RoadCondition
}
_TIME_OF_DAY_MAP: Dict[str, TimeOfDay] = {member.value: member for member in TimeOfDay}


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────


class DataLoadError(IOError):
    """Raised when a required data file cannot be read or parsed."""


class DataParseError(ValueError):
    """Raised when a data record contains an unrecoverable schema violation."""


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DataLoader:
    """Loads city transport data from JSON files and constructs a :class:`~core.graph.Graph`.

    Args:
        data_dir:        Path to the directory that contains the JSON data files.
        directed:        Whether the resulting graph should be directed
                         (default: ``False``).
        include_proposed: When *True* (default), proposed roads are also added
                          to the graph.

    Attributes:
        data_dir (Path):  Resolved path to the data directory.
        directed (bool):  Directionality of the resulting graph.

    Raises:
        DataLoadError:  If a required file is missing or cannot be parsed as
                        JSON.
        DataParseError: If a record contains an unrecoverable field error.

    Example::

        loader = DataLoader("data/sample_data")
        graph  = loader.load()
        graph.validate()
    """

    data_dir: Path = field(default_factory=lambda: Path("data/sample_data"))
    directed: bool = False
    include_proposed: bool = True

    # Populated after load() is called.
    _graph: Optional[Graph] = field(default=None, init=False, repr=False)
    _transport_data: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        # Accept both str and Path for convenience.
        self.data_dir = Path(self.data_dir)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def load(self) -> Graph:
        """Load all data files and return a populated :class:`~core.graph.Graph`.

        The method processes files in dependency order:

        1. Nodes are loaded and added first.
        2. Edges are loaded second (they reference nodes by id).
        3. Public-transport metadata is loaded last (no graph mutations).

        Returns:
            A :class:`~core.graph.Graph` ready for algorithm consumption.

        Raises:
            DataLoadError:  If a required file is missing or cannot be parsed.
            DataParseError: If a record cannot be converted to a domain object.
        """
        logger.info("DataLoader.load() started (data_dir=%s).", self.data_dir)

        graph = Graph(directed=self.directed)

        node_count = self._load_nodes(graph)
        edge_count = self._load_edges(graph)
        self._load_transport()

        self._graph = graph
        logger.info(
            "DataLoader.load() finished: %d node(s), %d edge(s) loaded.",
            node_count,
            edge_count,
        )
        return graph

    def get_public_transport(self) -> Dict[str, Any]:
        """Return public-transport metadata loaded from ``public_transport.json``.

        Must be called *after* :meth:`load`.

        Returns:
            Dictionary with keys ``"metro_lines"`` and ``"bus_routes"``.

        Raises:
            RuntimeError: If :meth:`load` has not been called yet.
        """
        if not self._transport_data and self._graph is None:
            raise RuntimeError(
                "Public transport data is not available. Call load() first."
            )
        return self._transport_data

    # ──────────────────────────────────────────────────────────────────────
    # Private loaders
    # ──────────────────────────────────────────────────────────────────────

    def _load_nodes(self, graph: Graph) -> int:
        """Parse nodes.json and add all nodes to *graph*.

        Processes both ``"neighborhoods"`` and ``"facilities"`` sections.
        Records that fail validation are skipped with an ERROR log entry;
        processing continues for the remaining records.

        Args:
            graph: Target :class:`~core.graph.Graph`.

        Returns:
            Number of nodes successfully added.
        """
        raw = self._read_json(_NODES_FILE)
        count = 0

        for section in ("neighborhoods", "facilities"):
            records: List[Dict[str, Any]] = raw.get(section, [])
            if not records:
                logger.warning("No records found in '%s' section of %s.", section, _NODES_FILE)

            for record in records:
                node = self._parse_node(record)
                if node is None:
                    continue
                try:
                    graph.add_node(node)
                    count += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to add node id=%r to graph: %s", record.get("id"), exc
                    )

        logger.info("Nodes loaded: %d node(s) added.", count)
        return count

    def _load_edges(self, graph: Graph) -> int:
        """Parse edges.json and add road segments to *graph*.

        Processes both ``"current_roads"`` and ``"proposed_roads"`` sections.
        Proposed roads are only added when :attr:`include_proposed` is *True*.

        Args:
            graph: Target :class:`~core.graph.Graph`.

        Returns:
            Number of edges successfully added.
        """
        raw = self._read_json(_EDGES_FILE)
        count = 0

        sections = [("current_roads", False)]
        if self.include_proposed:
            sections.append(("proposed_roads", True))

        for section_key, is_proposed_default in sections:
            records: List[Dict[str, Any]] = raw.get(section_key, [])
            if not records:
                logger.warning("No records found in '%s' section of %s.", section_key, _EDGES_FILE)

            for record in records:
                # Allow the JSON record to override the is_proposed flag.
                edge = self._parse_edge(record, is_proposed_default=is_proposed_default)
                if edge is None:
                    continue
                try:
                    graph.add_edge(edge)
                    count += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to add edge %r->%r to graph: %s",
                        record.get("from_node"),
                        record.get("to_node"),
                        exc,
                    )

        logger.info("Edges loaded: %d edge(s) added.", count)
        return count

    def _load_transport(self) -> None:
        """Parse public_transport.json and cache the result internally.

        The data is stored as-is (no graph mutations); callers access it via
        :meth:`get_public_transport`.
        """
        try:
            raw = self._read_json(_TRANSPORT_FILE)
            metro_count = len(raw.get("metro_lines", []))
            bus_count = len(raw.get("bus_routes", []))
            self._transport_data = raw
            logger.info(
                "Public transport loaded: %d metro line(s), %d bus route(s).",
                metro_count,
                bus_count,
            )
        except DataLoadError as exc:
            logger.warning(
                "Public transport file could not be loaded (%s). "
                "Proceeding without transport data.",
                exc,
            )
            self._transport_data = {"metro_lines": [], "bus_routes": []}

    # ──────────────────────────────────────────────────────────────────────
    # Parsers – raw dict → domain object
    # ──────────────────────────────────────────────────────────────────────

    def _parse_node(self, record: Dict[str, Any]) -> Optional[Node]:
        """Convert a raw dictionary to a :class:`~models.node.Node`.

        Args:
            record: Raw dictionary from JSON.

        Returns:
            A :class:`~models.node.Node` instance, or *None* on error.
        """
        node_id: str = record.get("id", "<unknown>")
        try:
            raw_type: str = record["node_type"]
            node_type = _NODE_TYPE_MAP.get(raw_type)
            if node_type is None:
                raise DataParseError(
                    f"Unknown node_type={raw_type!r} for node id={node_id!r}. "
                    f"Valid values: {list(_NODE_TYPE_MAP.keys())}."
                )

            return Node(
                id=record["id"],
                name=record["name"],
                node_type=node_type,
                population=int(record.get("population", 0)),
                x_coordinate=float(record["x_coordinate"]),
                y_coordinate=float(record["y_coordinate"]),
            )
        except (KeyError, TypeError, ValueError, DataParseError) as exc:
            logger.error("Cannot parse node id=%r: %s", node_id, exc)
            return None

    def _parse_edge(
        self,
        record: Dict[str, Any],
        is_proposed_default: bool = False,
    ) -> Optional[Edge]:
        """Convert a raw dictionary to an :class:`~models.edge.Edge`.

        Args:
            record:              Raw dictionary from JSON.
            is_proposed_default: Fallback ``is_proposed`` value when the
                                 record does not contain the field.

        Returns:
            An :class:`~models.edge.Edge` instance, or *None* on error.
        """
        from_id: str = record.get("from_node", "<unknown>")
        to_id: str = record.get("to_node", "<unknown>")
        try:
            raw_condition: str = record["road_condition"]
            road_condition = _ROAD_CONDITION_MAP.get(raw_condition)
            if road_condition is None:
                raise DataParseError(
                    f"Unknown road_condition={raw_condition!r} for edge "
                    f"{from_id!r}->{to_id!r}. "
                    f"Valid values: {list(_ROAD_CONDITION_MAP.keys())}."
                )

            raw_tf: Dict[str, float] = record.get("traffic_factors", {})
            traffic_factors: Dict[TimeOfDay, float] = {}
            for raw_tod, factor in raw_tf.items():
                tod = _TIME_OF_DAY_MAP.get(raw_tod)
                if tod is None:
                    logger.warning(
                        "Unknown time_of_day key %r in edge %r->%r; skipping factor.",
                        raw_tod, from_id, to_id,
                    )
                    continue
                traffic_factors[tod] = float(factor)

            if not traffic_factors:
                logger.warning(
                    "Edge %r->%r has no valid traffic_factors; using default 1.0 for all.",
                    from_id, to_id,
                )
                traffic_factors = {tod: 1.0 for tod in TimeOfDay}

            return Edge(
                from_node=record["from_node"],
                to_node=record["to_node"],
                distance=float(record["distance"]),
                base_capacity=int(record.get("base_capacity", 0)),
                road_condition=road_condition,
                traffic_factors=traffic_factors,
                is_proposed=bool(record.get("is_proposed", is_proposed_default)),
                is_bidirectional=bool(record.get("is_bidirectional", True)),
            )
        except (KeyError, TypeError, ValueError, DataParseError) as exc:
            logger.error("Cannot parse edge %r->%r: %s", from_id, to_id, exc)
            return None

    # ──────────────────────────────────────────────────────────────────────
    # File I/O helpers
    # ──────────────────────────────────────────────────────────────────────

    def _read_json(self, filename: str) -> Dict[str, Any]:
        """Read and parse a JSON file from :attr:`data_dir`.

        Args:
            filename: Base filename (e.g. ``"nodes.json"``).

        Returns:
            Parsed JSON content as a dictionary.

        Raises:
            DataLoadError: If the file does not exist or cannot be parsed.
        """
        filepath: Path = self.data_dir / filename
        logger.debug("Reading JSON file: %s", filepath)

        if not filepath.is_file():
            raise DataLoadError(f"Data file not found: {filepath}")

        try:
            with filepath.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise DataLoadError(
                f"Failed to parse JSON from {filepath}: {exc}"
            ) from exc

        logger.debug("Successfully parsed %s.", filepath)
        return data

    def __repr__(self) -> str:
        return (
            f"DataLoader(data_dir={str(self.data_dir)!r}, "
            f"directed={self.directed}, include_proposed={self.include_proposed})"
        )
