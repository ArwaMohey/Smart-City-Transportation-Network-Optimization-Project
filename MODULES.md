# Smart City Transportation Network – Module & File Reference

This document describes every file and module in the project so that new contributors and algorithm teams can quickly orient themselves.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Tree](#directory-tree)
3. [Entry Point](#entry-point)
4. [Module: `config`](#module-config)
5. [Module: `models`](#module-models)
6. [Module: `core`](#module-core)
7. [Module: `utils`](#module-utils)
8. [Module: `data`](#module-data)
9. [Module: `tests`](#module-tests)
10. [Supporting Files](#supporting-files)

---

## Project Overview

The **Smart City Transportation Network Optimization System** models a city's road network as a weighted graph and exposes a clean API that algorithm teams (Dijkstra, A\*, MST, Emergency Routing, Dynamic Programming) can consume without touching internal storage details.

Core capabilities:

- Load city topology (nodes + edges) from structured JSON files.
- Compute **dynamic edge weights** that vary with time-of-day traffic and road-condition penalties.
- Apply **real-time traffic updates** and automatically invalidate the weight cache.
- Query critical facilities (hospitals, fire stations, police, schools) and public-transport metadata.
- Validate graph structural integrity before algorithm execution.

---

## Directory Tree

```
.
├── main.py                        # Application entry point / demo
├── requirements.txt               # Python dependencies
├── conftest.py                    # Pytest bootstrap (sys.path setup)
│
├── config/
│   ├── __init__.py
│   └── logging_config.py          # Application-wide logging setup
│
├── models/
│   ├── __init__.py
│   ├── node.py                    # Node dataclass
│   └── edge.py                    # Edge dataclass
│
├── core/
│   ├── __init__.py
│   └── graph.py                   # Graph engine (adjacency-list)
│
├── utils/
│   ├── __init__.py
│   ├── enums.py                   # Domain enumerations
│   └── validators.py              # Graph structural validator
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py             # JSON → Graph loader
│   └── sample_data/
│       ├── nodes.json             # 15 neighbourhoods + 10 facilities
│       ├── edges.json             # Current roads + proposed roads
│       └── public_transport.json  # Metro lines + bus routes
│
└── tests/
    ├── __init__.py
    ├── test_node.py
    ├── test_edge.py
    ├── test_graph.py
    └── test_data_loader.py
```

---

## Entry Point

### `main.py`

Demonstrates the full **load → validate → query** workflow that algorithm teams will use as their starting point.

**What it does (step by step):**

| Step | Action |
|------|--------|
| 1 | Initialises logging via `setup_logging()`. |
| 2 | Creates a `DataLoader` pointing at `data/sample_data/` with proposed roads enabled. |
| 3 | Calls `loader.load()` to build the `Graph`. |
| 4 | Calls `graph.validate()` to check structural integrity. |
| 5 | Logs graph statistics (node/edge counts). |
| 6 | Queries hospitals, fire stations, and transit hubs. |
| 7 | Iterates over all `TimeOfDay` values and logs the dynamic weight for the N01→N02 edge. |
| 8 | Lists all neighbours of node N01 (Downtown) with their morning weights. |
| 9 | Simulates a real-time traffic incident by calling `graph.update_traffic_factor()`. |
| 10 | Logs metro lines and bus routes from public-transport metadata. |

**How to run:**

```bash
python main.py
```

---

## Module: `config`

### `config/logging_config.py`

Centralised logging configuration for the entire application.

**Rules enforced project-wide:**
- Every module must obtain its logger via `get_logger(name)` – never `logging.getLogger()` directly.
- `print()` statements are **not allowed** anywhere in the codebase.
- All loggers live under the `smart_city.*` namespace hierarchy.

#### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `setup_logging` | `(level, log_file, log_format) → Logger` | Configures the application root logger (`smart_city`). Attaches a console (stdout) handler and an optional rotating-file handler (10 MB, 5 backups). Must be called **once** at startup before any other import. |
| `get_logger` | `(name: str) → Logger` | Returns a named child logger under `smart_city.<name>`. Example: `get_logger("core.graph")` → logger `smart_city.core.graph`. |

**Log format (default):**
```
YYYY-MM-DD HH:MM:SS | LEVEL    | smart_city.module:line | message
```

---

## Module: `models`

Domain objects that represent the fundamental building blocks of the network.

### `models/node.py`

Defines the **immutable** `Node` dataclass.

#### Class: `Node`

A `Node` represents any location in the city – a neighbourhood, a hospital, a school, etc.

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Unique identifier (e.g. `"N01"`, `"H01"`). |
| `name` | `str` | Human-readable label (e.g. `"Downtown"`). |
| `node_type` | `NodeType` | Category (see `utils/enums.py`). |
| `population` | `int` | Resident/user count (≥ 0). |
| `x_coordinate` | `float` | Horizontal grid position in `[0, 1000]`. |
| `y_coordinate` | `float` | Vertical grid position in `[0, 1000]`. |

The dataclass is **frozen** (`@dataclass(frozen=True)`), so it is hashable and safe to use as a dictionary key or in sets.

**Validation** happens automatically in `__post_init__` and raises `NodeValidationError` for:
- Empty/non-string `id` or `name`.
- `node_type` that is not a `NodeType` enum member.
- Negative `population`.
- Coordinates outside `[0, 1000]`.

#### Key Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `is_critical_facility()` | `bool` | `True` if the node is HOSPITAL, FIRE_STATION, POLICE, or SCHOOL. |
| `euclidean_distance_to(other)` | `float` | Straight-line distance; useful as A\* heuristic. |

#### Exception: `NodeValidationError`

Subclass of `ValueError`. Raised when any `Node` attribute fails validation.

---

### `models/edge.py`

Defines the **mutable** `Edge` dataclass that represents a road segment.

#### Class: `Edge`

| Attribute | Type | Description |
|-----------|------|-------------|
| `from_node` | `str` | Identifier of the source node. |
| `to_node` | `str` | Identifier of the destination node. |
| `distance` | `float` | Road length in kilometres (> 0). |
| `base_capacity` | `int` | Max throughput in vehicles/hour (≥ 0). |
| `road_condition` | `RoadCondition` | Surface quality; drives a penalty multiplier. |
| `traffic_factors` | `Dict[TimeOfDay, float]` | Per-time-of-day congestion multipliers (all > 0). |
| `is_proposed` | `bool` | `True` for roads not yet constructed. |
| `is_bidirectional` | `bool` | `True` if traversable in both directions. |

**Weight formula:**

```
weight = distance × traffic_factor(time) × condition_penalty
```

Road-condition penalty multipliers:

| RoadCondition | Multiplier |
|---------------|-----------|
| EXCELLENT | 1.0 |
| GOOD | 1.1 |
| FAIR | 1.3 |
| POOR | 1.6 |

Computed weights are **cached** per `TimeOfDay` and reused on subsequent calls.

#### Key Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_weight(time_of_day)` | `float` | Returns the effective travel cost; uses internal cache. |
| `invalidate_cache()` | `None` | Clears the weight cache. Must be called after updating `traffic_factors` or `road_condition`. |

#### Exception: `EdgeValidationError`

Subclass of `ValueError`. Raised when any `Edge` attribute fails validation (e.g. self-loop, non-positive distance, invalid traffic factor).

---

## Module: `core`

### `core/graph.py`

The central graph data structure. Every algorithm team interacts with the network through this class.

#### Design Decisions

- **Adjacency-list** representation: `O(1)` neighbour look-up; compact for sparse networks.
- **Directed / undirected** mode: controlled by the `directed` flag at construction. In undirected mode, both `(u→v)` and `(v→u)` are registered in the adjacency map but **share the same `Edge` object**.
- **No tight coupling to validation**: graph validation is delegated to `GraphValidator` and triggered lazily via `validate()`.

#### Class: `Graph`

**Constructor:**

```python
Graph(directed: bool = False)
```

#### Node Operations

| Method | Description |
|--------|-------------|
| `add_node(node)` | Register a `Node`. Raises `DuplicateNodeError` if the id already exists. |
| `remove_node(node_id)` | Remove a node and all edges that reference it. |
| `get_node(node_id)` | Return the `Node` for the given id. |
| `get_all_nodes()` | Return a list of all registered nodes. |

#### Edge Operations

| Method | Description |
|--------|-------------|
| `add_edge(edge)` | Register an `Edge`. For undirected graphs, also registers the reverse direction automatically. Raises `DuplicateEdgeError` if the `(from, to)` pair already exists. |
| `remove_edge(from_node, to_node)` | Remove an edge (and its reverse if undirected). |
| `get_neighbors(node_id)` | Return all `Edge` objects originating from `node_id`. |
| `get_all_edges()` | Return a deduplicated list of all edges. |

#### Query API

| Method | Description |
|--------|-------------|
| `get_hospitals()` | Return all nodes of type `HOSPITAL`. |
| `get_facilities_by_type(node_type)` | Return all nodes matching the given `NodeType`. |
| `get_dynamic_edge_weight(u, v, time)` | Return the computed weight for edge `(u→v)` at the given `TimeOfDay`. |
| `update_traffic_factor(from, to, time, factor)` | Update the congestion multiplier for an edge at a specific time; automatically invalidates the weight cache. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `directed` | `bool` | Whether the graph is directed. |
| `node_count` | `int` | Number of registered nodes. |
| `edge_count` | `int` | Number of unique registered edges. |

#### Exceptions

| Exception | Description |
|-----------|-------------|
| `GraphError` | Base exception for structural graph errors. |
| `NodeNotFoundError` | Raised when a referenced node does not exist. |
| `DuplicateNodeError` | Raised when adding a node with an already-existing id. |
| `DuplicateEdgeError` | Raised when adding an edge that already exists. |

---

## Module: `utils`

### `utils/enums.py`

Single authoritative source for all domain-level symbolic constants. All other modules import enums from here to avoid magic strings.

#### `NodeType`

Semantic category of a city location.

| Value | Description |
|-------|-------------|
| `NEIGHBORHOOD` | Residential area. |
| `HOSPITAL` | Medical facility. |
| `FIRE_STATION` | Emergency fire-fighting station. |
| `POLICE` | Police station or headquarters. |
| `SCHOOL` | Educational institution. |
| `COMMERCIAL` | Shopping or business district. |
| `INDUSTRIAL` | Manufacturing or logistics zone. |
| `PARK` | Public recreational area. |
| `TRANSIT_HUB` | Major public-transport interchange. |

#### `RoadCondition`

Physical state of a road segment; determines the penalty multiplier applied in edge weight calculation.

| Value | Multiplier |
|-------|-----------|
| `EXCELLENT` | 1.0 |
| `GOOD` | 1.1 |
| `FAIR` | 1.3 |
| `POOR` | 1.6 |

#### `TimeOfDay`

Discrete time windows for dynamic traffic modelling.

| Value | Typical Hours |
|-------|--------------|
| `MORNING` | 06:00–10:00 |
| `AFTERNOON` | 10:00–16:00 |
| `EVENING` | 16:00–20:00 |
| `NIGHT` | 20:00–06:00 |

#### `TransportMode`

Modes of public transport supported by the network.

| Value | Description |
|-------|-------------|
| `METRO` | Underground or surface rapid transit. |
| `BUS` | Standard city bus service. |
| `TRAM` | Light-rail street tram. |

---

### `utils/validators.py`

Structural integrity checker for the graph. Decoupled from `Graph` (Single Responsibility Principle) so new validation rules can be added without modifying the core engine.

#### Class: `GraphValidator`

**Constructor:** `GraphValidator(graph: Graph)`

#### `validate()` method

Runs all checks. Warnings are logged at `WARNING` level; critical errors are logged at `ERROR` level and cause a `GraphValidationError` to be raised after all checks complete.

**Checks performed:**

| Check | Severity | Description |
|-------|----------|-------------|
| Edge reference integrity | Error | Every `from_node` and `to_node` in every edge must correspond to an existing node. |
| Node coordinate bounds | Error | All `x_coordinate` and `y_coordinate` values must be in `[0, 1000]`. |
| Isolated nodes | Warning | Nodes with zero outgoing **and** zero incoming edges are flagged (suspicious but not necessarily invalid). |

#### Exception: `GraphValidationError`

Raised by `validate()` when one or more critical errors are detected.

---

## Module: `data`

### `data/data_loader.py`

Reads structured JSON files from a data directory and constructs a fully-populated `Graph`.

#### Class: `DataLoader`

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | `Path` or `str` | `"data/sample_data"` | Directory containing the JSON files. |
| `directed` | `bool` | `False` | Whether to build a directed graph. |
| `include_proposed` | `bool` | `True` | Whether to include proposed (not yet built) roads. |

#### Public Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `load()` | `Graph` | Loads all JSON files in dependency order (nodes → edges → transport) and returns the populated graph. |
| `get_public_transport()` | `Dict` | Returns the public-transport metadata dict (`metro_lines`, `bus_routes`). Must be called after `load()`. |

**Processing order in `load()`:**
1. Parse `nodes.json` → add all nodes to the graph.
2. Parse `edges.json` → add current roads (and proposed roads if `include_proposed=True`).
3. Parse `public_transport.json` → cache metadata (no graph mutations).

**Error handling:** Records that fail validation are **skipped with an ERROR log** and processing continues for the remaining records. This means a single malformed entry will not abort the entire load.

#### Exceptions

| Exception | Description |
|-----------|-------------|
| `DataLoadError` | Raised when a required file is missing or cannot be parsed as JSON. |
| `DataParseError` | Raised when a data record contains an unrecoverable schema violation. |

---

### `data/sample_data/`

Sample dataset representing a small city network.

#### `nodes.json`

Contains two sections:

- **`neighborhoods`** – 15 residential district nodes (e.g. Downtown, Riverside, Westside).
- **`facilities`** – 10 critical-facility nodes (hospitals, fire stations, police stations, schools, transit hubs).

Each record schema:

```json
{
  "id": "N01",
  "name": "Downtown",
  "node_type": "neighborhood",
  "population": 50000,
  "x_coordinate": 500.0,
  "y_coordinate": 500.0
}
```

#### `edges.json`

Contains two sections:

- **`current_roads`** – existing road segments.
- **`proposed_roads`** – planned roads (`is_proposed: true`).

Each record schema:

```json
{
  "from_node": "N01",
  "to_node": "N02",
  "distance": 2.5,
  "base_capacity": 1500,
  "road_condition": "good",
  "is_bidirectional": true,
  "is_proposed": false,
  "traffic_factors": {
    "morning":   1.6,
    "afternoon": 1.2,
    "evening":   1.5,
    "night":     0.8
  }
}
```

#### `public_transport.json`

Metadata only (not added to the graph adjacency structure). Contains:

- **`metro_lines`** – each entry has `line_id`, `name`, `stops` (list of node ids), `frequency_minutes`.
- **`bus_routes`** – each entry has `route_id`, `name`, `stops`, `frequency_minutes`.

---

## Module: `tests`

All tests are collected and run with:

```bash
python -m pytest tests/ -v
```

All 82 tests should pass on a clean checkout.

### `tests/test_node.py`

Unit tests for `models/node.py`:
- Valid `Node` construction.
- `NodeValidationError` for each invalid attribute (empty id/name, bad type, negative population, out-of-range coordinates).
- `is_critical_facility()` for each `NodeType`.
- `euclidean_distance_to()` correctness.
- Frozen (immutable) behaviour.

### `tests/test_edge.py`

Unit tests for `models/edge.py`:
- Valid `Edge` construction.
- `EdgeValidationError` for invalid attributes (self-loops, non-positive distance, bad traffic factors).
- `get_weight()` formula verification for all `RoadCondition` values.
- Weight caching and `invalidate_cache()`.
- `__hash__` and `__eq__` behaviour.

### `tests/test_graph.py`

Unit tests for `core/graph.py`:
- Adding/removing nodes and edges in both directed and undirected modes.
- Duplicate node/edge detection.
- `get_neighbors()`, `get_all_nodes()`, `get_all_edges()` (deduplication in undirected mode).
- `get_hospitals()`, `get_facilities_by_type()`.
- `get_dynamic_edge_weight()` and `update_traffic_factor()` (including cache invalidation).
- `validate()` integration with `GraphValidator`.

### `tests/test_data_loader.py`

Integration tests for `data/data_loader.py`:
- Successful load from `sample_data/`.
- `include_proposed=True/False` behaviour.
- `directed=True` graph construction.
- `get_public_transport()` before and after `load()`.
- Missing file and malformed JSON error handling.

---

## Supporting Files

### `conftest.py`

Pytest configuration file. Ensures the project root is on `sys.path` so that absolute imports (e.g. `from models.node import Node`) work correctly in all test files without requiring package installation.

### `requirements.txt`

Lists Python dependencies. The core system uses **only the Python standard library**. The only external dependency is `pytest` for running the test suite.

### `README.md`

One-line project description.

### PDF documents

| File | Description |
|------|-------------|
| `CSE112-Practical Project.pdf` | Practical project specification. |
| `CSE112-Theoretical Project.pdf` | Theoretical project specification. |
| `Project_Provided_Data.pdf` | Data specification provided with the project. |
