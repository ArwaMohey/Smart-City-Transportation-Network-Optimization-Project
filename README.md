# 🚦 Smart City Transportation Network Optimization

A production-style **Intelligent Transportation System (ITS)** that integrates core graph algorithms with an interactive web dashboard for city-scale planning and decision support.

## ✨ Highlights

- **Infrastructure Planning (MST / Kruskal):** optimal road-network expansion with cost-aware prioritization
- **Traffic Routing (Dijkstra):** congestion-aware shortest path routing by time of day
- **Emergency Dispatch (A*):** priority routing for emergency vehicles with reduced congestion impact
- **Transit Optimization (Dynamic Programming):** bus fleet reallocation to improve utility and coverage
- **AI Traffic Prediction (ML Bonus):** `RandomForestRegressor` forecasting congestion factors from temporal traffic data
- **Interactive Streamlit UI:** one-click workflows, metrics, maps, and route comparisons

## 🧱 Project Architecture

```text
Smart-City-Transportation-Network-Optimization-Project/
├── core/
│   └── graph.py                       # Graph data structure and shared APIs
├── models/
│   ├── node.py                        # Node domain model
│   └── edge.py                        # Edge domain model and dynamic weights
├── algorithms/
│   ├── mst_infrastructure.py          # Kruskal MST optimizer
│   ├── dijkstra_routing.py            # Traffic-aware Dijkstra routing
│   ├── a_star_emergency.py            # Emergency A* routing
│   ├── dp_transit_optimization.py     # DP transit optimizer
│   └── ml_traffic_prediction.py       # ML congestion predictor
├── data/
│   └── sample_data/                   # Nodes, roads, and transit metadata
├── app.py                             # Streamlit integration UI
├── main.py                            # CLI demonstration entry point
├── Dockerfile                         # Container image definition
└── requirements.txt                   # Runtime + test dependencies
```

## 🧠 Algorithms Used

### 1) Graph & Data Engine
- Adjacency-list transport graph
- Time-dependent dynamic edge weights
- Structured data loading and validation

### 2) MST Infrastructure Optimization
- Kruskal + Union-Find
- Composite weighting using distance, construction cost, population, and critical facilities

### 3) Dijkstra Traffic Routing
- Dynamic travel cost using time-of-day traffic factors
- Incident simulation via road closures

### 4) A* Emergency Routing
- Heuristic-guided search (Euclidean)
- Emergency edge-weight policy to reduce congestion sensitivity
- Built-in comparison against standard Dijkstra

### 5) DP Transit Optimization
- Bottom-up dynamic programming for global fleet allocation
- Utility-maximizing redistribution with per-route analysis

### 6) ML Traffic Prediction (Bonus)
- `RandomForestRegressor` trained on temporal road records
- Predicts congestion factor by road segment and time of day

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

App URL: `http://localhost:8501`

## 🧪 Run Tests

```bash
python -m pytest tests/ -v
```

## 🐳 Docker Deployment

### Build image
```bash
docker build -t smart-city-its .
```

### Run container
```bash
docker run --rm -p 8501:8501 smart-city-its
```

Then open `http://localhost:8501`.

## 📊 UI Modules

- **Infrastructure Planning:** run MST and inspect selected edge set
- **Traffic Routing:** choose source, destination, and time-of-day routing
- **Emergency Dispatch:** compare Dijkstra vs A* (runtime, steps, cost)
- **Transit Optimization:** optimize bus allocations with configurable extra fleet
- **AI Prediction:** forecast congestion for a chosen road segment and time slot

## 🤝 Contributors

Core algorithm modules were implemented by the project team (Members 1–5), and the final integration/UI/bonus delivery is provided through Streamlit + ML + deployment packaging.
