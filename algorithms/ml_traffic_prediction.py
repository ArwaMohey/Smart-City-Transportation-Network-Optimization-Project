"""ML-based traffic congestion prediction utilities.

This module trains a lightweight regression model on temporal road traffic
records and predicts congestion factors for selected road segments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from utils.enums import TimeOfDay


_TIME_INDEX = {
    TimeOfDay.MORNING.value: 0,
    TimeOfDay.AFTERNOON.value: 1,
    TimeOfDay.EVENING.value: 2,
    TimeOfDay.NIGHT.value: 3,
}


@dataclass
class TrafficPredictionResult:
    """Container for a single congestion prediction."""

    from_node: str
    to_node: str
    time_of_day: str
    predicted_congestion_factor: float


class TrafficPredictor:
    """RandomForest-based predictor for road congestion factors."""

    def __init__(self, random_state: int = 42) -> None:
        self._model = RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        )
        self._is_fitted = False

    @classmethod
    def from_edges_file(cls, edges_path: Path | str) -> "TrafficPredictor":
        """Create and train a predictor from ``edges.json``."""
        edges_file = Path(edges_path)
        with edges_file.open(encoding="utf-8") as fh:
            payload = json.load(fh)

        roads = payload.get("current_roads", []) + payload.get("proposed_roads", [])
        predictor = cls()
        predictor.fit(roads)
        return predictor

    def fit(self, road_records: List[Dict[str, Any]]) -> None:
        """Train the model from road records in ``edges.json`` schema."""
        rows: List[Dict[str, float]] = []

        for road in road_records:
            traffic_factors = road.get("traffic_factors", {})
            traffic_flow = road.get("traffic_flow", {})

            for tod, factor in traffic_factors.items():
                if tod not in _TIME_INDEX:
                    continue
                rows.append(
                    {
                        "distance": float(road.get("distance", 0.0)),
                        "base_capacity": float(road.get("base_capacity", 0.0)),
                        "condition_score": float(road.get("condition_score", 7.0)),
                        "traffic_flow": float(traffic_flow.get(tod, 0.0)),
                        "time_index": float(_TIME_INDEX[tod]),
                        "target_factor": float(factor),
                    }
                )

        if not rows:
            raise ValueError("No valid temporal traffic records found for model training.")

        frame = pd.DataFrame(rows)
        x = frame[["distance", "base_capacity", "condition_score", "traffic_flow", "time_index"]]
        y = frame["target_factor"]

        self._model.fit(x, y)
        self._is_fitted = True

    def predict_factor(
        self,
        *,
        distance: float,
        base_capacity: float,
        condition_score: float,
        traffic_flow: float,
        time_of_day: TimeOfDay | str,
    ) -> float:
        """Predict congestion factor for one road segment/time slot."""
        if not self._is_fitted:
            raise RuntimeError("TrafficPredictor is not fitted. Call fit() first.")

        tod_value = time_of_day.value if isinstance(time_of_day, TimeOfDay) else str(time_of_day)
        if tod_value not in _TIME_INDEX:
            valid = ", ".join(_TIME_INDEX.keys())
            raise ValueError(f"Unsupported time_of_day={tod_value!r}. Valid: {valid}.")

        feature_row = pd.DataFrame(
            [
                {
                    "distance": float(distance),
                    "base_capacity": float(base_capacity),
                    "condition_score": float(condition_score),
                    "traffic_flow": float(traffic_flow),
                    "time_index": float(_TIME_INDEX[tod_value]),
                }
            ]
        )

        prediction = float(self._model.predict(feature_row)[0])
        return max(0.1, round(prediction, 3))

    def predict_for_edge(self, edge_record: Dict[str, Any], time_of_day: TimeOfDay | str) -> float:
        """Predict congestion factor for an edge record from ``edges.json``."""
        tod_value = time_of_day.value if isinstance(time_of_day, TimeOfDay) else str(time_of_day)
        traffic_flow = edge_record.get("traffic_flow", {}).get(tod_value, 0.0)

        return self.predict_factor(
            distance=float(edge_record.get("distance", 0.0)),
            base_capacity=float(edge_record.get("base_capacity", 0.0)),
            condition_score=float(edge_record.get("condition_score", 7.0)),
            traffic_flow=float(traffic_flow),
            time_of_day=tod_value,
        )
