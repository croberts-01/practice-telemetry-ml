from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODELS_DIR
from src.features import build_features  # reuse feature logic


# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = MODELS_DIR / "iso_time_split.joblib"
META_PATH = MODELS_DIR / "iso_time_split_meta.joblib"
THRESH_PATH = MODELS_DIR / "iso_time_split_threshold.json"


# -----------------------------
# Request / Response Schemas
# -----------------------------
class TelemetryInput(BaseModel):
    device_id: str
    timestamp: str
    sensor_value: float
    temp_c: float
    battery_v: Optional[float] = None
    rssi: float


class ScoreOutput(BaseModel):
    anomaly_score: float
    is_alert: bool
    threshold: float
    model: str

class TelemetryBatchInput(BaseModel):
    records: list[TelemetryInput]


class BatchScoreResult(BaseModel):
    timestamp: str
    anomaly_score: float | None
    is_alert: bool | None


class BatchScoreOutput(BaseModel):
    results: list[BatchScoreResult]
    threshold: float
    model: str


# -----------------------------
# App + Startup
# -----------------------------
app = FastAPI(
    title="Telemetry Anomaly Scoring API",
    version="1.0.0",
)


@app.on_event("startup")
def load_model():
    global model, meta, threshold

    if not MODEL_PATH.exists():
        raise RuntimeError("Model file not found.")

    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)

    import json
    threshold_payload = json.loads(THRESH_PATH.read_text())
    threshold = threshold_payload["threshold"]


# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Scoring Endpoint
# -----------------------------
@app.post("/score", response_model=ScoreOutput)
def score_telemetry(payload: TelemetryInput):
    """
    Score a single telemetry event for anomaly detection.
    """

    # Convert input to DataFrame
    df_raw = pd.DataFrame([payload.dict()])

    try:
        # Build features using the SAME pipeline as training
        df_feat = build_features(df_raw)

        feature_cols = meta["feature_cols"]
        X = df_feat[feature_cols]

        score = float(model.decision_function(X)[0])
        is_alert = score <= threshold

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ScoreOutput(
        anomaly_score=score,
        is_alert=is_alert,
        threshold=threshold,
        model=meta.get("model_type", "IsolationForest"),
    )


@app.post("/score_batch", response_model=BatchScoreOutput)
def score_telemetry_batch(payload: TelemetryBatchInput):
    """
    Score a batch (window) of telemetry records.
    Returns anomaly_score + is_alert for each record.
    """

    if not payload.records:
        raise HTTPException(status_code=400, detail="records list is empty")

    # Convert input to DataFrame
    df_raw = pd.DataFrame([r.dict() for r in payload.records])

    try:
        # Build features (do NOT drop NaNs here)
        df_feat = build_features(df_raw, drop_na=False)

        feature_cols = meta["feature_cols"]

        results = []

        for idx, row in df_feat.iterrows():
            # If engineered features are missing, we cannot score this row
            if row[feature_cols].isna().any():
                results.append(
                    BatchScoreResult(
                        timestamp=str(row["timestamp"]),
                        anomaly_score=None,
                        is_alert=None,
                    )
                )
                continue

            X = row[feature_cols].to_frame().T
            score = float(model.decision_function(X)[0])
            is_alert = score <= threshold

            results.append(
                BatchScoreResult(
                    timestamp=str(row["timestamp"]),
                    anomaly_score=score,
                    is_alert=is_alert,
                )
            )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return BatchScoreOutput(
        results=results,
        threshold=threshold,
        model=meta.get("model_type", "IsolationForest"),
    )
