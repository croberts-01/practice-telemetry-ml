import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import MODELS_DIR, PREDICTIONS_DIR, PROCESSED_DIR


FEATURES_PATH = PROCESSED_DIR / "telemetry_features.parquet"

MODEL_PATH = MODELS_DIR / "iso_time_split.joblib"
META_PATH = MODELS_DIR / "iso_time_split_meta.joblib"
THRESH_PATH = MODELS_DIR / "iso_time_split_threshold.json"

PRED_PATH = PREDICTIONS_DIR / "batch_predictions_time_split.parquet"


@dataclass
class SplitConfig:
    train_quantile: float = 0.70
    calib_quantile: float = 0.85
    alert_rate: float = 0.01


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric features excluding identifiers/debug fields."""
    non_feature_cols = {"device_id", "timestamp", "event_tag"}
    feature_cols = [
        c for c in df.columns
        if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found in feature table.")
    return feature_cols


def _split_by_time(df: pd.DataFrame, train_q: float, calib_q: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Split df into train / calibration / score windows using timestamp quantiles.
    Returns splits + a dict with boundary timestamps for metadata.
    """
    if "timestamp" not in df.columns:
        raise ValueError("Feature table must contain 'timestamp' column.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    train_end = df["timestamp"].quantile(train_q)
    calib_end = df["timestamp"].quantile(calib_q)

    train_df = df[df["timestamp"] <= train_end].copy()
    calib_df = df[(df["timestamp"] > train_end) & (df["timestamp"] <= calib_end)].copy()
    score_df = df[df["timestamp"] > calib_end].copy()

    boundaries = {
        "t_min": df["timestamp"].min().isoformat(),
        "t_max": df["timestamp"].max().isoformat(),
        "train_end": pd.Timestamp(train_end).isoformat(),
        "calib_end": pd.Timestamp(calib_end).isoformat(),
        "train_quantile": train_q,
        "calib_quantile": calib_q,
    }
    return train_df, calib_df, score_df, boundaries


def train_calibrate_score(
    split_cfg: SplitConfig = SplitConfig(),
    contamination: float = 0.02,
    n_estimators: int = 200,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    End-to-end batch job:
      1) read engineered feature table
      2) time-split into train/calib/score
      3) train IsolationForest on train
      4) calibrate percentile threshold on calib
      5) score + alert on score window
      6) save model, meta, threshold metadata, and predictions
    Returns a dict summary.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEATURES_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    feature_cols = _get_feature_columns(df)

    # Split windows
    train_df, calib_df, score_df, boundaries = _split_by_time(
        df,
        train_q=split_cfg.train_quantile,
        calib_q=split_cfg.calib_quantile,
    )

    # Drop any rows missing required features (safety)
    train_before, calib_before, score_before = len(train_df), len(calib_df), len(score_df)
    train_df = train_df.dropna(subset=feature_cols).reset_index(drop=True)
    calib_df = calib_df.dropna(subset=feature_cols).reset_index(drop=True)
    score_df = score_df.dropna(subset=feature_cols).reset_index(drop=True)

    # Train model on historical window
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_df[feature_cols])

    # Calibrate threshold on calibration window
    calib_scores = model.decision_function(calib_df[feature_cols])
    calib_df = calib_df.copy()
    calib_df["anomaly_score"] = calib_scores

    threshold = float(calib_df["anomaly_score"].quantile(split_cfg.alert_rate))

    # Score "future" window and apply threshold
    score_df = score_df.copy()
    score_df["anomaly_score"] = model.decision_function(score_df[feature_cols])
    score_df["is_alert"] = score_df["anomaly_score"] <= threshold

    # Write predictions (lean output)
    output_cols = [
        "device_id", "timestamp",
        "sensor_value", "temp_c", "rssi",
        "event_tag",
        "anomaly_score", "is_alert",
    ]
    # Some columns might not exist if you later change features; guard it:
    output_cols = [c for c in output_cols if c in score_df.columns]

    score_df[output_cols].to_parquet(PRED_PATH, index=False)

    # Save model artifact + meta contract
    meta = {
        "model_type": "IsolationForest",
        "feature_cols": feature_cols,
        "contamination": contamination,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "split_cfg": asdict(split_cfg),
        "boundaries": boundaries,
        "created_at_utc": datetime.utcnow().isoformat(),
        "prediction_path": str(PRED_PATH),
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(meta, META_PATH)

    # Save threshold separately (human-readable)
    thresh_payload = {
        "threshold": threshold,
        "alert_rate_target": split_cfg.alert_rate,
        "actual_alert_rate_score_window": float(score_df["is_alert"].mean()) if len(score_df) else None,
        "boundaries": boundaries,
        "created_at_utc": meta["created_at_utc"],
    }
    THRESH_PATH.write_text(json.dumps(thresh_payload, indent=2))

    # Summary stats (useful logs)
    summary = {
        "rows_total": len(df),
        "rows_train_raw": train_before,
        "rows_calib_raw": calib_before,
        "rows_score_raw": score_before,
        "rows_train_used": len(train_df),
        "rows_calib_used": len(calib_df),
        "rows_score_used": len(score_df),
        "num_features": len(feature_cols),
        "threshold": threshold,
        "alert_rate_target": split_cfg.alert_rate,
        "actual_alert_rate_score_window": float(score_df["is_alert"].mean()) if len(score_df) else None,
        "paths": {
            "model": str(MODEL_PATH),
            "meta": str(META_PATH),
            "threshold": str(THRESH_PATH),
            "predictions": str(PRED_PATH),
        },
        "boundaries": boundaries,
        "alerts_by_event_tag": (
            score_df.groupby("event_tag")["is_alert"].mean().sort_values(ascending=False).to_dict()
            if "event_tag" in score_df.columns and len(score_df)
            else {}
        ),
    }
    return summary


def main() -> None:
    print("🚀 Starting time-split train → calibrate → score job...")

    summary = train_calibrate_score(
        split_cfg=SplitConfig(train_quantile=0.70, calib_quantile=0.85, alert_rate=0.01),
        contamination=0.02,
        n_estimators=200,
        random_state=42,
    )

    print("✅ Job complete")
    print(f"Total rows: {summary['rows_total']:,}")
    print(f"Train used: {summary['rows_train_used']:,} | Calib used: {summary['rows_calib_used']:,} | Score used: {summary['rows_score_used']:,}")
    print(f"Num features: {summary['num_features']}")
    print(f"Threshold (calib quantile): {summary['threshold']:.6f}")
    print(f"Alert rate target: {summary['alert_rate_target']:.2%}")
    ar = summary["actual_alert_rate_score_window"]
    if ar is not None:
        print(f"Actual alert rate (score window): {ar:.2%}")
    if summary["alerts_by_event_tag"]:
        print(f"Alerts by event_tag (score window): {summary['alerts_by_event_tag']}")
    print(f"Saved predictions: {summary['paths']['predictions']}")
    print(f"Saved model: {summary['paths']['model']}")
    print(f"Saved meta: {summary['paths']['meta']}")
    print(f"Saved threshold: {summary['paths']['threshold']}")


if __name__ == "__main__":
    main()