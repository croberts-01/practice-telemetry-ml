import joblib
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, PREDICTIONS_DIR


FEATURES_PATH = PROCESSED_DIR / "telemetry_features.parquet"
MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"
META_PATH = MODELS_DIR / "model_meta.joblib"

OUTPUT_PATH = PREDICTIONS_DIR / "batch_predictions.parquet"


def score_batch(
    alert_rate: float = 0.01,
    output_path=OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Batch scoring job:
      - loads feature table
      - loads model + meta (feature column contract)
      - computes anomaly_score via decision_function
      - sets is_alert using a percentile threshold (bottom alert_rate)
      - writes predictions parquet
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)

    df_feat = pd.read_parquet(FEATURES_PATH)
    feature_cols = meta["feature_cols"]

    # Drop rows missing required features (safety)
    before = len(df_feat)
    df = df_feat.dropna(subset=feature_cols).copy()
    dropped = before - len(df)

    if dropped > 0:
        print(f"Dropped {dropped:,} rows with NaNs in required features before scoring.")

    X = df[feature_cols]
    df["anomaly_score"] = model.decision_function(X)

    # Percentile-based threshold: alert on bottom X%
    threshold = df["anomaly_score"].quantile(alert_rate)
    df["is_alert"] = df["anomaly_score"] <= threshold

    # Save a lean output table (good for downstream systems)
    output_cols = [
        "device_id", "timestamp",
        "sensor_value", "temp_c", "rssi",
        "event_tag",
        "anomaly_score", "is_alert",
    ]

    df_out = df[output_cols].copy()
    df_out.to_parquet(output_path, index=False)

    print("Batch scoring complete")
    print(f"Output: {output_path}")
    print(f"Rows scored: {len(df_out):,}")
    print(f"Alert rate target: {alert_rate:.2%}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Actual alert rate: {df_out['is_alert'].mean():.2%}")

    return df_out


def main() -> None:
    score_batch(alert_rate=0.01)


if __name__ == "__main__":
    main()
