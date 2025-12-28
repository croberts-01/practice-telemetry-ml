import pandas as pd

from src.config import RAW_TELEMETRY_PATH, PROCESSED_DIR


FEATURES_PATH = PROCESSED_DIR / "telemetry_features.parquet"


def build_features(
    df: pd.DataFrame,
    feature_cols: tuple[str, ...] = ("sensor_value", "temp_c", "rssi"),
    lags: tuple[int, ...] = (1, 2, 4, 8),
    rolling_windows: tuple[int, ...] = (4, 12, 24, 96),  # 1h, 3h, 6h, 24h @ 15-min
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Build strictly leakage-safe features per device.
    Leakage prevention rule:
      - lag features use shift(k)
      - rolling features use shift(1) THEN rolling(window)

    Keeps event_tag for debugging/evaluation (NOT used for training).
    """
    required = {"device_id", "timestamp", "event_tag", *feature_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    # Hard requirement: sorted by device, time
    out = out.sort_values(["device_id", "timestamp"]).reset_index(drop=True)

    g = out.groupby("device_id", group_keys=False)

    # --- Lag features ---
    for col in feature_cols:
        for k in lags:
            out[f"{col}_lag_{k}"] = g[col].shift(k)

    # --- Rolling features (STRICT: shift(1) first) ---
    for col in feature_cols:
        past = g[col].shift(1)  # leakage prevention

        # rolling() requires a Series grouped by device; we do:
        rolled = past.groupby(out["device_id"])

        for w in rolling_windows:
            r = rolled.rolling(window=w, min_periods=w)

            out[f"{col}_roll_mean_{w}"] = r.mean().reset_index(level=0, drop=True)
            out[f"{col}_roll_std_{w}"] = r.std().reset_index(level=0, drop=True)
            out[f"{col}_roll_min_{w}"] = r.min().reset_index(level=0, drop=True)
            out[f"{col}_roll_max_{w}"] = r.max().reset_index(level=0, drop=True)

    # Example derived feature (safe because it uses only lagged values)
    if "sensor_value_lag_1" in out.columns and "sensor_value_lag_2" in out.columns:
        out["sensor_value_delta_1"] = out["sensor_value_lag_1"] - out["sensor_value_lag_2"]

    # Optionally drop warmup rows where engineered features are NaN
    engineered = [c for c in out.columns if "_lag_" in c or "_roll_" in c or c.endswith("_delta_1")]
    if drop_na:
        out = out.dropna(subset=engineered).reset_index(drop=True)

    return out


def run_feature_job(
    input_path=RAW_TELEMETRY_PATH,
    output_path=FEATURES_PATH,
) -> pd.DataFrame:
    """
    Batch job: read raw -> build features -> write processed parquet.
    Returns the feature dataframe for convenience.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_parquet(input_path)
    df_feat = build_features(df_raw)

    df_feat.to_parquet(output_path, index=False)

    print("✅ Wrote feature table")
    print(f"Path: {output_path}")
    print(f"Rows: {len(df_feat):,}")
    print(f"Devices: {df_feat['device_id'].nunique()}")
    print(f"Time range: {df_feat['timestamp'].min()} -> {df_feat['timestamp'].max()}")
    print(f"Num columns: {df_feat.shape[1]}")

    return df_feat


def main() -> None:
    run_feature_job()


if __name__ == "__main__":
    main()
