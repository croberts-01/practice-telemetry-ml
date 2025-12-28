import numpy as np
import pandas as pd
from src.config import RAW_TELEMETRY_PATH, RAW_DIR

def generate_synthetic_telemetry(
    n_devices: int = 25,
    days: int = 14,
    freq_minutes: int = 15,
    seed: int = 42,
    end_timestamp: str | None = None,  # ISO string or None for "now"
) -> pd.DataFrame:
    """
    Generate synthetic telemetry for multiple devices with realistic patterns:
    - daily seasonality
    - noise
    - occasional drift, spikes, step changes, variance windows
    Includes a small amount of duplicates and nulls for cleaning practice.
    """
    rng = np.random.default_rng(seed)

    periods = int((24 * 60 / freq_minutes) * days)

    end_ts = pd.Timestamp.now().floor("min") if end_timestamp is None else pd.Timestamp(end_timestamp)
    timestamps = pd.date_range(
        end=end_ts,
        periods=periods,
        freq=f"{freq_minutes}min",
    )

    rows = []
    for d in range(n_devices):
        device_id = f"D{d:03d}"

        base = rng.normal(loc=10 + d * 0.05, scale=1.5)

        t = np.arange(periods)
        daily = 1.5 * np.sin(2 * np.pi * t / (24 * 60 / freq_minutes))

        noise = rng.normal(0, 0.6, size=periods)
        sensor_value = base + daily + noise

        event_tag = np.array(["normal"] * periods, dtype=object)

        # slow drift
        if rng.random() < 0.25:
            sensor_value = sensor_value + np.linspace(0, rng.uniform(2, 6), periods)

        # spikes
        for _ in range(rng.integers(2, 7)):
            idx = rng.integers(10, periods - 10)
            sensor_value[idx] += rng.uniform(8, 20)
            event_tag[idx] = "spike"

        # step change
        if rng.random() < 0.35:
            idx = rng.integers(int(periods * 0.3), int(periods * 0.8))
            sensor_value[idx:] += rng.uniform(2, 8)
            event_tag[idx: idx + 5] = "step"

        # variance window
        if rng.random() < 0.35:
            idx = rng.integers(int(periods * 0.2), int(periods * 0.9))
            window = rng.integers(10, 30)
            sensor_value[idx: idx + window] += rng.normal(0, 3.0, size=window)
            event_tag[idx: idx + window] = "variance"

        # extra signals
        temp_c = 20 + 5 * np.sin(2 * np.pi * t / (24 * 60 / freq_minutes)) + rng.normal(0, 0.8, size=periods)
        battery_v = 3.9 - np.linspace(0, rng.uniform(0.0, 0.25), periods) + rng.normal(0, 0.01, size=periods)
        rssi = -55 + rng.normal(0, 3, size=periods)

        df_d = pd.DataFrame(
            {
                "device_id": device_id,
                "timestamp": timestamps,
                "sensor_value": sensor_value.astype(float),
                "temp_c": temp_c.astype(float),
                "battery_v": battery_v.astype(float),
                "rssi": rssi.astype(float),
                "event_tag": event_tag,
            }
        )
        rows.append(df_d)

    df = pd.concat(rows, ignore_index=True)

    # Add some "messy data" for cleaning practice
    if len(df) > 0:
        dup = df.sample(frac=0.002, random_state=seed)
        df = pd.concat([df, dup], ignore_index=True)

        null_idx = df.sample(frac=0.001, random_state=seed + 1).index
        df.loc[null_idx, "sensor_value"] = np.nan

    return df


def save_raw_telemetry(
    n_devices: int = 25,
    days: int = 14,
    freq_minutes: int = 15,
    seed: int = 42,
    end_timestamp: str | None = None,
) -> pd.DataFrame:
    """
    Generate telemetry and write it to data/raw/telemetry_raw.parquet.
    Returns the generated dataframe for optional downstream use.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_telemetry(
        n_devices=n_devices,
        days=days,
        freq_minutes=freq_minutes,
        seed=seed,
        end_timestamp=end_timestamp,
    )

    df.to_parquet(RAW_TELEMETRY_PATH, index=False)

    # lightweight logging
    print("✅ Wrote raw telemetry")
    print(f"Path: {RAW_TELEMETRY_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Devices: {df['device_id'].nunique()}")
    print(f"Time range: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    return df

def main() -> None:
    save_raw_telemetry()


if __name__ == "__main__":
    main()

