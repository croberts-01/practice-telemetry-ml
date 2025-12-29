import pandas as pd

from src.config import PREDICTIONS_DIR

PRED_PATH = PREDICTIONS_DIR / "batch_predictions_time_split.parquet"
MON_DIR = PREDICTIONS_DIR / "monitoring"

ALERTS_BY_DAY_PATH = MON_DIR / "alerts_by_day.parquet"
ALERTS_BY_DEVICE_PATH = MON_DIR / "alerts_by_device.parquet"
SCORE_BY_DAY_PATH = MON_DIR / "score_distribution_by_day.parquet"


def build_monitoring_summaries(pred_path=PRED_PATH) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(pred_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day"] = df["timestamp"].dt.date

    # --- Alerts by day (overall) ---
    alerts_by_day = (
        df.groupby("day")
          .agg(
              rows=("is_alert", "size"),
              alerts=("is_alert", "sum"),
              alert_rate=("is_alert", "mean"),
              score_min=("anomaly_score", "min"),
              score_p05=("anomaly_score", lambda s: s.quantile(0.05)),
              score_p50=("anomaly_score", "median"),
              score_p95=("anomaly_score", lambda s: s.quantile(0.95)),
              score_max=("anomaly_score", "max"),
          )
          .reset_index()
          .sort_values("day")
    )

    # --- Alerts by device (overall) ---
    alerts_by_device = (
        df.groupby("device_id")
          .agg(
              rows=("is_alert", "size"),
              alerts=("is_alert", "sum"),
              alert_rate=("is_alert", "mean"),
              score_min=("anomaly_score", "min"),
              score_p05=("anomaly_score", lambda s: s.quantile(0.05)),
              score_p50=("anomaly_score", "median"),
              score_p95=("anomaly_score", lambda s: s.quantile(0.95)),
              score_max=("anomaly_score", "max"),
          )
          .reset_index()
          .sort_values(["alerts", "alert_rate"], ascending=False)
    )

    # --- Score distribution by day (compact quantiles for plotting) ---
    score_by_day = (
        df.groupby("day")["anomaly_score"]
          .quantile([0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
          .reset_index()
          .rename(columns={"level_1": "quantile", "anomaly_score": "value"})
          .sort_values(["day", "quantile"])
    )

    return {
        "alerts_by_day": alerts_by_day,
        "alerts_by_device": alerts_by_device,
        "score_by_day": score_by_day,
    }


def main() -> None:
    MON_DIR.mkdir(parents=True, exist_ok=True)

    summaries = build_monitoring_summaries()

    summaries["alerts_by_day"].to_parquet(ALERTS_BY_DAY_PATH, index=False)
    summaries["alerts_by_device"].to_parquet(ALERTS_BY_DEVICE_PATH, index=False)
    summaries["score_by_day"].to_parquet(SCORE_BY_DAY_PATH, index=False)

    print("Wrote monitoring summaries")
    print(f"- {ALERTS_BY_DAY_PATH}")
    print(f"- {ALERTS_BY_DEVICE_PATH}")
    print(f"- {SCORE_BY_DAY_PATH}")
    print("Preview (alerts_by_day):")
    print(summaries["alerts_by_day"].head())


if __name__ == "__main__":
    main()
