import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import PROCESSED_DIR, MODELS_DIR


FEATURES_PATH = PROCESSED_DIR / "telemetry_features.parquet"
MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"
META_PATH = MODELS_DIR / "model_meta.joblib"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select numeric columns excluding identifiers and debug labels."""
    non_feature_cols = {"device_id", "timestamp", "event_tag"}

    feature_cols = [
        c for c in df.columns
        if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns found. Check your feature table.")

    return feature_cols


def train_isolation_forest(
    df_feat: pd.DataFrame,
    contamination: float = 0.02,
    random_state: int = 42,
    n_estimators: int = 200,
) -> tuple[IsolationForest, dict]:
    """Train a global Isolation Forest model and return model + metadata."""
    feature_cols = get_feature_columns(df_feat)
    X = df_feat[feature_cols]

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)

    meta = {
        "feature_cols": feature_cols,
        "contamination": contamination,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "model_type": "IsolationForest",
    }
    return model, meta


def main() -> None:
    print("Starting Isolation Forest training job...")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_feat = pd.read_parquet(FEATURES_PATH)

    print(f"Loaded feature table with {len(df_feat):,} rows.")

    feature_cols = get_feature_columns(df_feat)

    before = len(df_feat)
    df_feat = df_feat.dropna(subset=feature_cols).reset_index(drop=True)
    dropped = before - len(df_feat)

    if dropped > 0:
        print(f"⚠️ Dropped {dropped:,} rows with NaNs in feature columns before training.")

    model, meta = train_isolation_forest(df_feat)

    print("Saving model artifacts...")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(meta, META_PATH)

    X = df_feat[meta["feature_cols"]]
    scores = model.decision_function(X)

    print("✅ Trained Isolation Forest")
    print(f"Feature rows: {len(df_feat):,}")
    print(f"Num features: {len(meta['feature_cols'])}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved meta:  {META_PATH}")
    print(f"Score range: {scores.min():.4f} -> {scores.max():.4f}")
    print(f"Score mean:  {scores.mean():.4f}")

if __name__ == "__main__":
    main()

