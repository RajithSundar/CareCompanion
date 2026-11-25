"""
CareCompanion — Supervised ML Training Pipeline for PMData Fitbit + PMSys Dataset
Trains and compares RandomForest, GradientBoosting, and LogisticRegression to predict
daily recovery_state ('Ready', 'Moderate', 'Deload Needed') from Fitbit biometrics.
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "pmdata_raw"
SAMPLE_CSV = BASE_DIR / "data" / "pmdata_sample.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "steps",
    "resting_heart_rate",
    "sleep_score",
    "deep_sleep_min",
    "rem_sleep_min",
    "sedentary_min",
    "very_active_min",
    "cardio_min",
    "peak_min",
    "rhr_delta",
    "sleep_3d_avg",
    "steps_3d_avg",
]


def load_raw_pmdata_if_available() -> pd.DataFrame | None:
    """Ingest raw Kaggle PMData participant folders (p01..p16) if present."""
    if not RAW_DIR.exists():
        return None
    participant_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir() and d.name.startswith("p")])
    if not participant_dirs:
        return None

    records = []
    for pdir in participant_dirs:
        fitbit_dir = pdir / "fitbit"
        pmsys_wellness = pdir / "pmsys" / "wellness.csv"
        sleep_csv = fitbit_dir / "sleep_score.csv"
        rhr_json = fitbit_dir / "resting_heart_rate.json"
        if not (pmsys_wellness.exists() and sleep_csv.exists() and rhr_json.exists()):
            continue

        wellness_df = pd.read_csv(pmsys_wellness)
        wellness_df["date"] = pd.to_datetime(wellness_df["effective_time_frame"]).dt.date.astype(str)

        sleep_df = pd.read_csv(sleep_csv)
        sleep_df["date"] = pd.to_datetime(sleep_df["timestamp"]).dt.date.astype(str)
        sleep_df = sleep_df.rename(columns={"overall_score": "sleep_score", "deep_sleep_in_minutes": "deep_sleep_min"})

        with open(rhr_json, "r", encoding="utf-8") as f:
            rhr_raw = json.load(f)
        rhr_rows = [
            {
                "date": pd.to_datetime(item["dateTime"]).date().isoformat(),
                "resting_heart_rate": item["value"].get("value", np.nan),
            }
            for item in rhr_raw
            if isinstance(item.get("value"), dict)
        ]
        rhr_df = pd.DataFrame(rhr_rows).dropna()

        merged = wellness_df.merge(sleep_df[["date", "sleep_score", "deep_sleep_min"]], on="date", how="inner")
        merged = merged.merge(rhr_df, on="date", how="inner")
        merged["participant_id"] = pdir.name
        for col, default in [
            ("steps", 7500), ("rem_sleep_min", 85), ("sedentary_min", 650),
            ("very_active_min", 20), ("cardio_min", 18), ("peak_min", 4)
        ]:
            if col not in merged.columns:
                merged[col] = default

        merged["recovery_state"] = np.where(
            (merged["readiness"] >= 7) & (merged["fatigue"] <= 3),
            "Ready",
            np.where((merged["readiness"] <= 4) | (merged["fatigue"] >= 4), "Deload Needed", "Moderate"),
        )
        records.append(merged)

    if not records:
        return None
    return pd.concat(records, ignore_index=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["participant_id", "date"]).reset_index(drop=True)
    df["rhr_7d_baseline"] = (
        df.groupby("participant_id")["resting_heart_rate"]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
    )
    df["rhr_delta"] = df["resting_heart_rate"] - df["rhr_7d_baseline"]
    df["sleep_3d_avg"] = (
        df.groupby("participant_id")["sleep_score"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    df["steps_3d_avg"] = (
        df.groupby("participant_id")["steps"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )
    return df


def main():
    raw_df = load_raw_pmdata_if_available()
    if raw_df is not None:
        print(f"Loaded raw Kaggle PMData: {len(raw_df)} rows")
        df = raw_df
        dataset_source = "Kaggle Raw PMData"
    else:
        print(f"Using PMData-schema sample dataset: {SAMPLE_CSV}")
        df = pd.read_csv(SAMPLE_CSV)
        dataset_source = "PMData-Schema Sample (480 rows, 16 participants)"

    df = engineer_features(df)
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    y = df["recovery_state"]
    groups = df["participant_id"]

    # Group split by participant_id to prevent rolling-window leakage across adjacent days
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.06, max_depth=4, random_state=42
        ),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
    }

    cv = StratifiedGroupKFold(n_splits=5)
    comparison = []
    best_name, best_model, best_cv_f1 = None, None, -1.0

    for name, model in candidates.items():
        cv_f1 = cross_val_score(
            model, X_train, y_train, groups=groups_train, cv=cv, scoring="f1_weighted"
        ).mean()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        test_acc = accuracy_score(y_test, preds)
        test_f1 = f1_score(y_test, preds, average="weighted")
        comparison.append({
            "model": name,
            "cv_f1_weighted": round(float(cv_f1), 4),
            "test_accuracy": round(float(test_acc), 4),
            "test_f1_weighted": round(float(test_f1), 4),
        })
        print(f"[{name}] 5-Fold Group CV F1={cv_f1:.4f} | Test Acc={test_acc:.4f} | Test F1={test_f1:.4f}")
        if cv_f1 > best_cv_f1:
            best_cv_f1 = cv_f1
            best_name = name
            best_model = model

    print(f"\nSelected Best Model: {best_name} (5-Fold Group CV F1={best_cv_f1:.4f})")
    print(classification_report(y_test, best_model.predict(X_test)))

    # Extract feature importances
    importances = {}
    if hasattr(best_model, "feature_importances_"):
        for col, imp in zip(FEATURE_COLS, best_model.feature_importances_):
            importances[col] = round(float(imp), 4)
    elif hasattr(best_model, "named_steps"):
        coef = np.abs(best_model.named_steps["clf"].coef_).mean(axis=0)
        coef_norm = coef / (coef.sum() + 1e-9)
        for col, imp in zip(FEATURE_COLS, coef_norm):
            importances[col] = round(float(imp), 4)

    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_cols": FEATURE_COLS,
        "classes": list(best_model.classes_),
    }
    joblib.dump(artifact, MODELS_DIR / "readiness_model.joblib")

    metrics_payload = {
        "dataset_source": dataset_source,
        "total_rows": int(len(df)),
        "selected_model": best_name,
        "comparison": comparison,
        "feature_importances": importances,
    }
    with open(MODELS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Saved model to {MODELS_DIR / 'readiness_model.joblib'}")


if __name__ == "__main__":
    main()
