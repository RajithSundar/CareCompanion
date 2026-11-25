"""
CareCompanion: supervised training pipeline for next-day recovery forecasting.

Task
----
Given the biometrics recorded for participant p on day t, predict the recovery
state on day t+1 (Ready / Moderate / Deload Needed).

Why a forecast?
---------------
The recovery_state label is itself a fixed rule on same-day readiness and fatigue
(see `label_recovery_state`). Predicting the same day's label from biometrics only
measures how well the model reproduces that rule. Predicting the next day's label
asks whether today's biometrics carry information about tomorrow's state.

Evaluation
----------
* Nested leave-one-participant-out (LOPO) validation. The outer loop holds out one
  participant at a time. The inner loop tunes hyperparameters with grouped CV on the
  remaining participants only, so the held-out participant never influences tuning.
* Pooled out-of-fold predictions are scored with macro F1, per-class precision and
  recall, a confusion matrix and a multi-class Brier score. Per-participant accuracy
  shows the spread.
* Baselines: majority class, and persistence (tomorrow = today's state).
* Permutation importance is computed on each held-out participant and averaged.

Leakage controls
----------------
* Rolling features use only the current and past days of the same participant.
* Imputation sits inside the sklearn Pipeline, so it is fit on training folds only.
* Today's readiness and fatigue define today's label, not tomorrow's, so they are
  valid inputs for a next-day forecast. The persistence baseline uses them too,
  which makes the comparison fair.
* Rolling windows are calendar-based (see features.py), shared with serving.
* Features the app does not store (sedentary and very-active minutes) are excluded,
  so the trained model sees exactly the inputs the app can supply.
"""

import json
import platform
from importlib.metadata import version as pkg_version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import BASE_COLS, FEATURE_COLS, engineer_features

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "pmdata_raw"
SAMPLE_CSV = BASE_DIR / "data" / "pmdata_sample.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "next_day_recovery_state"
MISSING_FILES: list[str] = []  # Fitbit files absent from the raw export, recorded in the metrics
CLASSES = ["Deload Needed", "Moderate", "Ready"]

# FEATURE_COLS and engineer_features live in features.py so serving uses the same code.


def label_recovery_state(df: pd.DataFrame) -> pd.Series:
    """Same-day rule used to create the labels. Kept explicit for the write-up."""
    return pd.Series(
        np.where(
            (df["readiness"] >= 7) & (df["fatigue"] <= 3),
            "Ready",
            np.where((df["readiness"] <= 4) | (df["fatigue"] >= 4), "Deload Needed", "Moderate"),
        ),
        index=df.index,
    )


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fitbit_date(value) -> str:
    return pd.to_datetime(value, format="mixed").date().isoformat()


def _load_participant(pdir: Path) -> pd.DataFrame | None:
    """One PMData participant folder -> one row per day with the canonical columns.

    Files used (PMData layout):
      pmsys/wellness.csv                     readiness, fatigue, soreness, stress (labels + inputs)
      fitbit/sleep_score.csv                 overall_score, deep_sleep_in_minutes
      fitbit/resting_heart_rate.json         daily resting heart rate (0 means missing)
      fitbit/steps.json                      per-minute steps, summed per day
      fitbit/sleep.json                      REM minutes from the main sleep's stage summary
      fitbit/time_in_heart_rate_zones.json   zone 2 (cardio) and zone 3 (peak) minutes
    wellness.csv is required: it holds the labels. A missing Fitbit file leaves its
    columns NaN (imputed inside the model pipeline) and is recorded in
    MISSING_FILES, so the metrics say exactly what was absent. Values are never
    filled with constants here.
    """
    wellness_csv = pdir / "pmsys" / "wellness.csv"
    fitbit = pdir / "fitbit"
    if not wellness_csv.exists():
        raise FileNotFoundError(f"{pdir.name}: missing pmsys/wellness.csv")
    required = {
        "sleep_score": fitbit / "sleep_score.csv",
        "rhr": fitbit / "resting_heart_rate.json",
        "steps": fitbit / "steps.json",
        "sleep": fitbit / "sleep.json",
        "zones": fitbit / "time_in_heart_rate_zones.json",
    }
    present = {k: p.exists() for k, p in required.items()}
    for k, p in required.items():
        if not present[k]:
            MISSING_FILES.append(f"{pdir.name}/{p.relative_to(pdir)}")

    wellness = pd.read_csv(wellness_csv)
    wellness["date"] = pd.to_datetime(wellness["effective_time_frame"], utc=True, format="mixed").dt.date.astype(str)
    wellness = wellness.groupby("date", as_index=False)[["readiness", "fatigue", "soreness", "stress"]].mean()

    empty = pd.DataFrame()
    sleep_score = pd.read_csv(required["sleep_score"]) if present["sleep_score"] else empty
    if not sleep_score.empty:
        sleep_score["date"] = pd.to_datetime(sleep_score["timestamp"], utc=True, format="mixed").dt.date.astype(str)
        sleep_score = sleep_score.rename(columns={"overall_score": "sleep_score", "deep_sleep_in_minutes": "deep_sleep_min"})
        sleep_score = sleep_score.groupby("date", as_index=False)[["sleep_score", "deep_sleep_min"]].mean()

    rhr = pd.DataFrame([
        {"date": _fitbit_date(item["dateTime"]), "resting_heart_rate": item["value"].get("value")}
        for item in (_read_json(required["rhr"]) if present["rhr"] else []) if isinstance(item.get("value"), dict)
    ])
    if not rhr.empty:
        rhr["resting_heart_rate"] = pd.to_numeric(rhr["resting_heart_rate"], errors="coerce").replace(0, np.nan)
        rhr = rhr.groupby("date", as_index=False)["resting_heart_rate"].mean()

    steps = pd.DataFrame(_read_json(required["steps"])) if present["steps"] else empty
    if not steps.empty:
        steps["date"] = pd.to_datetime(steps["dateTime"], format="mixed").dt.date.astype(str)
        steps["value"] = pd.to_numeric(steps["value"], errors="coerce")
        steps = steps.groupby("date", as_index=False)["value"].sum().rename(columns={"value": "steps"})

    rem_rows = []
    for log in (_read_json(required["sleep"]) if present["sleep"] else []):
        if not log.get("mainSleep", True):
            continue
        summary = (log.get("levels") or {}).get("summary") or {}
        rem = (summary.get("rem") or {}).get("minutes")
        rem_rows.append({"date": str(log.get("dateOfSleep")), "rem_sleep_min": rem})
    rem = pd.DataFrame(rem_rows)
    if not rem.empty:
        rem = rem.groupby("date", as_index=False)["rem_sleep_min"].max()

    zone_rows = []
    for item in (_read_json(required["zones"]) if present["zones"] else []):
        zones = (item.get("value") or {}).get("valuesInZones") or {}
        zone_rows.append({
            "date": _fitbit_date(item["dateTime"]),
            "cardio_min": zones.get("IN_DEFAULT_ZONE_2"),
            "peak_min": zones.get("IN_DEFAULT_ZONE_3"),
        })
    zones = pd.DataFrame(zone_rows)
    if not zones.empty:
        zones = zones.groupby("date", as_index=False)[["cardio_min", "peak_min"]].sum()

    merged = wellness
    for part in (sleep_score, rhr, steps, rem, zones):
        if not part.empty:
            merged = merged.merge(part, on="date", how="left")
    for col in BASE_COLS:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged.dropna(subset=["readiness", "fatigue"])
    if merged.empty:
        return None
    merged["participant_id"] = pdir.name
    merged["recovery_state"] = label_recovery_state(merged)
    return merged


def load_raw_pmdata_if_available() -> pd.DataFrame | None:
    """Ingest raw PMData participant folders (p01..p16) from data/pmdata_raw/ if present."""
    if not RAW_DIR.exists():
        return None
    participant_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir() and d.name.startswith("p"))
    records = [r for r in (_load_participant(p) for p in participant_dirs) if r is not None]
    if not records:
        return None
    return pd.concat(records, ignore_index=True)


def add_next_day_target(df: pd.DataFrame) -> pd.DataFrame:
    """Pair each day's features with the next day's state. Drops each participant's last day."""
    df = df.copy()
    grouped = df.groupby("participant_id")
    df["today_state"] = df["recovery_state"]
    df[TARGET] = grouped["recovery_state"].shift(-1)
    # Only keep rows where the next calendar day exists in the data.
    next_date = grouped["date"].shift(-1)
    df["next_date"] = next_date
    consecutive = (df["next_date"] - df["date"]) == pd.Timedelta(days=1)
    return df[df[TARGET].notna() & consecutive].reset_index(drop=True)


def make_candidates():
    """Each candidate: a pipeline plus the grid searched inside every training fold."""
    return {
        "RandomForest": (
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
            ]),
            {"clf__max_depth": [4, 8, None], "clf__min_samples_leaf": [1, 5]},
        ),
        "GradientBoosting": (
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("clf", GradientBoostingClassifier(n_estimators=150, random_state=42)),
            ]),
            {"clf__learning_rate": [0.03, 0.06], "clf__max_depth": [2, 4]},
        ),
        "LogisticRegression": (
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
    }


def tune(pipe, grid, X, y, groups):
    """Inner grouped CV. Returns a fitted GridSearchCV on the given rows only."""
    n_groups = len(np.unique(groups))
    inner = GroupKFold(n_splits=min(3, n_groups))
    search = GridSearchCV(pipe, grid, cv=inner, scoring="f1_macro", n_jobs=1)
    search.fit(X, y, groups=groups)
    return search


def brier_multiclass(y_true, proba, classes):
    onehot = np.array([[1.0 if t == c else 0.0 for c in classes] for t in y_true])
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def summarise(y_true, y_pred, classes):
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    per_class = {
        c: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
            "f1": round(float(f[i]), 4), "support": int(support[i])}
        for i, c in enumerate(classes)
    }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)), 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }


def main():
    raw_df = load_raw_pmdata_if_available()
    if raw_df is not None:
        print(f"Loaded raw PMData: {len(raw_df)} rows")
        df = raw_df
        dataset_source = "PMData (raw participant folders)"
    else:
        print(f"Using PMData-schema sample dataset: {SAMPLE_CSV}")
        df = pd.read_csv(SAMPLE_CSV)
        df["recovery_state"] = label_recovery_state(df)
        dataset_source = f"PMData-schema synthetic sample ({len(df)} rows, {df['participant_id'].nunique()} participants)"

    df = add_next_day_target(engineer_features(df))
    raw_missing = df[FEATURE_COLS].isna().mean()
    X = df[FEATURE_COLS]
    y = df[TARGET]
    today = df["today_state"]
    groups = df["participant_id"].to_numpy()
    classes = CLASSES
    print(f"Forecast rows: {len(df)} across {len(np.unique(groups))} participants")

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, groups))
    candidates = make_candidates()

    # ----- Nested LOPO evaluation -----
    oof = {name: np.empty(len(y), dtype=object) for name in candidates}
    oof_proba = {name: np.zeros((len(y), len(classes))) for name in candidates}
    chosen_params = {name: [] for name in candidates}
    perm_rows = {name: [] for name in candidates}
    majority_pred = np.empty(len(y), dtype=object)

    for fold_i, (tr, te) in enumerate(folds, start=1):
        held_out = np.unique(groups[te])[0]
        print(f"[fold {fold_i}/{len(folds)}] held out {held_out}")
        dummy = DummyClassifier(strategy="most_frequent").fit(X.iloc[tr], y.iloc[tr])
        majority_pred[te] = dummy.predict(X.iloc[te])
        for name, (pipe, grid) in candidates.items():
            search = tune(pipe, grid, X.iloc[tr], y.iloc[tr], groups[tr])
            best = search.best_estimator_
            chosen_params[name].append(search.best_params_)
            oof[name][te] = best.predict(X.iloc[te])
            proba = best.predict_proba(X.iloc[te])
            order = [list(best.classes_).index(c) for c in classes]
            oof_proba[name][te] = proba[:, order]
            pi = permutation_importance(
                best, X.iloc[te], y.iloc[te], scoring="f1_macro", n_repeats=5, random_state=42
            )
            perm_rows[name].append(pi.importances_mean)

    comparison = []
    for name in candidates:
        summary = summarise(y, oof[name], classes)
        per_part = [
            accuracy_score(y.iloc[te], oof[name][te]) for _, te in folds
        ]
        comparison.append({
            "model": name,
            **summary,
            "brier": round(brier_multiclass(y.to_numpy(), oof_proba[name], classes), 4),
            "per_participant_accuracy_mean": round(float(np.mean(per_part)), 4),
            "per_participant_accuracy_std": round(float(np.std(per_part)), 4),
            "chosen_params_per_fold": chosen_params[name],
        })

    baselines = {
        "majority_class": summarise(y, majority_pred, classes),
        "persistence_today_state": summarise(y, today.to_numpy(), classes),
    }
    print(json.dumps({"comparison": [(c["model"], c["macro_f1"]) for c in comparison], "baselines": {k: v["macro_f1"] for k, v in baselines.items()}}, indent=2))

    persistence_f1 = baselines["persistence_today_state"]["macro_f1"]
    for row in comparison:
        row["beats_persistence_macro_f1"] = bool(row["macro_f1"] > persistence_f1)
    best_row = max(comparison, key=lambda c: c["macro_f1"])
    best_name = best_row["model"]
    print(f"\nSelected by pooled LOPO macro F1: {best_name} ({best_row['macro_f1']:.4f})")

    # ----- Permutation importance, averaged across held-out participants -----
    pi_mean = np.mean(np.vstack(perm_rows[best_name]), axis=0)
    pi_std = np.std(np.vstack(perm_rows[best_name]), axis=0)
    importances = {
        col: {"mean": round(float(m), 4), "std": round(float(s), 4)}
        for col, m, s in sorted(zip(FEATURE_COLS, pi_mean, pi_std), key=lambda t: -t[1])
    }

    # ----- Final model: tune on all data, then fit -----
    pipe, grid = candidates[best_name]
    final = tune(pipe, grid, X, y, groups).best_estimator_
    artifact = {
        "model": final,
        "model_name": best_name,
        "target": TARGET,
        "horizon_days": 1,
        "feature_cols": FEATURE_COLS,
        "classes": classes,
    }
    joblib.dump(artifact, MODELS_DIR / "readiness_model.joblib")

    metrics = {
        "task": "Next-day recovery state forecast",
        "dataset_source": dataset_source,
        "rows": int(len(df)),
        "participants": int(len(np.unique(groups))),
        "label_rule": "Ready if readiness>=7 and fatigue<=3; Deload Needed if readiness<=4 or fatigue>=4; else Moderate. Labels are rule-based, not clinical ground truth.",
        "validation": "Nested leave-one-participant-out. Inner grouped 3-fold CV tunes hyperparameters per outer fold.",
        "selected_model": best_name,
        "selection_metric": "macro_f1 (pooled out-of-fold)",
        "class_counts": {c: int(v) for c, v in y.value_counts().reindex(classes, fill_value=0).items()},
        "comparison": comparison,
        "baselines": baselines,
        "selected_per_class": comparison[[c["model"] for c in comparison].index(best_name)]["per_class"],
        "feature_importances": importances,
        "feature_importance_method": "permutation (macro F1 drop) on each held-out participant, mean and std across participants",
        "excluded_features": ["sedentary_min", "very_active_min"],
        "excluded_reason": "The app does not store these fields, so the model is trained only on inputs the app can supply.",
        "calibration": "Not calibrated. Brier score is reported so probabilities can be judged.",
        "missing_source_files": MISSING_FILES,
        "feature_missing_rate": {c: round(float(raw_missing[c]), 4) for c in FEATURE_COLS},
        "library_versions": {
            "python": platform.python_version(),
            **{p: pkg_version(p) for p in ["scikit-learn", "pandas", "numpy", "joblib"]},
        },
    }
    with open(MODELS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {MODELS_DIR / 'readiness_model.joblib'}")
    print(f"Saved metrics to {MODELS_DIR / 'model_metrics.json'}")


if __name__ == "__main__":
    main()
