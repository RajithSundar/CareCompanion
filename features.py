"""Feature definitions shared by training (train_model.py) and serving (app.py).

Both sides call `engineer_features` on the same column names, so the model always
sees inputs computed the same way. Rolling windows are calendar-based (by date),
not row-based, so a gap in the record does not stretch a "7-day" window.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

# Today's measurements. Readiness and fatigue define today's label, but the model
# forecasts tomorrow's label, so today's values are legitimate inputs (the
# persistence baseline already uses them).
BASE_COLS = [
    "steps",
    "resting_heart_rate",
    "sleep_score",
    "deep_sleep_min",
    "rem_sleep_min",
    "cardio_min",
    "peak_min",
    "readiness",
    "fatigue",
    "soreness",
    "stress",
]

DERIVED_COLS = [
    "rhr_delta",
    "sleep_3d_avg",
    "steps_3d_avg",
    "readiness_3d_avg",
    "fatigue_3d_avg",
]

FEATURE_COLS = BASE_COLS + DERIVED_COLS

# App day-record keys -> canonical column names.
APP_TO_CANONICAL = {
    "steps": "steps",
    "rhr": "resting_heart_rate",
    "sleepScore": "sleep_score",
    "deepMin": "deep_sleep_min",
    "remMin": "rem_sleep_min",
    "cardio": "cardio_min",
    "peak": "peak_min",
    "readiness": "readiness",
    "fatigue": "fatigue",
    "soreness": "soreness",
    "stress": "stress",
}

_SYNTHETIC_START = pd.Timestamp("2000-01-01")


def _rolling_mean(df: pd.DataFrame, col: str, days: int) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby("participant_id", sort=False):
        out.loc[g.index] = (
            g.rolling(f"{days}D", on="date", min_periods=1)[col].mean().to_numpy()
        )
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns. Each value uses only that day and earlier days of the same participant.

    Requires `participant_id`, `date` and every column in BASE_COLS (NaN allowed;
    the model pipeline imputes).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["participant_id", "date"]).reset_index(drop=True)
    df["rhr_7d_mean"] = _rolling_mean(df, "resting_heart_rate", 7)
    df["rhr_delta"] = df["resting_heart_rate"] - df["rhr_7d_mean"]
    df["sleep_3d_avg"] = _rolling_mean(df, "sleep_score", 3)
    df["steps_3d_avg"] = _rolling_mean(df, "steps", 3)
    df["readiness_3d_avg"] = _rolling_mean(df, "readiness", 3)
    df["fatigue_3d_avg"] = _rolling_mean(df, "fatigue", 3)
    return df


def app_days_to_frame(days: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert app day records to the canonical frame.

    If every record has an ISO `date`, those dates are used. Otherwise the records
    are treated as consecutive days in list order.
    """
    rows = []
    use_dates = bool(days) and all(d.get("date") for d in days)
    for i, d in enumerate(days):
        missing = [k for k in APP_TO_CANONICAL if d.get(k) is None]
        if missing:
            raise KeyError(missing[0])
        row = {canon: float(d[key]) for key, canon in APP_TO_CANONICAL.items()}
        row["participant_id"] = "app"
        row["date"] = pd.Timestamp(d["date"]) if use_dates else _SYNTHETIC_START + pd.Timedelta(days=i)
        row["_pos"] = i
        rows.append(row)
    return pd.DataFrame(rows)


def features_for_app_day(days: List[Dict[str, Any]], idx: int) -> Dict[str, float]:
    """Feature row for days[idx], using only days[0..idx]."""
    frame = app_days_to_frame(days[: idx + 1])
    feats = engineer_features(frame)
    row = feats.loc[feats["_pos"] == idx].iloc[0]
    return {c: round(float(row[c]), 4) for c in FEATURE_COLS}
