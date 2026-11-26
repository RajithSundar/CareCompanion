"""Feature definitions shared by training (train_model.py) and serving (app.py).

Both sides call `engineer_features` on the same column names, so the model always
sees inputs computed the same way. Rolling windows are calendar-based (by date),
not row-based, so a gap in the record does not stretch a "7-day" window.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
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

# Yesterday and the past week (added after experiments/explore_techniques.py). "Yesterday"
# is the previous calendar day: after a gap in the record these are missing and imputed.
LAG_COLS = [
    "readiness_lag1",
    "fatigue_lag1",
    "soreness_lag1",
    "stress_lag1",
    "sleep_lag1",
    "readiness_change",
    "fatigue_change",
    "today_state_num",
    "state_streak",
    "readiness_7d_avg",
    "fatigue_7d_avg",
    "readiness_7d_std",
]

FEATURE_COLS = BASE_COLS + DERIVED_COLS + LAG_COLS

# Features that need an earlier day, so they are missing on someone's first day.
OPTIONAL_COLS = {"readiness_lag1", "fatigue_lag1", "soreness_lag1", "stress_lag1", "sleep_lag1",
                 "readiness_change", "fatigue_change", "readiness_7d_std"}

STATE_NUM = {"Deload Needed": 0, "Moderate": 1, "Ready": 2}

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


def _rolling_std(df: pd.DataFrame, col: str, days: int) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby("participant_id", sort=False):
        out.loc[g.index] = g.rolling(f"{days}D", on="date", min_periods=2)[col].std().to_numpy()
    return out


def state_rule(readiness: pd.Series, fatigue: pd.Series) -> pd.Series:
    """The labelling rule: Ready if readiness >= 7 and fatigue <= 3; Deload Needed if
    readiness <= 4 or fatigue >= 4; otherwise Moderate. Missing inputs give a missing state."""
    out = pd.Series(np.where((readiness >= 7) & (fatigue <= 3), "Ready",
                             np.where((readiness <= 4) | (fatigue >= 4), "Deload Needed", "Moderate")),
                    index=readiness.index, dtype=object)
    return out.where(readiness.notna() & fatigue.notna())


def _add_lags(df: pd.DataFrame) -> None:
    g = df.groupby("participant_id", sort=False)
    consecutive = (df["date"] - g["date"].shift(1)) == pd.Timedelta(days=1)
    for col, name in (("readiness", "readiness_lag1"), ("fatigue", "fatigue_lag1"), ("soreness", "soreness_lag1"),
                      ("stress", "stress_lag1"), ("sleep_score", "sleep_lag1")):
        df[name] = g[col].shift(1).where(consecutive)
    df["readiness_change"] = df["readiness"] - df["readiness_lag1"]
    df["fatigue_change"] = df["fatigue"] - df["fatigue_lag1"]
    state = state_rule(df["readiness"], df["fatigue"])
    df["today_state_num"] = state.map(STATE_NUM).astype(float)
    # Consecutive calendar days in today's state, counting today.
    streak = np.ones(len(df))
    prev_state = state.groupby(df["participant_id"]).shift(1)
    same = consecutive & state.notna() & (state == prev_state)
    for i in range(len(df)):
        if same.iloc[i]:
            streak[i] = streak[i - 1] + 1
    df["state_streak"] = np.where(state.notna(), streak, np.nan)
    df["readiness_7d_avg"] = _rolling_mean(df, "readiness", 7)
    df["fatigue_7d_avg"] = _rolling_mean(df, "fatigue", 7)
    df["readiness_7d_std"] = _rolling_std(df, "readiness", 7)


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
    _add_lags(df)
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
    return {c: (None if pd.isna(row[c]) else round(float(row[c]), 4)) for c in FEATURE_COLS}
