"""Extra next-day features from PMData's training-load and injury logs, plus personal baselines.

Every feature for day t uses only information available by the end of day t
(or, for the weekday, the known calendar date of t+1).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

EXTRA_COLS = [
    "load_today",          # sum of RPE x minutes for sessions on day t
    "load_7d",             # mean daily load over the last 7 days (acute)
    "load_28d",            # mean daily load over the last 28 days (chronic)
    "acwr",                # acute:chronic workload ratio
    "readiness_dev",       # today's readiness minus this person's past mean
    "fatigue_dev",
    "sleep_dev",
    "rhr_dev",
    "injury_3d",           # injuries reported in the last 3 days
    "tomorrow_weekend",    # tomorrow is Saturday or Sunday
]


def _daily_load(pdir: Path) -> pd.Series:
    f = pdir / "pmsys" / "srpe.csv"
    if not f.exists():
        return pd.Series(dtype=float)
    s = pd.read_csv(f)
    s["date"] = pd.to_datetime(s["end_date_time"], utc=True, format="mixed").dt.tz_localize(None).dt.normalize()
    s["load"] = pd.to_numeric(s["perceived_exertion"], errors="coerce") * pd.to_numeric(s["duration_min"], errors="coerce")
    return s.groupby("date")["load"].sum()


def _daily_injury(pdir: Path) -> pd.Series:
    f = pdir / "pmsys" / "injury.csv"
    if not f.exists():
        return pd.Series(dtype=float)
    s = pd.read_csv(f)
    s["date"] = pd.to_datetime(s["effective_time_frame"], utc=True, format="mixed").dt.tz_localize(None).dt.normalize()

    def n_injuries(v):
        try:
            d = ast.literal_eval(str(v))
            return float(len(d)) if isinstance(d, dict) else 0.0
        except (ValueError, SyntaxError):
            return 0.0

    s["inj"] = s["injuries"].map(n_injuries)
    return s.groupby("date")["inj"].max()


def _past_mean(s: pd.Series) -> pd.Series:
    """Mean of earlier days only (expanding, shifted), at least 3 values."""
    return s.shift(1).expanding(min_periods=3).mean()


def add_extra_features(df: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    out = []
    for pid, g in df.groupby("participant_id", sort=False):
        g = g.sort_values("date").copy()
        pdir = Path(raw_dir) / pid
        days = pd.date_range(g["date"].min() - pd.Timedelta(days=28), g["date"].max(), freq="D")
        load = _daily_load(pdir).reindex(days, fill_value=0.0)
        inj = _daily_injury(pdir).reindex(days, fill_value=0.0)
        l7 = load.rolling(7, min_periods=1).mean()
        l28 = load.rolling(28, min_periods=1).mean()
        idx = pd.DatetimeIndex(g["date"])
        g["load_today"] = load.reindex(idx).to_numpy()
        g["load_7d"] = l7.reindex(idx).to_numpy()
        g["load_28d"] = l28.reindex(idx).to_numpy()
        g["acwr"] = np.where(g["load_28d"] > 0, g["load_7d"] / g["load_28d"].replace(0, np.nan), np.nan)
        g["injury_3d"] = inj.rolling(3, min_periods=1).sum().reindex(idx).to_numpy()
        for src, dst in (("readiness", "readiness_dev"), ("fatigue", "fatigue_dev"),
                         ("sleep_score", "sleep_dev"), ("resting_heart_rate", "rhr_dev")):
            g[dst] = g[src] - _past_mean(g[src])
        g["tomorrow_weekend"] = ((g["date"] + pd.Timedelta(days=1)).dt.dayofweek >= 5).astype(float)
        out.append(g)
    return pd.concat(out).sort_values(["participant_id", "date"]).reset_index(drop=True)


def split_by_time(df: pd.DataFrame, frac: float):
    """Per participant: the first `frac` of their rows (by date) vs the rest."""
    first, rest = [], []
    for _, g in df.groupby("participant_id", sort=False):
        g = g.sort_values("date")
        cut = int(np.floor(len(g) * frac))
        first.append(g.iloc[:cut])
        rest.append(g.iloc[cut:])
    return pd.concat(first), pd.concat(rest)
