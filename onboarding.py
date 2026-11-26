"""Starting data for accounts created through sign-up.

There is no live Fitbit connection, so a new account starts one of two ways:

* fresh: no days at all. The person logs their own check-ins (or imports a CSV of earlier
  days), and the forecast and coach start once the first day is in.
* sample: the most recent SAMPLE_DAYS complete days of one real PMData participant, so
  the app can be tried straight away. Labelled as sample data everywhere it shows.

Participants without resting heart rate in the release (p12, p13) are not offered as
samples, since the app would have to invent it. Other gaps in a chosen day (REM sleep,
minutes in zones, wellness scores) are filled with that participant's own median.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import demo

SAMPLE_DAYS = 14
FRESH = "fresh"
SAMPLE_ROLE = "Sample data: PMData participant {pid}"
FRESH_ROLE = "Your own check-ins"

# App field -> PMData daily column. fatBurn is not in PMData and is left out.
_COLUMNS = {
    "steps": "steps", "rhr": "resting_heart_rate", "sleepScore": "sleep_score",
    "deepMin": "deep_sleep_min", "remMin": "rem_sleep_min", "cardio": "cardio_min", "peak": "peak_min",
    "readiness": "readiness", "fatigue": "fatigue", "soreness": "soreness", "stress": "stress",
}
_CORE = ["steps", "resting_heart_rate", "sleep_score", "readiness", "fatigue"]
_LIMITS = {"readiness": (0, 10), "fatigue": (1, 5), "soreness": (1, 5), "stress": (1, 5)}


def sample_participants() -> List[str]:
    """PMData participants whose days can seed a sample account."""
    daily = demo._daily()
    if daily is None:
        return []
    out = []
    for pid, g in daily.groupby("participant_id"):
        if g["resting_heart_rate"].notna().any() and g[_CORE].notna().all(axis=1).sum() >= SAMPLE_DAYS:
            out.append(str(pid))
    return sorted(out)


def valid_source(source: str) -> bool:
    return source == FRESH or (source.startswith("pmdata:") and source[7:] in sample_participants())


def role_for(source: str) -> str:
    return SAMPLE_ROLE.format(pid=source[7:].upper()) if source.startswith("pmdata:") else FRESH_ROLE


def _sample_days(pid: str) -> List[Dict[str, Any]]:
    daily = demo._daily()
    g = daily[daily["participant_id"] == pid].sort_values("date")
    medians = g.median(numeric_only=True)
    complete = g[g[_CORE].notna().all(axis=1)].tail(SAMPLE_DAYS)
    days = []
    for i, (_, row) in enumerate(complete.iterrows()):
        rec: Dict[str, Any] = {"day": f"Day {i + 1}", "date": row["date"].strftime("%Y-%m-%d")}
        for key, col in _COLUMNS.items():
            v = row[col]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                v = medians.get(col)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                v = 0.0
            v = float(v)
            if key in _LIMITS:
                lo, hi = _LIMITS[key]
                v = int(min(hi, max(lo, round(v))))
            elif key == "rhr":
                v = round(v, 1)
            else:
                v = int(round(v))
            rec[key] = v
        days.append(rec)
    return days


def baseline(days: List[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    """Average of the first seven days, or None before the first day."""
    first = days[:7]
    if not first:
        return None
    avg = lambda k: int(round(sum(float(d[k]) for d in first) / len(first)))  # noqa: E731
    return {"steps": avg("steps"), "rhr": avg("rhr"), "sleepScore": avg("sleepScore"), "readiness": avg("readiness")}


def starting_state(name: str, account_id: str, source: str) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """(profile state, memories) for a new account."""
    if source.startswith("pmdata:"):
        pid = source[7:]
        days = _sample_days(pid)
        role = role_for(source)
        intro = (f"Welcome, **{name}**. This account is using **sample data**: the last {len(days)} days of real "
                 f"PMData participant {pid.upper()}, so you can try the coach straight away. Tell me your goal, "
                 "any injury and when you can train, and I'll remember them for the next session.")
    else:
        days = []
        role = role_for(source)
        intro = (f"Welcome, **{name}**. Log today's numbers from your tracker on the **Check-in** page and I'll "
                 "forecast tomorrow's recovery. Then tell me your goal, any injury and when you can train, and "
                 "I'll remember them for the next session.")
    last = max(0, len(days) - 1)
    session = {
        "id": "S1", "label": "Session 1", "dayIndex": last,
        "dayLabel": days[last]["day"] if days else "No days yet",
        "title": "Session 1: Getting started", "subtitle": "Your goals, limits and first numbers",
        "adaptation": "First session. Nothing saved yet." if days else "No days logged yet. Log your first check-in.",
        "plan": [],
        "messages": [{"role": "coach", "text": intro, "citations": ["Session 1"]}],
    }
    state = {
        "id": account_id.upper(), "name": name, "role": role,
        "baseline": baseline(days), "days": days, "sessions": [session], "evolutionLog": [],
        "source": source,
    }
    return state, []
