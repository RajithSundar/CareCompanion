"""Simulated multi-session demo: the same coach, three sessions, advice that changes.

For each session the tool-using agent (agent.py) sees that day's biometrics and forecast,
and looks up the facts it has saved so far, what it advised in earlier sessions and recent
trends; it saves new facts and rewrites the plan as it goes. Next to every reply is the
answer a stateless assistant gives to the same question on the same day, with no
memory and no history, so the effect of memory is visible.

Profiles: the three fictional personas, or any of the 16 real PMData participants
(days from data/pmdata_daily.csv, derived from PMData under CC BY 4.0).
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

import agent
import coach
from demo_data import DEFAULT_APP_STATE
from features import APP_TO_CANONICAL, BASE_COLS, engineer_features, features_for_app_day

DAILY_CSV = Path(__file__).resolve().parent / "data" / "pmdata_daily.csv"

# Script for real participants: an intake, then the same question asked twice.
REAL_SCRIPT = [
    "I'm training for a 10 km race in about six weeks. My right calf gets tight after interval sessions. "
    "I can only train before work. What should I focus on this week?",
    "I have intervals planned for today. Should I do them?",
    "I have intervals planned for today. Should I do them?",
]


def _label_state(readiness: Optional[float], fatigue: Optional[float]) -> Optional[str]:
    if readiness is None or fatigue is None:
        return None
    if readiness >= 7 and fatigue >= 3:  # fatigue: 5 = very fresh (PMSys scale)
        return "Ready"
    if readiness <= 4 or fatigue <= 2:
        return "Deload Needed"
    return "Moderate"


@lru_cache(maxsize=1)
def _daily() -> Optional[pd.DataFrame]:
    if not DAILY_CSV.exists():
        return None
    df = pd.read_csv(DAILY_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df


def profiles() -> List[Dict[str, Any]]:
    out = [{"id": pid, "label": f"{p['name'].split()[0]}, made up: {p['role'].split(',')[0].lower()}", "kind": "persona"}
           for pid, p in DEFAULT_APP_STATE.items()]
    daily = _daily()
    if daily is not None:
        for pid, g in daily.groupby("participant_id"):
            out.append({"id": f"pmdata:{pid}", "label": f"Real person {pid[1:].lstrip('0')} from the dataset ({len(g)} days)", "kind": "pmdata"})
    return out


def _clean(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


def _real_sessions(pid: str) -> List[Dict[str, Any]]:
    """Pick three real days, about two weeks apart, preferring days whose state differs from the previous pick."""
    daily = _daily()
    if daily is None:
        raise KeyError("PMData daily table not available")
    g = daily[daily["participant_id"] == pid].sort_values("date").reset_index(drop=True)
    if g.empty:
        raise KeyError(pid)
    feats = engineer_features(g.copy()).reset_index(drop=True)
    complete = feats[["steps", "resting_heart_rate", "sleep_score", "readiness", "fatigue"]].notna().all(axis=1)
    states = [_label_state(r, f) for r, f in zip(feats["readiness"], feats["fatigue"])]

    def pick(start: int, avoid: Optional[str]) -> int:
        candidates = [i for i in range(start, len(feats)) if complete[i]]
        if not candidates:
            return min(start, len(feats) - 1)
        for i in candidates[:21]:
            if avoid is None or states[i] != avoid:
                return i
        return candidates[0]

    first = pick(min(7, len(feats) - 1), None)
    second = pick(first + 14, states[first])
    third = pick(second + 14, states[second])
    out = []
    for n, idx in enumerate((first, second, third)):
        row = feats.iloc[idx]
        history = feats.iloc[: idx + 1]
        out.append({
            "label": f"Session {n + 1}",
            "day": row["date"].strftime("%Y-%m-%d"),
            "features": _clean({c: float(row[c]) if pd.notna(row[c]) else None for c in feats.columns if c in _FEATURES}),
            "days": agent.normalize_days([_clean({**r, "date": r["date"].strftime("%Y-%m-%d")})
                                          for r in history.to_dict("records")]),
            "message": REAL_SCRIPT[n],
        })
    return out


def _persona_sessions(pid: str) -> List[Dict[str, Any]]:
    p = DEFAULT_APP_STATE[pid]
    out = []
    for n, sess in enumerate(p["sessions"][:3]):
        user_msg = next(m["text"] for m in sess["messages"] if m["role"] == "user")
        out.append({
            "label": f"Session {n + 1}",
            "day": sess["dayLabel"],
            "features": features_for_app_day(p["days"], sess["dayIndex"]),
            "days": agent.normalize_days(p["days"][: sess["dayIndex"] + 1]),
            "message": user_msg,
        })
    return out


_FEATURES: set = set()


def run_demo(profile: str, predict: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
             cfg: Optional[Dict[str, str]], client=None) -> Dict[str, Any]:
    """Run three sessions in order. `predict` maps a feature dict (None allowed) to a forecast."""
    from features import FEATURE_COLS
    _FEATURES.update(FEATURE_COLS)

    if profile.startswith("pmdata:"):
        pid = profile.split(":", 1)[1]
        sessions = _real_sessions(pid)
        name = f"PMData participant {pid.upper()}"
    elif profile in DEFAULT_APP_STATE:
        pid = profile
        sessions = _persona_sessions(pid)
        name = DEFAULT_APP_STATE[pid]["name"]
    else:
        raise KeyError(profile)

    memories: List[Dict[str, str]] = []  # the demo starts from a blank slate and builds memory as it goes
    prior: List[Dict[str, Any]] = []
    plan: List[Dict[str, Any]] = []
    results = []
    for sess in sessions:
        f = sess["features"]
        forecast = predict(f)
        memories_before = list(memories)
        ctx = agent.AgentContext(session_label=sess["label"], features=f, forecast=forecast, days=sess["days"],
                                 memories=memories, prior=list(prior), plan=plan)
        if cfg and client is not None:
            advice = agent.run(client, cfg, ctx, sess["message"])
            agent.ensure_memory(client, cfg, ctx, sess["message"])
            stateless = coach.complete(client, cfg, [
                {"role": "system", "content": coach.build_stateless_prompt(forecast, f)},
                {"role": "user", "content": sess["message"]},
            ])
        else:
            advice = agent.run_offline(ctx, pid, sess["message"])
            stateless = coach.offline_stateless_reply(f, forecast)
        memories = memories + ctx.new_memories
        if ctx.new_plan is not None:
            plan = ctx.new_plan
        snap = coach.snapshot(f)
        results.append({
            "label": sess["label"],
            "day": sess["day"],
            "snapshot": snap,
            "forecast": forecast,
            "message": sess["message"],
            "memories_used": memories_before,
            "new_memories": ctx.new_memories,
            "steps": ctx.steps,
            "plan": plan if ctx.new_plan is not None else None,
            "advice": advice,
            "stateless_advice": stateless,
        })
        prior.append({"label": sess["label"], "day": sess["day"], "snapshot": snap,
                      "forecast": forecast["predicted_recovery_state"] if forecast else None, "advice": advice})

    return {
        "profile": profile,
        "name": name,
        "mode": "llm_live" if cfg else "offline_fallback",
        "provider": f"{cfg['provider']}: {cfg['model']}" if cfg else "Offline rules (no LLM configured)",
        "sessions": results,
    }
