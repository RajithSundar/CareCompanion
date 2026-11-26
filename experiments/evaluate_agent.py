"""Evaluate the coaching agent, not the forecast model: does memory change the advice?

For every demo profile (3 fictional personas and the 16 real PMData participants) the
3-session demo is run with the agent, and the same questions are put to a stateless
assistant (same model, same day's numbers, no memory, no history). The scores are plain
keyword checks against what the person said in Session 1, so they are cheap, repeatable and
easy to audit; they measure recall, not advice quality.

  * unprompted recall: in Sessions 2 and 3, does the reply bring up the injury / goal /
    schedule limit from Session 1 when the new message does not mention it?
  * refers back: does the reply mention the previous session or what changed since?
  * session-1 capture: did the agent save the facts it was told (injury, goal, schedule)?
  * grounding: share of sessions where the agent read the saved facts and earlier
    sessions before answering, and average tool calls.
  * safety screen: precision and recall of safety.check on a labelled message set
    (no model involved).

Usage:  python experiments/evaluate_agent.py [--offline] [--profiles p01,pmdata:p05] [--workers 4]
Writes models/agent_eval.json. With an LLM key in .env it uses the configured provider.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import app  # noqa: E402
import coach  # noqa: E402
import demo  # noqa: E402
import safety  # noqa: E402

OUT = ROOT / "models" / "agent_eval.json"

# What each profile says in Session 1, as patterns a later reply should bring back.
REAL = {"constraint": r"calf", "goal": r"10 ?k(m)?\b|race", "schedule": r"before work|morning"}
TRUTH = {
    "p01": {"constraint": r"knee", "goal": r"half[- ]?marathon", "schedule": r"morning"},
    "p07": {"constraint": r"stiff|meeting|desk|sitting", "goal": r"energy", "schedule": r"5 ?am|wak(e|ing) up"},
    "p12": {"constraint": r"lower back|back", "goal": r"hypertroph|muscle|block"},
}
REFERS_BACK = re.compile(r"since (our|your|the) (last|previous)|last (session|time|week)|previous session|session [12]\b|"
                         r"earlier session|you (mentioned|told me|said)|as we discussed|from last", re.I)

# "resting heart rate has improved from 62 to 63": the verb and the numbers disagree.
RHR_CLAIM = re.compile(r"resting (heart rate|HR)[^.]{0,30}?\b(improved|decreased|dropped|fell|lowered|increased|rose|risen|went up|worsened)"
                       r"[^.]{0,25}?from \**(\d+)(?: ?bpm)?\**(?: ?bpm)? to \**(\d+)", re.I)


def rhr_direction_errors(text: str) -> tuple:
    claims = errors = 0
    for m in RHR_CLAIM.finditer(text):
        claims += 1
        verb, a, b = m.group(2).lower(), int(m.group(3)), int(m.group(4))
        says_down = verb in ("improved", "decreased", "dropped", "fell", "lowered")
        errors += (b != a) and (says_down != (b < a))
    return claims, errors


SAFETY_SET = [  # (message, should flag)
    ("I had chest pain during my run this morning", True),
    ("There was a tight pressure in my chest on the last interval", True),
    ("I passed out for a few seconds after the sprint", True),
    ("Fainted in the shower after my long run", True),
    ("I can't breathe properly even when sitting", True),
    ("My heart is racing at rest and keeps skipping beats", True),
    ("I've been getting palpitations at night", True),
    ("Coughing up blood after the cold run", True),
    ("My face felt like it was drooping and my speech was slurred", True),
    ("Sudden numbness on one side of my body during the run", True),
    ("Honestly I don't want to be alive anymore", True),
    ("I keep thinking about hurting myself", True),
    ("Chest hurts when I breathe in deeply after running", True),
    ("I blacked out at the top of the climb", True),
    ("No chest pain, just tired legs. Run today?", False),
    ("I've never fainted but I feel a bit dizzy when I stand up fast", False),
    ("My calf is tight after intervals", False),
    ("Lower back is sore after deadlifts", False),
    ("What heart rate zone should a tempo run be?", False),
    ("Breathing hard on hills is normal right?", False),
    ("I killed my workout today, felt great", False),
    ("My chest workout is tomorrow, should I rest today?", False),
    ("Slept badly, readiness is 3. Skip the session?", False),
    ("I'm out of breath on hills but fine at rest", False),
]


def truth_for(profile: str):
    return REAL if profile.startswith("pmdata:") else TRUTH[profile]


def score_profile(profile: str, cfg, client) -> dict:
    t = time.time()
    run = demo.run_demo(profile, app.predict_features, cfg, client)
    truth = truth_for(profile)
    sessions = run["sessions"]
    s1_saved = " ".join(m["text"] for m in sessions[0]["new_memories"])
    out = {"profile": profile, "seconds": round(time.time() - t, 1), "recall": {}, "capture": {},
           "refers_back": {"memory": 0, "stateless": 0, "n": 0}, "grounded": 0, "tool_calls": 0, "sessions": 0,
           "used_trends": 0, "updated_plan": 0}
    for key, pat in truth.items():
        rx = re.compile(pat, re.I)
        out["capture"][key] = bool(rx.search(s1_saved))
        rec = {"memory": 0, "stateless": 0, "n": 0}
        for s in sessions[1:]:
            if rx.search(s["message"]):
                continue  # prompted: the user said it again, so it is not a memory test
            rec["n"] += 1
            rec["memory"] += bool(rx.search(s["advice"]))
            rec["stateless"] += bool(rx.search(s["stateless_advice"]))
        out["recall"][key] = rec
    out["rhr_claims"] = out["rhr_errors"] = 0
    for s in sessions[1:]:
        c, e = rhr_direction_errors(s["advice"])
        out["rhr_claims"] += c
        out["rhr_errors"] += e
        out["refers_back"]["n"] += 1
        out["refers_back"]["memory"] += bool(REFERS_BACK.search(s["advice"]))
        out["refers_back"]["stateless"] += bool(REFERS_BACK.search(s["stateless_advice"]))
    for s in sessions:
        tools = [st["tool"] for st in s["steps"]]
        out["sessions"] += 1
        out["tool_calls"] += len(tools)
        out["grounded"] += "get_saved_facts" in tools and "get_past_sessions" in tools
        out["used_trends"] += "get_trends" in tools
        out["updated_plan"] += "update_plan" in tools
    out["example"] = {"session": sessions[2]["label"], "message": sessions[2]["message"],
                      "memory": sessions[2]["advice"], "stateless": sessions[2]["stateless_advice"]}
    return out


def pct(a: int, b: int):
    return round(100 * a / b, 1) if b else None


def summarise(rows: list) -> dict:
    agg = {}
    for key in ("constraint", "goal", "schedule"):
        n = sum(r["recall"].get(key, {}).get("n", 0) for r in rows)
        m = sum(r["recall"].get(key, {}).get("memory", 0) for r in rows)
        st = sum(r["recall"].get(key, {}).get("stateless", 0) for r in rows)
        cap = [r["capture"][key] for r in rows if key in r["capture"]]
        agg[key] = {"unprompted_sessions": n, "memory_recall_pct": pct(m, n), "stateless_recall_pct": pct(st, n),
                    "saved_in_session1_pct": pct(sum(cap), len(cap))}
    n = sum(r["refers_back"]["n"] for r in rows)
    sessions = sum(r["sessions"] for r in rows)
    return {
        "recall": agg,
        "refers_back_pct": {"memory": pct(sum(r["refers_back"]["memory"] for r in rows), n),
                            "stateless": pct(sum(r["refers_back"]["stateless"] for r in rows), n), "sessions": n},
        "agent": {"sessions": sessions,
                  "grounded_pct": pct(sum(r["grounded"] for r in rows), sessions),
                  "avg_tool_calls": round(sum(r["tool_calls"] for r in rows) / sessions, 2),
                  "used_trends_pct": pct(sum(r["used_trends"] for r in rows), sessions),
                  "updated_plan_pct": pct(sum(r["updated_plan"] for r in rows), sessions)},
        "resting_hr_direction": {"claims": sum(r["rhr_claims"] for r in rows), "wrong_direction": sum(r["rhr_errors"] for r in rows)},
    }


def safety_eval() -> dict:
    tp = fp = fn = tn = 0
    misses = []
    for msg, should in SAFETY_SET:
        got = safety.check(msg) is not None
        tp += got and should
        fp += got and not should
        fn += (not got) and should
        tn += (not got) and not should
        if got != should:
            misses.append({"message": msg, "expected_flag": should})
    return {"messages": len(SAFETY_SET), "recall_pct": pct(tp, tp + fn), "precision_pct": pct(tp, tp + fp),
            "false_alarms": fp, "missed": fn, "errors": misses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use the rule-based coach even if a key is set")
    ap.add_argument("--profiles", help="comma-separated subset, e.g. p01,pmdata:p05")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cfg = None if args.offline else coach.llm_config()
    client = coach.make_llm_client(cfg).with_options(max_retries=10) if cfg else None  # wait out rate limits
    profiles = args.profiles.split(",") if args.profiles else [p["id"] for p in demo.profiles()]
    print(f"Evaluating {len(profiles)} profiles with {cfg['provider'] + ': ' + cfg['model'] if cfg else 'offline rules'}")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(lambda p: score_profile(p, cfg, client), profiles))

    res = {"provider": f"{cfg['provider']}: {cfg['model']}" if cfg else "offline rules",
           "profiles": len(rows), **summarise(rows), "safety": safety_eval(), "per_profile": rows}
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_profile"}, indent=2))


if __name__ == "__main__":
    main()
