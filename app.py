"""
CareCompanion — FastAPI Backend
Combines the trained PMData Supervised ML Readiness Model with a Groq-powered
Agentic Health Coach that persists biometric logs, sessions, and user memory.
"""

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = DATA_DIR / "agent_memory.json"
STATE_FILE = DATA_DIR / "app_state.json"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="CareCompanion: Biometric Health Coach")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


DEFAULT_MEMORY_STORE: Dict[str, List[Dict[str, str]]] = {
    "p01": [
        {"id": "m1", "category": "Goal", "text": "Training for half-marathon in 8 weeks", "session": "S1"},
        {"id": "m2", "category": "Constraint", "text": "Left knee tightness after high-speed work; cap Peak HR", "session": "S1"},
        {"id": "m3", "category": "Baseline", "text": "Morning workout preference; baseline RHR 58 bpm", "session": "S1"},
    ],
    "p02": [
        {"id": "pm1", "category": "Goal", "text": "Boost afternoon energy & reduce desk stiffness", "session": "S1"},
        {"id": "pm2", "category": "Constraint", "text": "No 5 AM workouts; prefers lunchtime walks & evening mobility", "session": "S1"},
        {"id": "pm3", "category": "Baseline", "text": "Baseline RHR 64 bpm; desk-heavy workday", "session": "S1"},
    ],
    "p03": [
        {"id": "jm1", "category": "Goal", "text": "Increase lower-body strength while protecting sleep quality", "session": "S1"},
        {"id": "jm2", "category": "Constraint", "text": "Lower-back tightness after heavy deadlifts", "session": "S1"},
        {"id": "jm3", "category": "Baseline", "text": "Baseline RHR 55 bpm; 4x/week lifting schedule", "session": "S1"},
    ],
    "p07": [
        {"id": "pm1", "category": "Goal", "text": "Boost afternoon energy & reduce desk stiffness", "session": "S1"},
        {"id": "pm2", "category": "Constraint", "text": "No 5 AM workouts; prefers lunchtime walks & evening yoga", "session": "S1"},
        {"id": "pm3", "category": "Baseline", "text": "Baseline RHR 65 bpm; desk-bound tech lead schedule", "session": "S1"},
    ],
    "p12": [
        {"id": "jm1", "category": "Goal", "text": "Increase lower-body strength & hypertrophy while protecting sleep", "session": "S1"},
        {"id": "jm2", "category": "Constraint", "text": "Lower-back tightness after heavy deadlifts", "session": "S1"},
        {"id": "jm3", "category": "Baseline", "text": "Baseline RHR 55 bpm; 4-day upper/lower split", "session": "S1"},
    ],
}


DEFAULT_APP_STATE: Dict[str, Dict[str, Any]] = {
    "p01": {
        "id": "P01",
        "name": "Alex Rivera",
        "role": "Half-Marathon Prep, Morning Runner",
        "baseline": {"steps": 9400, "rhr": 58, "sleepScore": 79, "readiness": 7},
        "days": [
            {"day": "Day 1", "steps": 9650, "rhr": 58, "sleepScore": 80, "deepMin": 88, "remMin": 95, "readiness": 8, "fatigue": 2, "soreness": 2, "stress": 2, "fatBurn": 38, "cardio": 26, "peak": 8},
            {"day": "Day 2", "steps": 10800, "rhr": 59, "sleepScore": 77, "deepMin": 82, "remMin": 90, "readiness": 7, "fatigue": 3, "soreness": 2, "stress": 2, "fatBurn": 42, "cardio": 31, "peak": 11},
            {"day": "Day 3", "steps": 11400, "rhr": 60, "sleepScore": 72, "deepMin": 71, "remMin": 84, "readiness": 6, "fatigue": 3, "soreness": 3, "stress": 3, "fatBurn": 45, "cardio": 34, "peak": 14},
            {"day": "Day 4", "steps": 8900, "rhr": 62, "sleepScore": 64, "deepMin": 58, "remMin": 74, "readiness": 5, "fatigue": 4, "soreness": 4, "stress": 4, "fatBurn": 32, "cardio": 18, "peak": 4},
            {"day": "Day 5", "steps": 5400, "rhr": 63, "sleepScore": 59, "deepMin": 49, "remMin": 68, "readiness": 4, "fatigue": 4, "soreness": 4, "stress": 4, "fatBurn": 21, "cardio": 6, "peak": 0},
            {"day": "Day 6", "steps": 6100, "rhr": 62, "sleepScore": 68, "deepMin": 66, "remMin": 81, "readiness": 5, "fatigue": 3, "soreness": 3, "stress": 3, "fatBurn": 28, "cardio": 8, "peak": 0},
            {"day": "Day 7", "steps": 7400, "rhr": 60, "sleepScore": 75, "deepMin": 79, "remMin": 89, "readiness": 6, "fatigue": 3, "soreness": 3, "stress": 2, "fatBurn": 34, "cardio": 14, "peak": 2},
            {"day": "Day 8", "steps": 8500, "rhr": 59, "sleepScore": 79, "deepMin": 85, "remMin": 94, "readiness": 7, "fatigue": 2, "soreness": 2, "stress": 2, "fatBurn": 39, "cardio": 22, "peak": 5},
            {"day": "Day 9", "steps": 9200, "rhr": 58, "sleepScore": 82, "deepMin": 92, "remMin": 98, "readiness": 8, "fatigue": 2, "soreness": 2, "stress": 2, "fatBurn": 41, "cardio": 25, "peak": 7},
            {"day": "Day 10", "steps": 10350, "rhr": 57, "sleepScore": 84, "deepMin": 96, "remMin": 104, "readiness": 8, "fatigue": 2, "soreness": 2, "stress": 2, "fatBurn": 44, "cardio": 29, "peak": 9},
        ],
        "sessions": [
            {
                "id": "S1", "label": "Session 1", "dayIndex": 0, "dayLabel": "Day 1",
                "title": "Session 1: Baseline Intake", "subtitle": "Establishing biometric baselines and half-marathon goal",
                "adaptation": "Captured half-marathon goal, morning preference, and left knee Peak HR cap.",
                "plan": [
                    {"id": "p1_1", "text": "35 min Zone 2/3 morning run", "done": True},
                    {"id": "p1_2", "text": "Hit 9,500+ daily steps", "done": True},
                    {"id": "p1_3", "text": "7.5h sleep target (lights out 10:30 PM)", "done": True},
                    {"id": "p1_4", "text": "Log PMSys soreness after runs", "done": True},
                ],
                "messages": [
                    {"role": "user", "text": "I'm training for a half-marathon in 8 weeks. My left knee sometimes feels tight after speed work, and I prefer morning workouts. What should I focus on this week?"},
                    {"role": "coach", "text": "Today's baseline shows **9,650 steps**, **58 bpm RHR**, **80 sleep score**, and **8/10 readiness**. Aim for **35 to 40 min morning Zone 2/3 runs** and **9,500 to 10,000 steps**. Because of your left knee history, cap Peak HR zone work under **10 min/day** and log knee soreness in PMSys.", "citations": ["Baseline Recorded", "Half-Marathon Goal", "Left Knee Constraint"]},
                ],
            },
            {
                "id": "S2", "label": "Session 2", "dayIndex": 4, "dayLabel": "Day 5",
                "title": "Session 2: Fatigue & RHR Shift", "subtitle": "Adapting plan after sleep drop and knee soreness spike",
                "adaptation": "RHR climbed +5 bpm (58 to 63) and sleep fell to 59: pivoted tempo run to 20 min mobility.",
                "plan": [
                    {"id": "p2_1", "text": "Replace tempo run with 20 min mobility", "done": True},
                    {"id": "p2_2", "text": "Cap HR in Fat Burn zone (0 min Peak)", "done": True},
                    {"id": "p2_3", "text": "Prioritize 8h recovery sleep tonight", "done": False},
                    {"id": "p2_4", "text": "Ice and quad/glute activation for left knee", "done": True},
                ],
                "messages": [
                    {"role": "user", "text": "I feel sluggish today, but I have a tempo run on my calendar. Should I push through it?"},
                    {"role": "coach", "text": "**Hold off on the tempo run today.** Our supervised model flagged **Deload Needed** because your Fitbit RHR jumped **+5 bpm (58 to 63 bpm)** and sleep score fell from **80 to 59**. Combined with your **left knee sensitivity** from Session 1 and **4/5 soreness**, swap the run for **20 min mobility** in the Fat Burn zone and aim for **8h sleep**.", "citations": ["ML: Deload Needed", "Recalled S1: Left Knee", "Detected: RHR +5 bpm"]},
                ],
            },
            {
                "id": "S3", "label": "Session 3", "dayIndex": 9, "dayLabel": "Day 10",
                "title": "Session 3: Recovery & Ramp-Up", "subtitle": "Validating deload impact and resuming progression",
                "adaptation": "RHR settled to 57 bpm and sleep score rebounded to 84: cleared progressive Zone 3.",
                "plan": [
                    {"id": "p3_1", "text": "40 min progressive morning run (12 min Zone 3)", "done": False},
                    {"id": "p3_2", "text": "Maintain 10,000+ daily steps", "done": True},
                    {"id": "p3_3", "text": "Keep 7.5 to 8h sleep consistency", "done": True},
                    {"id": "p3_4", "text": "Pre-run 5 min knee warm-up drill", "done": False},
                ],
                "messages": [
                    {"role": "user", "text": "My legs feel much fresher after taking those lighter days. Am I ready to ramp back up toward my half-marathon?"},
                    {"role": "coach", "text": "**Your deload worked.** RHR settled to **57 bpm**, sleep score rebounded to **84**, and soreness dropped to **2/5**. Resume with a **40 min morning run** (12 min Zone 3) and a **5 min pre-run knee warm-up**.", "citations": ["Recalled S2: Deload Worked", "Recalled S1: Morning Run"]},
                ],
            },
            {
                "id": "S4", "label": "Session 4", "dayIndex": 9, "dayLabel": "Day 10+",
                "title": "Session 4: Live Check-in", "subtitle": "Ongoing coaching grounded in current biometrics and memory",
                "adaptation": "Active ML classifier and persistent memory graph loaded for Alex Rivera.",
                "plan": [
                    {"id": "p4_1", "text": "40 min progressive morning run (12 min Zone 3)", "done": False},
                    {"id": "p4_2", "text": "Maintain 10,000+ daily steps", "done": True},
                    {"id": "p4_3", "text": "Keep 7.5 to 8h sleep consistency", "done": True},
                    {"id": "p4_4", "text": "Pre-run 5 min knee warm-up drill", "done": False},
                ],
                "messages": [
                    {"role": "coach", "text": "Your PMData telemetry and persistent memory nodes from Sessions 1 to 3 are active. Ask a training question or log an updated biometric check-in to adapt today's protocol.", "citations": ["ML + Memory Active", "Window: Day 1 to Day 10"]},
                ],
            },
        ],
        "evolutionLog": [
            {"session": "S1", "text": "Captured half-marathon goal, morning preference, and left knee constraint."},
            {"session": "S2", "text": "ML flagged Deload Needed; added automatic tempo deload rule."},
            {"session": "S3", "text": "RHR recovered to 57 bpm and sleep reached 84; unlocked progressive Zone 3."},
        ],
    },
    "p07": {
        "id": "P07",
        "name": "Maya Lin",
        "role": "Tech Lead, Desk-Bound + Evening Yoga",
        "baseline": {"steps": 6400, "rhr": 65, "sleepScore": 74, "readiness": 6},
        "days": [
            {"day": "Day 1", "steps": 6200, "rhr": 65, "sleepScore": 74, "deepMin": 72, "remMin": 81, "readiness": 6, "fatigue": 3, "soreness": 2, "stress": 3, "fatBurn": 24, "cardio": 10, "peak": 1},
            {"day": "Day 2", "steps": 5800, "rhr": 66, "sleepScore": 71, "deepMin": 68, "remMin": 76, "readiness": 6, "fatigue": 3, "soreness": 2, "stress": 4, "fatBurn": 20, "cardio": 8, "peak": 0},
            {"day": "Day 3", "steps": 4900, "rhr": 67, "sleepScore": 66, "deepMin": 59, "remMin": 70, "readiness": 5, "fatigue": 4, "soreness": 3, "stress": 4, "fatBurn": 16, "cardio": 5, "peak": 0},
            {"day": "Day 4", "steps": 4100, "rhr": 69, "sleepScore": 60, "deepMin": 48, "remMin": 62, "readiness": 4, "fatigue": 4, "soreness": 3, "stress": 5, "fatBurn": 12, "cardio": 3, "peak": 0},
            {"day": "Day 5", "steps": 3900, "rhr": 70, "sleepScore": 57, "deepMin": 44, "remMin": 58, "readiness": 3, "fatigue": 5, "soreness": 3, "stress": 5, "fatBurn": 10, "cardio": 2, "peak": 0},
            {"day": "Day 6", "steps": 5500, "rhr": 68, "sleepScore": 67, "deepMin": 61, "remMin": 72, "readiness": 5, "fatigue": 3, "soreness": 2, "stress": 4, "fatBurn": 22, "cardio": 7, "peak": 0},
            {"day": "Day 7", "steps": 6900, "rhr": 67, "sleepScore": 73, "deepMin": 70, "remMin": 80, "readiness": 6, "fatigue": 3, "soreness": 2, "stress": 3, "fatBurn": 28, "cardio": 12, "peak": 2},
            {"day": "Day 8", "steps": 7600, "rhr": 65, "sleepScore": 78, "deepMin": 80, "remMin": 88, "readiness": 7, "fatigue": 2, "soreness": 2, "stress": 2, "fatBurn": 33, "cardio": 16, "peak": 3},
            {"day": "Day 9", "steps": 8100, "rhr": 64, "sleepScore": 81, "deepMin": 85, "remMin": 92, "readiness": 8, "fatigue": 2, "soreness": 1, "stress": 2, "fatBurn": 36, "cardio": 18, "peak": 4},
            {"day": "Day 10", "steps": 8450, "rhr": 64, "sleepScore": 83, "deepMin": 89, "remMin": 95, "readiness": 8, "fatigue": 2, "soreness": 1, "stress": 2, "fatBurn": 38, "cardio": 20, "peak": 4},
        ],
        "sessions": [
            {
                "id": "S1", "label": "Session 1", "dayIndex": 0, "dayLabel": "Day 1",
                "title": "Session 1: Baseline Intake", "subtitle": "Desk posture, stress management, and daily movement",
                "adaptation": "Set 7,500 step micro-walk goal and anchored 20 min evening yoga.",
                "plan": [
                    {"id": "m1_1", "text": "Two 10 min post-lunch walking breaks", "done": True},
                    {"id": "m1_2", "text": "Reach 7,500 daily steps", "done": False},
                    {"id": "m1_3", "text": "20 min evening wind-down yoga", "done": True},
                    {"id": "m1_4", "text": "Screen curfew 45 min before bed", "done": False},
                ],
                "messages": [
                    {"role": "user", "text": "I sit in back-to-back meetings all day and feel stiff by 6 PM. I want more energy without waking up at 5 AM."},
                    {"role": "coach", "text": "Today's metrics show **6,200 steps**, **65 bpm RHR**, and **74 sleep score**. Anchor two **10 min walking breaks** between meetings to hit **7,500 steps**, paired with **20 min evening yoga**.", "citations": ["Schedule: Evening Yoga", "Goal: 7,500 Steps"]},
                ],
            },
            {
                "id": "S2", "label": "Session 2", "dayIndex": 4, "dayLabel": "Day 5",
                "title": "Session 2: Crunch Week Stress Spike", "subtitle": "Detecting PMSys stress 5/5 and sleep fragmentation",
                "adaptation": "ML flagged Deload Needed: dropped step pressure and pivoted to breathwork.",
                "plan": [
                    {"id": "m2_1", "text": "Drop step pressure (5,000 gentle steps)", "done": True},
                    {"id": "m2_2", "text": "10 min parasympathetic box breathing", "done": True},
                    {"id": "m2_3", "text": "Skip intense evening workouts tonight", "done": True},
                    {"id": "m2_4", "text": "No caffeine after 1:00 PM", "done": False},
                ],
                "messages": [
                    {"role": "user", "text": "Product launch week is brutal. Should I force a hard cardio session tonight to catch up?"},
                    {"role": "coach", "text": "**Skip hard cardio tonight.** The ML model flagged **Deload Needed**: your **RHR climbed to 70 bpm (+5)**, sleep score dropped to **57**, and stress is **5/5**. Pause the 7,500 step target and do **10 min box breathing plus restorative yoga**.", "citations": ["ML: Deload Needed", "Adjusted S1: Paused Step Goal"]},
                ],
            },
            {
                "id": "S3", "label": "Session 3", "dayIndex": 9, "dayLabel": "Day 10",
                "title": "Session 3: Post-Launch Rebound", "subtitle": "Sustaining 8,450 steps and 83 sleep score",
                "adaptation": "Sleep rebounded to 83 and RHR returned to 64 bpm: graduated step target to 8,500.",
                "plan": [
                    {"id": "m3_1", "text": "Maintain 8,500 daily steps", "done": True},
                    {"id": "m3_2", "text": "2x/week 25 min brisk Zone 2 walks", "done": True},
                    {"id": "m3_3", "text": "Keep evening wind-down yoga", "done": True},
                    {"id": "m3_4", "text": "Log weekly stress trend", "done": False},
                ],
                "messages": [
                    {"role": "user", "text": "Launch is over and I slept well the last two nights. What is my next step?"},
                    {"role": "coach", "text": "Sleep score jumped from **57 to 83**, RHR settled to **64 bpm**, and you reached **8,450 steps**. Lock in **8,500 daily steps** and add two **25 min brisk Zone 2 walks**.", "citations": ["Rebound Verified", "Updated Goal: 8,500 Steps"]},
                ],
            },
            {
                "id": "S4", "label": "Session 4", "dayIndex": 9, "dayLabel": "Day 10+",
                "title": "Session 4: Live Check-in", "subtitle": "Ongoing coaching grounded in current biometrics and memory",
                "adaptation": "Active ML classifier and memory graph loaded for Maya Lin.",
                "plan": [
                    {"id": "m4_1", "text": "Maintain 8,500 daily steps", "done": True},
                    {"id": "m4_2", "text": "2x/week 25 min brisk Zone 2 walks", "done": True},
                    {"id": "m4_3", "text": "Keep evening wind-down yoga", "done": True},
                    {"id": "m4_4", "text": "Log weekly stress trend", "done": False},
                ],
                "messages": [
                    {"role": "coach", "text": "Welcome back, Maya. Your PMData history and stress-adaptation rules are active. Ask a question or record a check-in to update today's plan.", "citations": ["ML + Memory Active"]},
                ],
            },
        ],
        "evolutionLog": [
            {"session": "S1", "text": "Logged evening workout preference and desk stiffness goal."},
            {"session": "S2", "text": "ML flagged Deload Needed; replaced cardio with breathwork."},
            {"session": "S3", "text": "Rebound confirmed; graduated step target to 8,500."},
        ],
    },
    "p12": {
        "id": "P12",
        "name": "Jordan Patel",
        "role": "Strength Athlete, Hypertrophy Focus",
        "baseline": {"steps": 7800, "rhr": 55, "sleepScore": 81, "readiness": 8},
        "days": [
            {"day": "Day 1", "steps": 8100, "rhr": 55, "sleepScore": 82, "deepMin": 94, "remMin": 99, "readiness": 8, "fatigue": 2, "soreness": 2, "stress": 2, "fatBurn": 30, "cardio": 18, "peak": 6},
            {"day": "Day 2", "steps": 8400, "rhr": 56, "sleepScore": 80, "deepMin": 90, "remMin": 93, "readiness": 8, "fatigue": 2, "soreness": 3, "stress": 2, "fatBurn": 34, "cardio": 20, "peak": 7},
            {"day": "Day 3", "steps": 7900, "rhr": 57, "sleepScore": 76, "deepMin": 82, "remMin": 88, "readiness": 7, "fatigue": 3, "soreness": 3, "stress": 2, "fatBurn": 29, "cardio": 16, "peak": 5},
            {"day": "Day 4", "steps": 7200, "rhr": 59, "sleepScore": 69, "deepMin": 68, "remMin": 79, "readiness": 5, "fatigue": 4, "soreness": 4, "stress": 3, "fatBurn": 24, "cardio": 11, "peak": 2},
            {"day": "Day 5", "steps": 6500, "rhr": 61, "sleepScore": 63, "deepMin": 56, "remMin": 71, "readiness": 4, "fatigue": 4, "soreness": 5, "stress": 3, "fatBurn": 19, "cardio": 6, "peak": 0},
            {"day": "Day 6", "steps": 7100, "rhr": 59, "sleepScore": 72, "deepMin": 74, "remMin": 84, "readiness": 6, "fatigue": 3, "soreness": 3, "stress": 2, "fatBurn": 25, "cardio": 10, "peak": 1},
            {"day": "Day 7", "steps": 7800, "rhr": 57, "sleepScore": 78, "deepMin": 86, "remMin": 91, "readiness": 7, "fatigue": 2, "soreness": 3, "stress": 2, "fatBurn": 31, "cardio": 15, "peak": 4},
            {"day": "Day 8", "steps": 8300, "rhr": 56, "sleepScore": 82, "deepMin": 91, "remMin": 96, "readiness": 8, "fatigue": 2, "soreness": 2, "stress": 2, "fatBurn": 33, "cardio": 19, "peak": 5},
            {"day": "Day 9", "steps": 8600, "rhr": 55, "sleepScore": 84, "deepMin": 96, "remMin": 101, "readiness": 8, "fatigue": 2, "soreness": 2, "stress": 1, "fatBurn": 35, "cardio": 21, "peak": 6},
            {"day": "Day 10", "steps": 8900, "rhr": 54, "sleepScore": 86, "deepMin": 99, "remMin": 105, "readiness": 9, "fatigue": 1, "soreness": 1, "stress": 1, "fatBurn": 37, "cardio": 23, "peak": 7},
        ],
        "sessions": [
            {
                "id": "S1", "label": "Session 1", "dayIndex": 0, "dayLabel": "Day 1",
                "title": "Session 1: Baseline Intake", "subtitle": "Heavy lower-body block and recovery monitoring",
                "adaptation": "Scheduled 4-day upper/lower split with lower-back PMSys tracking.",
                "plan": [
                    {"id": "j1_1", "text": "4-day Upper/Lower hypertrophy split", "done": True},
                    {"id": "j1_2", "text": "160g+ daily protein intake", "done": True},
                    {"id": "j1_3", "text": "Track lower-back soreness in PMSys", "done": True},
                    {"id": "j1_4", "text": "8,000 steps on rest days", "done": True},
                ],
                "messages": [
                    {"role": "user", "text": "Starting a 4-day hypertrophy block. My lower back sometimes fatigues after heavy deadlifts. How should I manage volume?"},
                    {"role": "coach", "text": "Baseline biometrics show **55 bpm RHR**, **82 sleep score**, and **8/10 readiness**. Run your 4-day Upper/Lower split at RPE 7 to 8, and log lower-back soreness in PMSys after posterior-chain days.", "citations": ["Baseline Recorded", "Lower-Back Constraint"]},
                ],
            },
            {
                "id": "S2", "label": "Session 2", "dayIndex": 4, "dayLabel": "Day 5",
                "title": "Session 2: CNS & Soreness Spike", "subtitle": "Soreness 5/5 and RHR +6 bpm after heavy pull day",
                "adaptation": "ML flagged Deload Needed: swapped barbell squats for machine/isolation work at RPE 6.",
                "plan": [
                    {"id": "j2_1", "text": "Swap heavy squats for leg press and curls", "done": True},
                    {"id": "j2_2", "text": "Keep RPE <= 6 today", "done": True},
                    {"id": "j2_3", "text": "15 min hip flexor and thoracic mobility", "done": True},
                    {"id": "j2_4", "text": "Extra 45 min sleep tonight", "done": False},
                ],
                "messages": [
                    {"role": "user", "text": "Lower back and hamstrings are wrecked (5/5 soreness), but today is heavy squat day. Should I load up?"},
                    {"role": "coach", "text": "**Do not load heavy axial squats today.** The ML classifier flagged **Deload Needed**: your **RHR spiked +6 bpm (55 to 61)** and soreness is **5/5**, matching your lower-back sensitivity from Session 1. Pivot to **supported machine work** at **RPE 6**.", "citations": ["ML: Deload Needed", "Recalled S1: Lower-Back"]},
                ],
            },
            {
                "id": "S3", "label": "Session 3", "dayIndex": 9, "dayLabel": "Day 10",
                "title": "Session 3: Supercompensation", "subtitle": "Readiness 9/10 and RHR 54 bpm",
                "adaptation": "RHR reached 54 bpm and sleep hit 86: cleared progressive overload with trap-bar/RDL.",
                "plan": [
                    {"id": "j3_1", "text": "Resume compound lifts at RPE 8", "done": False},
                    {"id": "j3_2", "text": "Use Trap Bar or RDL to spare lower back", "done": True},
                    {"id": "j3_3", "text": "Maintain 8,500+ steps", "done": True},
                    {"id": "j3_4", "text": "Continue PMSys post-session logs", "done": True},
                ],
                "messages": [
                    {"role": "user", "text": "Soreness is completely gone today. Ready to push for a PR?"},
                    {"role": "coach", "text": "**Green light for progressive overload.** You hit a 10-day best **54 bpm RHR** and **86 sleep score**. Hit your top sets at **RPE 8**, using **trap-bar or controlled RDLs** to protect your lower back.", "citations": ["Recalled S2: Spine-Sparing Rule"]},
                ],
            },
            {
                "id": "S4", "label": "Session 4", "dayIndex": 9, "dayLabel": "Day 10+",
                "title": "Session 4: Live Check-in", "subtitle": "Ongoing coaching grounded in current biometrics and memory",
                "adaptation": "Active ML classifier and memory graph loaded for Jordan Patel.",
                "plan": [
                    {"id": "j4_1", "text": "Resume compound lifts at RPE 8", "done": False},
                    {"id": "j4_2", "text": "Use Trap Bar or RDL to spare lower back", "done": True},
                    {"id": "j4_3", "text": "Maintain 8,500+ steps", "done": True},
                    {"id": "j4_4", "text": "Continue PMSys post-session logs", "done": True},
                ],
                "messages": [
                    {"role": "coach", "text": "All 10 days of your PMData biometrics and strength-block memories are active. Ask a question or update today's check-in to adapt your next session.", "citations": ["ML + Memory Active"]},
                ],
            },
        ],
        "evolutionLog": [
            {"session": "S1", "text": "Recorded hypertrophy goal and lower-back fatigue risk."},
            {"session": "S2", "text": "ML flagged Deload Needed; swapped heavy squats for machines."},
            {"session": "S3", "text": "Supercompensation confirmed; cleared progressive overload."},
        ],
    },
}


def normalize_pid(participant_id: str) -> str:
    return participant_id.strip().lower()


def load_ml_bundle():
    model_path = MODELS_DIR / "readiness_model.joblib"
    metrics_path = MODELS_DIR / "model_metrics.json"
    if not model_path.exists():
        return None, None
    bundle = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return bundle, metrics


def load_memory_store() -> Dict[str, List[Dict[str, str]]]:
    if MEMORY_FILE.exists():
        store = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        changed = False
        for pid, items in DEFAULT_MEMORY_STORE.items():
            if pid not in store:
                store[pid] = copy.deepcopy(items)
                changed = True
        if changed:
            save_memory_store(store)
        return store
    store = copy.deepcopy(DEFAULT_MEMORY_STORE)
    save_memory_store(store)
    return store


def save_memory_store(store: Dict[str, Any]):
    MEMORY_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")


def load_app_state() -> Dict[str, Dict[str, Any]]:
    if STATE_FILE.exists():
        store = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        changed = False
        for pid, default_data in DEFAULT_APP_STATE.items():
            if pid not in store:
                store[pid] = copy.deepcopy(default_data)
                changed = True
        if changed:
            save_app_state(store)
        return store
    store = copy.deepcopy(DEFAULT_APP_STATE)
    save_app_state(store)
    return store


def save_app_state(store: Dict[str, Any]):
    STATE_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")


def build_biometrics_from_days(days: List[Dict[str, Any]], baseline_rhr: float, day_idx: int) -> "BiometricInput":
    idx = max(0, min(day_idx, len(days) - 1))
    d = days[idx]
    win = days[max(0, idx - 2) : idx + 1]
    sleep_3d = sum(float(x.get("sleepScore", 75)) for x in win) / len(win)
    steps_3d = sum(float(x.get("steps", 7500)) for x in win) / len(win)
    cardio = float(d.get("cardio", 18))
    return BiometricInput(
        steps=float(d.get("steps", 7500)),
        resting_heart_rate=float(d.get("rhr", 60)),
        sleep_score=float(d.get("sleepScore", 75)),
        deep_sleep_min=float(d.get("deepMin", 80)),
        rem_sleep_min=float(d.get("remMin", 88)),
        sedentary_min=640.0,
        very_active_min=round(cardio * 0.9, 1),
        cardio_min=cardio,
        peak_min=float(d.get("peak", 4)),
        rhr_delta=round(float(d.get("rhr", 60)) - baseline_rhr, 2),
        sleep_3d_avg=round(sleep_3d, 2),
        steps_3d_avg=round(steps_3d, 1),
    )


def run_ml_prediction(biometrics: "BiometricInput") -> Optional[Dict[str, Any]]:
    bundle, _ = load_ml_bundle()
    if not bundle:
        return None
    cols = bundle["feature_cols"]
    row = pd.DataFrame([biometrics.model_dump()])[cols]
    pred_class = str(bundle["model"].predict(row)[0])
    probs = bundle["model"].predict_proba(row)[0]
    return {
        "predicted_recovery_state": pred_class,
        "probabilities": {str(c): round(float(p), 4) for c, p in zip(bundle["classes"], probs)},
        "model_name": bundle["model_name"],
    }


def extract_new_memories_with_groq(
    client: Groq,
    model_name: str,
    user_message: str,
    session_label: str,
    existing_memories: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Use Groq JSON mode to extract durable goals, constraints, baselines, or adaptations."""
    existing_texts = {m["text"].strip().lower() for m in existing_memories}
    existing_summary = "\n".join(f"- [{m['category']}] {m['text']}" for m in existing_memories) or "None"

    extraction_prompt = (
        "Extract any NEW, durable health coaching facts from the user's message that should be remembered "
        "in future sessions (e.g., new injury/pain, schedule preference, goal update, or biometric trigger).\n"
        f"Already known memories (DO NOT duplicate these):\n{existing_summary}\n\n"
        'Return JSON ONLY in this exact format: {"memories": [{"category": "Goal|Constraint|Baseline|Adaptation", "text": "..."}]}\n'
        'If there are no new durable facts, return {"memories": []}.'
    )

    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": extraction_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = res.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        candidates = parsed.get("memories", [])
        if not isinstance(candidates, list):
            return []

        valid_cats = {"Goal", "Constraint", "Baseline", "Adaptation"}
        added: List[Dict[str, str]] = []
        ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        for idx, item in enumerate(candidates[:3]):
            if not isinstance(item, dict):
                continue
            cat = item.get("category", "Baseline")
            if cat not in valid_cats:
                cat = "Baseline"
            text = str(item.get("text", "")).strip()
            if not text or text.lower() in existing_texts:
                continue
            existing_texts.add(text.lower())
            added.append({
                "id": f"m_{ts}_{idx}",
                "category": cat,
                "text": text,
                "session": session_label,
            })
        return added
    except Exception:
        return []


def build_offline_coach_reply(
    req: "CoachRequest",
    ml_result: Optional[Dict[str, Any]],
    memories: List[Dict[str, str]],
) -> str:
    """Memory-aware fallback when no GROQ_API_KEY is configured."""
    pred_label = ml_result["predicted_recovery_state"] if ml_result else "Moderate"
    conf = ""
    if ml_result and pred_label in ml_result["probabilities"]:
        conf = f" ({round(ml_result['probabilities'][pred_label] * 100)}% confidence)"

    goal = next((m["text"] for m in memories if m["category"] == "Goal"), "long-term fitness")
    constraint = next((m["text"] for m in memories if m["category"] == "Constraint"), "no active constraints")
    q = req.message.lower()

    if any(k in q for k in ("remember", "memory", "history")):
        mem_lines = "\n".join(f"• **{m['category']}** ({m['session']}): {m['text']}" for m in memories)
        return (
            f"Active memories for **{req.participant_id.upper()}**:\n{mem_lines}\n\n"
            f"Today's ML classifier predicts **{pred_label}{conf}** "
            f"(RHR **{req.biometrics.resting_heart_rate} bpm**, Sleep **{req.biometrics.sleep_score}/100**)."
        )

    if pred_label == "Deload Needed":
        return (
            f"**ML Classifier: Deload Needed{conf}.** With RHR at **{req.biometrics.resting_heart_rate} bpm** "
            f"(delta {req.biometrics.rhr_delta:+.1f}) and sleep score **{req.biometrics.sleep_score}/100**, "
            f"protect your constraint (**{constraint}**) and **pivot to 20 min light mobility and early sleep** today "
            f"before resuming work toward **{goal}**."
        )

    if pred_label == "Ready":
        return (
            f"**ML Classifier: Ready{conf}.** With sleep score **{req.biometrics.sleep_score}/100** and RHR at "
            f"**{req.biometrics.resting_heart_rate} bpm**, you are cleared to **progress toward {goal}** today "
            f"while keeping **{constraint}** protected."
        )

    return (
        f"**ML Classifier: Moderate{conf}.** Given sleep score **{req.biometrics.sleep_score}/100** and "
        f"**{req.biometrics.steps:,.0f} steps**, stick to **steady Zone 2 and controlled volume** today so you stay "
        f"on track for **{goal}** without aggravating **{constraint}**."
    )


class BiometricInput(BaseModel):
    steps: float = 8500
    resting_heart_rate: float = 59
    sleep_score: float = 78
    deep_sleep_min: float = 82
    rem_sleep_min: float = 90
    sedentary_min: float = 640
    very_active_min: float = 24
    cardio_min: float = 20
    peak_min: float = 5
    rhr_delta: float = 0.0
    sleep_3d_avg: float = 77.0
    steps_3d_avg: float = 8600.0


class CheckinInput(BaseModel):
    day: Optional[str] = None
    dayIndex: Optional[int] = None
    steps: float
    rhr: float
    sleepScore: float
    readiness: int
    fatigue: int
    soreness: int
    stress: int


class ImportDaysInput(BaseModel):
    days: List[Dict[str, Any]]


class NewSessionInput(BaseModel):
    title: Optional[str] = None
    subtitle: Optional[str] = None


class CoachRequest(BaseModel):
    participant_id: str = "p01"
    session_label: str = "Session 4"
    session_id: Optional[str] = None
    message: str
    biometrics: BiometricInput
    history: List[Dict[str, str]] = Field(default_factory=list)
    auto_extract_memory: bool = True
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"


class MemoryItemInput(BaseModel):
    category: str
    text: str
    session: str = "Live"


@app.get("/")
def serve_root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"status": "CareCompanion API running. Place index.html in static/."}


@app.get("/api/model-info")
def get_model_info():
    bundle, metrics = load_ml_bundle()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    llm_info = {
        "llm_configured": bool(openrouter_key or groq_key),
        "llm_provider": "OpenRouter" if openrouter_key else ("Groq" if groq_key else "Local"),
        "llm_model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini") if openrouter_key else os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    }
    if not bundle:
        return {"trained": False, "message": "Run `python train_model.py` first.", **llm_info}
    return {"trained": True, **metrics, **llm_info}


@app.post("/api/predict")
def predict_recovery(payload: BiometricInput):
    result = run_ml_prediction(payload)
    if not result:
        raise HTTPException(status_code=400, detail="Model not trained yet. Run `python train_model.py` first.")
    return result


@app.get("/api/state/{participant_id}")
def get_participant_state(participant_id: str):
    pid = normalize_pid(participant_id)
    app_state = load_app_state()
    mem_store = load_memory_store()
    pdata = app_state.get(pid)
    if not pdata:
        raise HTTPException(status_code=404, detail=f"Participant '{participant_id}' not found.")
    return {
        "participant_id": pid,
        "id": pdata["id"],
        "name": pdata["name"],
        "role": pdata["role"],
        "baseline": pdata["baseline"],
        "days": pdata["days"],
        "sessions": pdata["sessions"],
        "evolutionLog": pdata["evolutionLog"],
        "memories": mem_store.get(pid, []),
    }


@app.post("/api/checkin/{participant_id}")
def log_biometric_checkin(participant_id: str, payload: CheckinInput):
    pid = normalize_pid(participant_id)
    app_state = load_app_state()
    pdata = app_state.get(pid)
    if not pdata:
        raise HTTPException(status_code=404, detail=f"Participant '{participant_id}' not found.")

    days = pdata["days"]
    if payload.dayIndex is not None and 0 <= payload.dayIndex < len(days):
        idx = payload.dayIndex
        existing = days[idx]
        existing["steps"] = payload.steps
        existing["rhr"] = payload.rhr
        existing["sleepScore"] = payload.sleepScore
        existing["readiness"] = payload.readiness
        existing["fatigue"] = payload.fatigue
        existing["soreness"] = payload.soreness
        existing["stress"] = payload.stress
        day_record = existing
    else:
        idx = len(days)
        day_record = {
            "day": payload.day or f"Day {idx + 1}",
            "steps": payload.steps,
            "rhr": payload.rhr,
            "sleepScore": payload.sleepScore,
            "deepMin": 80,
            "remMin": 88,
            "readiness": payload.readiness,
            "fatigue": payload.fatigue,
            "soreness": payload.soreness,
            "stress": payload.stress,
            "fatBurn": 30,
            "cardio": 18,
            "peak": 4,
        }
        days.append(day_record)

    baseline_rhr = float(pdata["baseline"]["rhr"])
    bio = build_biometrics_from_days(days, baseline_rhr, idx)
    ml_result = run_ml_prediction(bio)
    save_app_state(app_state)
    return {
        "day_record": day_record,
        "day_index": idx,
        "ml_prediction": ml_result,
        "days": days,
    }


@app.post("/api/sessions/{participant_id}")
def create_session(participant_id: str, payload: NewSessionInput):
    pid = normalize_pid(participant_id)
    app_state = load_app_state()
    pdata = app_state.get(pid)
    if not pdata:
        raise HTTPException(status_code=404, detail=f"Participant '{participant_id}' not found.")

    sessions = pdata["sessions"]
    days = pdata["days"]
    next_num = len(sessions) + 1
    day_idx = max(0, len(days) - 1)
    day_label = days[day_idx]["day"] if days else "Today"
    prev_plan = copy.deepcopy(sessions[-1]["plan"]) if sessions else []

    new_session = {
        "id": f"S{next_num}",
        "label": f"Session {next_num}",
        "dayIndex": day_idx,
        "dayLabel": day_label,
        "title": payload.title or f"Session {next_num}: Follow-up Check-in",
        "subtitle": payload.subtitle or "Adaptive coaching grounded in persisted memory and biometrics",
        "adaptation": f"Session {next_num} opened against {day_label} biometrics.",
        "plan": prev_plan,
        "messages": [
            {
                "role": "coach",
                "text": f"Session {next_num} initialized for **{pdata['name']}** using {day_label} biometrics. Ask a coaching question or log a check-in to update your protocol.",
                "citations": [f"Session {next_num}", f"Biometrics: {day_label}"],
            }
        ],
    }
    sessions.append(new_session)
    save_app_state(app_state)
    return {"session": new_session, "sessions": sessions}


@app.post("/api/import/{participant_id}")
def import_days(participant_id: str, payload: ImportDaysInput):
    pid = normalize_pid(participant_id)
    app_state = load_app_state()
    pdata = app_state.get(pid)
    if not pdata:
        raise HTTPException(status_code=404, detail=f"Participant '{participant_id}' not found.")
    if not payload.days:
        raise HTTPException(status_code=400, detail="No biometric records provided.")

    pdata["days"] = payload.days
    save_app_state(app_state)
    return {"count": len(payload.days), "days": payload.days}


@app.post("/api/state/{participant_id}/reset")
def reset_full_state(participant_id: str):
    pid = normalize_pid(participant_id)
    app_state = load_app_state()
    mem_store = load_memory_store()
    if pid in DEFAULT_APP_STATE:
        app_state[pid] = copy.deepcopy(DEFAULT_APP_STATE[pid])
        save_app_state(app_state)
    mem_store[pid] = copy.deepcopy(DEFAULT_MEMORY_STORE.get(pid, []))
    save_memory_store(mem_store)
    return {"participant_id": pid, "state": app_state.get(pid), "memories": mem_store[pid]}


@app.get("/api/memory/{participant_id}")
def get_memory(participant_id: str):
    pid = normalize_pid(participant_id)
    store = load_memory_store()
    return {"participant_id": pid, "memories": store.get(pid, [])}


@app.post("/api/memory/{participant_id}")
def add_memory(participant_id: str, item: MemoryItemInput):
    pid = normalize_pid(participant_id)
    store = load_memory_store()
    memories = store.setdefault(pid, [])
    new_item = {
        "id": f"m_{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        "category": item.category,
        "text": item.text,
        "session": item.session,
    }
    memories.insert(0, new_item)
    save_memory_store(store)
    return {"added": new_item, "memories": memories}


@app.delete("/api/memory/{participant_id}/{memory_id}")
def delete_memory(participant_id: str, memory_id: str):
    pid = normalize_pid(participant_id)
    store = load_memory_store()
    memories = store.get(pid, [])
    filtered = [m for m in memories if m.get("id") != memory_id]
    store[pid] = filtered
    save_memory_store(store)
    return {"deleted": memory_id, "memories": filtered}


@app.post("/api/memory/{participant_id}/reset")
def reset_memory(participant_id: str):
    pid = normalize_pid(participant_id)
    store = load_memory_store()
    store[pid] = copy.deepcopy(DEFAULT_MEMORY_STORE.get(pid, []))
    save_memory_store(store)
    return {"participant_id": pid, "memories": store[pid]}


@app.post("/api/coach")
def coach_chat(req: CoachRequest):
    pid = normalize_pid(req.participant_id)
    ml_result = run_ml_prediction(req.biometrics)

    store = load_memory_store()
    memories: List[Dict[str, str]] = store.get(pid, [])
    mem_text = "\n".join(
        [f"- [{m['category']} | {m['session']}] {m['text']}" for m in memories]
    ) or "No prior memories recorded."

    req_key = (req.groq_api_key or "").strip()
    env_openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    env_groq_key = os.getenv("GROQ_API_KEY", "").strip()

    use_openrouter = False
    api_key = ""
    if req_key:
        api_key = req_key
        use_openrouter = req_key.startswith("sk-or-")
    elif env_openrouter_key:
        api_key = env_openrouter_key
        use_openrouter = True
    elif env_groq_key:
        api_key = env_groq_key

    newly_extracted: List[Dict[str, str]] = []
    mode = "offline_fallback"

    if not api_key:
        reply = build_offline_coach_reply(req, ml_result, memories)
    else:
        system_prompt = (
            "You are CareCompanion, an empathetic, evidence-based Agentic Health Coach grounded in PMData Fitbit biometrics.\n"
            f"Current Session: {req.session_label} | Participant: {pid.upper()}\n"
            f"Supervised ML Recovery Prediction: {json.dumps(ml_result)}\n"
            f"Current Fitbit Biometrics: {json.dumps(req.biometrics.model_dump())}\n"
            f"Persistent Agent Memory from Past Sessions:\n{mem_text}\n\n"
            "Instructions:\n"
            "1. Explicitly reference relevant past memories, today's ML recovery prediction, and Fitbit shifts.\n"
            "2. Adapt your advice based on prior sessions rather than giving generic one-off tips.\n"
            "3. Keep your response concise (3-5 sentences) with clear bolded action items."
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for turn in req.history[-6:]:
            role = "assistant" if turn.get("role") in ("coach", "assistant") else "user"
            content = str(turn.get("text") or turn.get("content") or "").strip()
            if content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": req.message})

        if use_openrouter:
            model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            provider_label = "OpenRouter"
            mode = "openrouter_live"
        else:
            model_name = (req.groq_model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")).strip()
            client = Groq(api_key=api_key)
            provider_label = "Groq"
            mode = "groq_live"

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.4,
                max_tokens=450,
            )
            reply = completion.choices[0].message.content or ""
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"{provider_label} API error: {exc}") from exc

        if req.auto_extract_memory:
            newly_extracted = extract_new_memories_with_groq(
                client=client,
                model_name=model_name,
                user_message=req.message,
                session_label=req.session_label,
                existing_memories=memories,
            )
            if newly_extracted:
                memories = newly_extracted + memories
                store[pid] = memories
                save_memory_store(store)

    if req.session_id:
        app_state = load_app_state()
        pdata = app_state.get(pid)
        if pdata:
            for sess in pdata.get("sessions", []):
                if sess.get("id") == req.session_id:
                    pred_label = ml_result["predicted_recovery_state"] if ml_result else "Moderate"
                    if mode == "openrouter_live":
                        mode_tag = "OpenRouter"
                    elif mode == "groq_live":
                        mode_tag = "Groq Llama-3.3-70B"
                    else:
                        mode_tag = "Local ML Coach"
                    sess.setdefault("messages", []).append({"role": "user", "text": req.message})
                    sess["messages"].append({
                        "role": "coach",
                        "text": reply,
                        "citations": [mode_tag, f"ML: {pred_label}", f"Memory: {len(memories)}"],
                    })
                    save_app_state(app_state)
                    break

    return {
        "reply": reply,
        "ml_prediction": ml_result,
        "memories_used": len(memories),
        "memories": memories,
        "newly_extracted_memories": newly_extracted,
        "mode": mode,
    }
