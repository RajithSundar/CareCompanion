"""Seed data for the three fictional demo personas.

Every visitor gets a private copy of this data (see sandbox.py). Day records are
illustrative and are not taken from the PMData sample.
"""

from typing import Any, Dict, List

DEFAULT_MEMORY_STORE: Dict[str, List[Dict[str, str]]] = {
    "p01": [
        {"id": "p01_m1", "category": "Goal", "text": "Training for half-marathon in 8 weeks", "session": "S1"},
        {"id": "p01_m2", "category": "Constraint", "text": "Left knee tightness after high-speed work; cap all-out efforts", "session": "S1"},
        {"id": "p01_m3", "category": "Baseline", "text": "Morning workout preference; baseline resting heart rate 58 bpm", "session": "S1"},
    ],
    "p07": [
        {"id": "p07_m1", "category": "Goal", "text": "Boost afternoon energy & reduce desk stiffness", "session": "S1"},
        {"id": "p07_m2", "category": "Constraint", "text": "No 5 AM workouts; prefers lunchtime walks & evening yoga", "session": "S1"},
        {"id": "p07_m3", "category": "Baseline", "text": "Baseline resting heart rate 65 bpm; desk-bound tech lead schedule", "session": "S1"},
    ],
    "p12": [
        {"id": "p12_m1", "category": "Goal", "text": "Increase lower-body strength & hypertrophy while protecting sleep", "session": "S1"},
        {"id": "p12_m2", "category": "Constraint", "text": "Lower-back tightness after heavy deadlifts", "session": "S1"},
        {"id": "p12_m3", "category": "Baseline", "text": "Baseline resting heart rate 55 bpm; 4-day upper/lower split", "session": "S1"},
    ],
}


DEFAULT_APP_STATE: Dict[str, Dict[str, Any]] = {
    "p01": {
        "id": "P01",
        "name": "Alex Rivera",
        "role": "Half-Marathon Prep, Morning Runner",
        "baseline": {"steps": 9400, "rhr": 58, "sleepScore": 79, "readiness": 7},
        "days": [
            {"day": "Day 1", "steps": 9650, "rhr": 58, "sleepScore": 80, "deepMin": 88, "remMin": 95, "readiness": 8, "fatigue": 4, "soreness": 4, "stress": 4, "fatBurn": 38, "cardio": 26, "peak": 8},
            {"day": "Day 2", "steps": 10800, "rhr": 59, "sleepScore": 77, "deepMin": 82, "remMin": 90, "readiness": 7, "fatigue": 3, "soreness": 4, "stress": 4, "fatBurn": 42, "cardio": 31, "peak": 11},
            {"day": "Day 3", "steps": 11400, "rhr": 60, "sleepScore": 72, "deepMin": 71, "remMin": 84, "readiness": 6, "fatigue": 3, "soreness": 3, "stress": 3, "fatBurn": 45, "cardio": 34, "peak": 14},
            {"day": "Day 4", "steps": 8900, "rhr": 62, "sleepScore": 64, "deepMin": 58, "remMin": 74, "readiness": 5, "fatigue": 2, "soreness": 2, "stress": 2, "fatBurn": 32, "cardio": 18, "peak": 4},
            {"day": "Day 5", "steps": 5400, "rhr": 63, "sleepScore": 59, "deepMin": 49, "remMin": 68, "readiness": 4, "fatigue": 2, "soreness": 2, "stress": 2, "fatBurn": 21, "cardio": 6, "peak": 0},
            {"day": "Day 6", "steps": 6100, "rhr": 62, "sleepScore": 68, "deepMin": 66, "remMin": 81, "readiness": 5, "fatigue": 3, "soreness": 3, "stress": 3, "fatBurn": 28, "cardio": 8, "peak": 0},
            {"day": "Day 7", "steps": 7400, "rhr": 60, "sleepScore": 75, "deepMin": 79, "remMin": 89, "readiness": 6, "fatigue": 3, "soreness": 3, "stress": 4, "fatBurn": 34, "cardio": 14, "peak": 2},
            {"day": "Day 8", "steps": 8500, "rhr": 59, "sleepScore": 79, "deepMin": 85, "remMin": 94, "readiness": 7, "fatigue": 4, "soreness": 4, "stress": 4, "fatBurn": 39, "cardio": 22, "peak": 5},
            {"day": "Day 9", "steps": 9200, "rhr": 58, "sleepScore": 82, "deepMin": 92, "remMin": 98, "readiness": 8, "fatigue": 4, "soreness": 4, "stress": 4, "fatBurn": 41, "cardio": 25, "peak": 7},
            {"day": "Day 10", "steps": 10350, "rhr": 57, "sleepScore": 84, "deepMin": 96, "remMin": 104, "readiness": 8, "fatigue": 4, "soreness": 4, "stress": 4, "fatBurn": 44, "cardio": 29, "peak": 9},
        ],
        "sessions": [
            {
                "id": "S1", "label": "Session 1", "dayIndex": 0, "dayLabel": "Day 1",
                "title": "Session 1: Getting started", "subtitle": "Half-marathon goal, knee and mornings",
                "adaptation": "Captured half-marathon goal, morning preference, and a cap on all-out efforts for the left knee.",
                "plan": [
                    {"id": "p1_1", "text": "35 min easy morning run", "done": True},
                    {"id": "p1_2", "text": "Hit 9,500+ daily steps", "done": True},
                    {"id": "p1_3", "text": "7.5 hours of sleep (lights out 10:30 PM)", "done": True},
                    {"id": "p1_4", "text": "Note how the knee feels after runs", "done": True},
                ],
                "messages": [
                    {"role": "user", "text": "I'm training for a half-marathon in 8 weeks. My left knee sometimes feels tight after speed work, and I prefer morning workouts. What should I focus on this week?"},
                    {"role": "coach", "text": "You're starting from a good place: **9,650 steps**, **resting heart rate 58**, **sleep score 80** and you feel **8/10 ready**. This week, do **35 to 40 minute morning runs at a comfortable pace** and aim for **about 10,000 steps** a day. Because of your knee, keep all-out efforts **under 10 minutes a day** and note how the knee feels after each run.", "citations": ["Scripted example", "Starting point saved", "Goal: half-marathon", "Watch: left knee"]},
                ],
            },
            {
                "id": "S2", "label": "Session 2", "dayIndex": 4, "dayLabel": "Day 5",
                "title": "Session 2: Tired, heart rate up", "subtitle": "Bad sleep and a sore knee, so the plan changes",
                "adaptation": "Resting heart rate rose from 58 to 63 and sleep fell to 59, so the hard run became 20 minutes of stretching.",
                "plan": [
                    {"id": "p2_1", "text": "20 min gentle stretching instead of the run", "done": True},
                    {"id": "p2_2", "text": "Keep it easy: no all-out efforts", "done": True},
                    {"id": "p2_3", "text": "8 hours of sleep tonight", "done": False},
                    {"id": "p2_4", "text": "Ice the knee and do light leg exercises", "done": True},
                ],
                "messages": [
                    {"role": "user", "text": "I feel sluggish today, but I have a tempo run on my calendar. Should I push through it?"},
                    {"role": "coach", "text": "**Skip the hard run today.** The prediction says **rest day**: your resting heart rate rose from **58 to 63** and your sleep score fell from **80 to 59**. With your **knee** from last time and the **soreness** you're feeling, do **20 minutes of gentle stretching** instead and aim for **8 hours of sleep**.", "citations": ["Scripted example", "Tomorrow: Rest day", "Remembered: left knee", "Resting heart rate up 5"]},
                ],
            },
            {
                "id": "S3", "label": "Session 3", "dayIndex": 9, "dayLabel": "Day 10",
                "title": "Session 3: Back on track", "subtitle": "The rest days worked, so training builds again",
                "adaptation": "Resting heart rate back to 57 and sleep back to 84, so harder running is back on.",
                "plan": [
                    {"id": "p3_1", "text": "40 min morning run, 12 min of it harder", "done": False},
                    {"id": "p3_2", "text": "About 10,000 steps a day", "done": True},
                    {"id": "p3_3", "text": "7.5 to 8 hours of sleep", "done": True},
                    {"id": "p3_4", "text": "5 min knee warm-up before runs", "done": False},
                ],
                "messages": [
                    {"role": "user", "text": "My legs feel much fresher after taking those lighter days. Am I ready to ramp back up toward my half-marathon?"},
                    {"role": "coach", "text": "**The rest days worked.** Your resting heart rate is back to **57**, sleep score is up to **84**, and the soreness has **eased**. Go back to a **40 minute morning run** with **12 minutes at a harder pace**, and do a **5 minute knee warm-up** first.", "citations": ["Scripted example", "Remembered: rest days helped", "Remembered: mornings"]},
                ],
            },
            {
                "id": "S4", "label": "Session 4", "dayIndex": 9, "dayLabel": "Day 10+",
                "title": "Session 4: Today", "subtitle": "Ask anything, the coach remembers sessions 1 to 3",
                "adaptation": "The coach has your last three sessions and 10 days of numbers.",
                "plan": [
                    {"id": "p4_1", "text": "40 min morning run, 12 min of it harder", "done": False},
                    {"id": "p4_2", "text": "About 10,000 steps a day", "done": True},
                    {"id": "p4_3", "text": "7.5 to 8 hours of sleep", "done": True},
                    {"id": "p4_4", "text": "5 min knee warm-up before runs", "done": False},
                ],
                "messages": [
                    {"role": "coach", "text": "Welcome back, Alex. I remember your half-marathon goal, your knee and our last three chats. Ask me anything, or log today's numbers and I'll adjust your plan.", "citations": ["Scripted example", "Remembers 3 sessions", "Days 1 to 10"]},
                ],
            },
        ],
        "evolutionLog": [
            {"session": "S1", "text": "Saved: half-marathon goal, mornings, left knee."},
            {"session": "S2", "text": "Rest day predicted, so hard runs paused."},
            {"session": "S3", "text": "Heart rate and sleep recovered, so harder running is back."},
        ],
    },
    "p07": {
        "id": "P07",
        "name": "Maya Lin",
        "role": "Tech Lead, Desk-Bound + Evening Yoga",
        "baseline": {"steps": 6400, "rhr": 65, "sleepScore": 74, "readiness": 6},
        "days": [
            {"day": "Day 1", "steps": 6200, "rhr": 65, "sleepScore": 74, "deepMin": 72, "remMin": 81, "readiness": 6, "fatigue": 3, "soreness": 4, "stress": 3, "fatBurn": 24, "cardio": 10, "peak": 1},
            {"day": "Day 2", "steps": 5800, "rhr": 66, "sleepScore": 71, "deepMin": 68, "remMin": 76, "readiness": 6, "fatigue": 3, "soreness": 4, "stress": 2, "fatBurn": 20, "cardio": 8, "peak": 0},
            {"day": "Day 3", "steps": 4900, "rhr": 67, "sleepScore": 66, "deepMin": 59, "remMin": 70, "readiness": 5, "fatigue": 2, "soreness": 3, "stress": 2, "fatBurn": 16, "cardio": 5, "peak": 0},
            {"day": "Day 4", "steps": 4100, "rhr": 69, "sleepScore": 60, "deepMin": 48, "remMin": 62, "readiness": 4, "fatigue": 2, "soreness": 3, "stress": 1, "fatBurn": 12, "cardio": 3, "peak": 0},
            {"day": "Day 5", "steps": 3900, "rhr": 70, "sleepScore": 57, "deepMin": 44, "remMin": 58, "readiness": 3, "fatigue": 1, "soreness": 3, "stress": 1, "fatBurn": 10, "cardio": 2, "peak": 0},
            {"day": "Day 6", "steps": 5500, "rhr": 68, "sleepScore": 67, "deepMin": 61, "remMin": 72, "readiness": 5, "fatigue": 3, "soreness": 4, "stress": 2, "fatBurn": 22, "cardio": 7, "peak": 0},
            {"day": "Day 7", "steps": 6900, "rhr": 67, "sleepScore": 73, "deepMin": 70, "remMin": 80, "readiness": 6, "fatigue": 3, "soreness": 4, "stress": 3, "fatBurn": 28, "cardio": 12, "peak": 2},
            {"day": "Day 8", "steps": 7600, "rhr": 65, "sleepScore": 78, "deepMin": 80, "remMin": 88, "readiness": 7, "fatigue": 4, "soreness": 4, "stress": 4, "fatBurn": 33, "cardio": 16, "peak": 3},
            {"day": "Day 9", "steps": 8100, "rhr": 64, "sleepScore": 81, "deepMin": 85, "remMin": 92, "readiness": 8, "fatigue": 4, "soreness": 5, "stress": 4, "fatBurn": 36, "cardio": 18, "peak": 4},
            {"day": "Day 10", "steps": 8450, "rhr": 64, "sleepScore": 83, "deepMin": 89, "remMin": 95, "readiness": 8, "fatigue": 4, "soreness": 5, "stress": 4, "fatBurn": 38, "cardio": 20, "peak": 4},
        ],
        "sessions": [
            {
                "id": "S1", "label": "Session 1", "dayIndex": 0, "dayLabel": "Day 1",
                "title": "Session 1: Getting started", "subtitle": "Stiff from meetings, wants more energy",
                "adaptation": "Set a 7,500 step goal with short walks, plus 20 min of evening yoga.",
                "plan": [
                    {"id": "m1_1", "text": "Two 10 min walks after lunch", "done": True},
                    {"id": "m1_2", "text": "Reach 7,500 daily steps", "done": False},
                    {"id": "m1_3", "text": "20 min evening yoga", "done": True},
                    {"id": "m1_4", "text": "No screens 45 min before bed", "done": False},
                ],
                "messages": [
                    {"role": "user", "text": "I sit in back-to-back meetings all day and feel stiff by 6 PM. I want more energy without waking up at 5 AM."},
                    {"role": "coach", "text": "Today you're at **6,200 steps**, **resting heart rate 65** and **sleep score 74**. Add two **10 minute walks** between meetings to reach **7,500 steps**, and keep **20 minutes of yoga in the evening**, so no early mornings needed.", "citations": ["Scripted example", "Remembered: evenings only", "Goal: 7,500 steps"]},
                ],
            },
            {
                "id": "S2", "label": "Session 2", "dayIndex": 4, "dayLabel": "Day 5",
                "title": "Session 2: Launch week stress", "subtitle": "High stress and broken sleep",
                "adaptation": "Rest day predicted, so the step goal paused and breathing exercises replaced cardio.",
                "plan": [
                    {"id": "m2_1", "text": "Just 5,000 gentle steps today", "done": True},
                    {"id": "m2_2", "text": "10 min slow breathing", "done": True},
                    {"id": "m2_3", "text": "No hard workout tonight", "done": True},
                    {"id": "m2_4", "text": "No caffeine after 1:00 PM", "done": False},
                ],
                "messages": [
                    {"role": "user", "text": "Product launch week is brutal. Should I force a hard cardio session tonight to catch up?"},
                    {"role": "coach", "text": "**Skip hard cardio tonight.** The prediction says **rest day**: your **resting heart rate rose to 70**, your sleep score dropped to **57**, and your stress is **very high**. Pause the step goal for now and do **10 minutes of slow breathing and gentle yoga** instead.", "citations": ["Scripted example", "Tomorrow: Rest day", "Changed: step goal paused"]},
                ],
            },
            {
                "id": "S3", "label": "Session 3", "dayIndex": 9, "dayLabel": "Day 10",
                "title": "Session 3: Bouncing back", "subtitle": "Sleep is better, time to build up",
                "adaptation": "Sleep back to 83 and heart rate back to 64, so the step goal went up to 8,500.",
                "plan": [
                    {"id": "m3_1", "text": "8,500 steps a day", "done": True},
                    {"id": "m3_2", "text": "Two brisk 25 min walks a week", "done": True},
                    {"id": "m3_3", "text": "Keep the evening yoga", "done": True},
                    {"id": "m3_4", "text": "Check in on stress each week", "done": False},
                ],
                "messages": [
                    {"role": "user", "text": "Launch is over and I slept well the last two nights. What is my next step?"},
                    {"role": "coach", "text": "Your sleep score jumped from **57 to 83**, your resting heart rate is back to **64**, and you reached **8,450 steps**. Make **8,500 steps** the daily goal and add **two brisk 25 minute walks** a week.", "citations": ["Scripted example", "Recovered", "New goal: 8,500 steps"]},
                ],
            },
            {
                "id": "S4", "label": "Session 4", "dayIndex": 9, "dayLabel": "Day 10+",
                "title": "Session 4: Today", "subtitle": "Ask anything, the coach remembers sessions 1 to 3",
                "adaptation": "The coach has your last three sessions and 10 days of numbers.",
                "plan": [
                    {"id": "m4_1", "text": "8,500 steps a day", "done": True},
                    {"id": "m4_2", "text": "Two brisk 25 min walks a week", "done": True},
                    {"id": "m4_3", "text": "Keep the evening yoga", "done": True},
                    {"id": "m4_4", "text": "Check in on stress each week", "done": False},
                ],
                "messages": [
                    {"role": "coach", "text": "Welcome back, Maya. I remember your evening-only schedule, launch week and our last three chats. Ask me anything, or log today's numbers and I'll adjust your plan.", "citations": ["Scripted example", "Remembers 3 sessions"]},
                ],
            },
        ],
        "evolutionLog": [
            {"session": "S1", "text": "Saved: evenings only, less stiffness, more energy."},
            {"session": "S2", "text": "Rest day predicted, so cardio became breathing exercises."},
            {"session": "S3", "text": "Recovered, so the step goal went up to 8,500."},
        ],
    },
    "p12": {
        "id": "P12",
        "name": "Jordan Patel",
        "role": "Strength Athlete, Hypertrophy Focus",
        "baseline": {"steps": 7800, "rhr": 55, "sleepScore": 81, "readiness": 8},
        "days": [
            {"day": "Day 1", "steps": 8100, "rhr": 55, "sleepScore": 82, "deepMin": 94, "remMin": 99, "readiness": 8, "fatigue": 4, "soreness": 4, "stress": 4, "fatBurn": 30, "cardio": 18, "peak": 6},
            {"day": "Day 2", "steps": 8400, "rhr": 56, "sleepScore": 80, "deepMin": 90, "remMin": 93, "readiness": 8, "fatigue": 4, "soreness": 3, "stress": 4, "fatBurn": 34, "cardio": 20, "peak": 7},
            {"day": "Day 3", "steps": 7900, "rhr": 57, "sleepScore": 76, "deepMin": 82, "remMin": 88, "readiness": 7, "fatigue": 3, "soreness": 3, "stress": 4, "fatBurn": 29, "cardio": 16, "peak": 5},
            {"day": "Day 4", "steps": 7200, "rhr": 59, "sleepScore": 69, "deepMin": 68, "remMin": 79, "readiness": 5, "fatigue": 2, "soreness": 2, "stress": 3, "fatBurn": 24, "cardio": 11, "peak": 2},
            {"day": "Day 5", "steps": 6500, "rhr": 61, "sleepScore": 63, "deepMin": 56, "remMin": 71, "readiness": 4, "fatigue": 2, "soreness": 1, "stress": 3, "fatBurn": 19, "cardio": 6, "peak": 0},
            {"day": "Day 6", "steps": 7100, "rhr": 59, "sleepScore": 72, "deepMin": 74, "remMin": 84, "readiness": 6, "fatigue": 3, "soreness": 3, "stress": 4, "fatBurn": 25, "cardio": 10, "peak": 1},
            {"day": "Day 7", "steps": 7800, "rhr": 57, "sleepScore": 78, "deepMin": 86, "remMin": 91, "readiness": 7, "fatigue": 4, "soreness": 3, "stress": 4, "fatBurn": 31, "cardio": 15, "peak": 4},
            {"day": "Day 8", "steps": 8300, "rhr": 56, "sleepScore": 82, "deepMin": 91, "remMin": 96, "readiness": 8, "fatigue": 4, "soreness": 4, "stress": 4, "fatBurn": 33, "cardio": 19, "peak": 5},
            {"day": "Day 9", "steps": 8600, "rhr": 55, "sleepScore": 84, "deepMin": 96, "remMin": 101, "readiness": 8, "fatigue": 4, "soreness": 4, "stress": 5, "fatBurn": 35, "cardio": 21, "peak": 6},
            {"day": "Day 10", "steps": 8900, "rhr": 54, "sleepScore": 86, "deepMin": 99, "remMin": 105, "readiness": 9, "fatigue": 5, "soreness": 5, "stress": 5, "fatBurn": 37, "cardio": 23, "peak": 7},
        ],
        "sessions": [
            {
                "id": "S1", "label": "Session 1", "dayIndex": 0, "dayLabel": "Day 1",
                "title": "Session 1: Getting started", "subtitle": "New strength plan and a sensitive lower back",
                "adaptation": "Set a 4-day lifting plan and a watch on the lower back.",
                "plan": [
                    {"id": "j1_1", "text": "Lift 4 days a week, upper and lower body", "done": True},
                    {"id": "j1_2", "text": "About 160 g of protein a day", "done": True},
                    {"id": "j1_3", "text": "Note how the lower back feels", "done": True},
                    {"id": "j1_4", "text": "8,000 steps on rest days", "done": True},
                ],
                "messages": [
                    {"role": "user", "text": "Starting a 4-day hypertrophy block. My lower back sometimes fatigues after heavy deadlifts. How should I manage volume?"},
                    {"role": "coach", "text": "You're starting well: **resting heart rate 55**, **sleep score 82**, and you feel **8/10 ready**. Lift **4 days a week**, alternating upper and lower body, at an effort of **7 to 8 out of 10**, and note how your lower back feels after deadlift days.", "citations": ["Scripted example", "Starting point saved", "Watch: lower back"]},
                ],
            },
            {
                "id": "S2", "label": "Session 2", "dayIndex": 4, "dayLabel": "Day 5",
                "title": "Session 2: Very sore", "subtitle": "Sore back and legs, heart rate up after a heavy day",
                "adaptation": "Rest day predicted, so heavy squats became lighter machine work.",
                "plan": [
                    {"id": "j2_1", "text": "Leg press and curls instead of heavy squats", "done": True},
                    {"id": "j2_2", "text": "Effort no higher than 6/10 today", "done": True},
                    {"id": "j2_3", "text": "15 min hip and upper-back stretching", "done": True},
                    {"id": "j2_4", "text": "45 min extra sleep tonight", "done": False},
                ],
                "messages": [
                    {"role": "user", "text": "Lower back and hamstrings are wrecked (really sore), but today is heavy squat day. Should I load up?"},
                    {"role": "coach", "text": "**Don't do heavy squats today.** The prediction says **rest day**: your **resting heart rate jumped from 55 to 61** and you're **very sore**, which fits the lower-back problem you mentioned last time. Do **lighter machine exercises** instead, at an effort of about **6 out of 10**.", "citations": ["Scripted example", "Tomorrow: Rest day", "Remembered: lower back"]},
                ],
            },
            {
                "id": "S3", "label": "Session 3", "dayIndex": 9, "dayLabel": "Day 10",
                "title": "Session 3: Stronger than before", "subtitle": "Feeling 9/10 ready, heart rate at its best",
                "adaptation": "Heart rate down to 54 and sleep up to 86, so heavier lifting is back on.",
                "plan": [
                    {"id": "j3_1", "text": "Big lifts again, effort 8/10", "done": False},
                    {"id": "j3_2", "text": "Back-friendly deadlifts only", "done": True},
                    {"id": "j3_3", "text": "8,500+ steps a day", "done": True},
                    {"id": "j3_4", "text": "Keep logging how you feel after sessions", "done": True},
                ],
                "messages": [
                    {"role": "user", "text": "Soreness is completely gone today. Ready to push for a PR?"},
                    {"role": "coach", "text": "**Go ahead and lift heavier.** Your resting heart rate hit a 10-day best of **54** and your sleep score is **86**. Push your top sets to an effort of **8 out of 10**, and use **back-friendly deadlift variations** (trap-bar or slow Romanian deadlifts) to protect your lower back.", "citations": ["Scripted example", "Remembered: protect the lower back"]},
                ],
            },
            {
                "id": "S4", "label": "Session 4", "dayIndex": 9, "dayLabel": "Day 10+",
                "title": "Session 4: Today", "subtitle": "Ask anything, the coach remembers sessions 1 to 3",
                "adaptation": "The coach has your last three sessions and 10 days of numbers.",
                "plan": [
                    {"id": "j4_1", "text": "Big lifts again, effort 8/10", "done": False},
                    {"id": "j4_2", "text": "Back-friendly deadlifts only", "done": True},
                    {"id": "j4_3", "text": "8,500+ steps a day", "done": True},
                    {"id": "j4_4", "text": "Keep logging how you feel after sessions", "done": True},
                ],
                "messages": [
                    {"role": "coach", "text": "Welcome back, Jordan. I remember your strength plan, your lower back and our last three chats. Ask me anything, or log today's numbers and I'll adjust your plan.", "citations": ["Scripted example", "Remembers 3 sessions"]},
                ],
            },
        ],
        "evolutionLog": [
            {"session": "S1", "text": "Saved: strength goal and lower-back risk."},
            {"session": "S2", "text": "Rest day predicted, so heavy squats became machines."},
            {"session": "S3", "text": "Recovered and stronger, so heavier lifting is back."},
        ],
    },
}
