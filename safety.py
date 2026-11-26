"""Red-flag screen that runs before the coach answers.

A wellness coach should not give training advice to someone describing a possible medical
emergency. Messages are checked with plain rules (no model, so it works offline and cannot
be talked out of it). A flagged message gets a fixed escalation reply instead of coaching,
and the event is saved as a constraint, so later sessions know to ask about medical clearance.

Negated mentions ("no chest pain", "never fainted") do not trigger.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

# (kind, pattern). Patterns are matched case-insensitively against the whole message.
RED_FLAGS = [
    ("self_harm", r"\b(kill(ing)? myself|suicid\w*|end (it all|my life)|self[- ]?harm|hurt(ing)? myself|don'?t want to (live|be alive))\b"),
    ("chest_pain", r"\bchest (pain|pressure|tightness|hurts?|ache|aching|discomfort)|(pain|pressure|tightness) in (my|the) chest\b"),
    ("fainting", r"\b(faint(ed|ing)?|passed out|pass(ing)? out|black(ed)? out|lost consciousness|collaps(e|ed|ing))\b"),
    ("breathing", r"\b(can'?t breathe|cannot breathe|struggling to breathe|short(ness)? of breath at rest|gasping for (air|breath))\b"),
    ("heart_rhythm", r"\b(heart (is )?(racing|pounding|fluttering|skipping)( beats)? (at rest|while resting|for no reason)|irregular heart ?beat|palpitations)\b"),
    ("stroke_signs", r"\b(slurr(ed|ing)|(face|mouth|eyelid)\b[\w\s]{0,25}\bdroop\w*|droop\w* (face|mouth)|numb(ness)? (on|in|down) one side|"
                     r"one side of (my|the) (body|face)[\w\s]{0,15}(numb|weak)|sudden (weakness|numbness|confusion)|worst headache)"),
    ("bleeding", r"\b(coughing (up )?blood|blood in (my )?(urine|pee|stool|vomit)|vomiting blood)\b"),
]
_COMPILED = [(kind, re.compile(p, re.I)) for kind, p in RED_FLAGS]
_NEGATION = re.compile(r"\b(no|not|never|without|don'?t have|didn'?t|haven'?t|hasn'?t|isn'?t|wasn'?t|zero)\b[\w\s,']{0,20}$", re.I)

EMERGENCY = ("If it is severe, sudden or getting worse, **call your local emergency number now** "
             "(112 in India and Europe, 911 in the US).")

REPLIES = {
    "self_harm": ("I'm really sorry you're feeling this way, and I'm glad you said it. I'm a training coach and can't help "
                  "with this safely, but people can: **please call a crisis line now**, such as **Tele-MANAS 14416** in "
                  "India or **988** in the US, or your local emergency number. If you can, tell someone you trust "
                  "and stay with them."),
    "chest_pain": "Chest pain or pressure needs a doctor before any more training.",
    "fainting": "Fainting or passing out needs a doctor before any more training.",
    "breathing": "Trouble breathing needs a doctor before any more training.",
    "heart_rhythm": "An irregular or racing heartbeat at rest needs a doctor before any more training.",
    "stroke_signs": "Slurred speech, a drooping face, one-sided numbness or a sudden severe headache can be a stroke.",
    "bleeding": "Coughing up or passing blood needs a doctor before any more training.",
}

MEMORY_TEXT = {
    "self_harm": "Said they were struggling with thoughts of self-harm; check in on how they are before any training talk",
    "chest_pain": "Reported chest pain or pressure; needs medical clearance before hard training",
    "fainting": "Reported fainting or passing out; needs medical clearance before hard training",
    "breathing": "Reported trouble breathing; needs medical clearance before hard training",
    "heart_rhythm": "Reported an irregular or racing heartbeat at rest; needs medical clearance before hard training",
    "stroke_signs": "Reported possible stroke signs; needs medical clearance before any training",
    "bleeding": "Reported coughing up or passing blood; needs medical clearance before training",
}


def check(message: str) -> Optional[Dict[str, str]]:
    """The first red flag in the message, or None. Returns {kind, reply, memory}."""
    for kind, pat in _COMPILED:
        for m in pat.finditer(message):
            before = message[max(0, m.start() - 40):m.start()]
            if _NEGATION.search(before):
                continue
            reply = REPLIES[kind]
            if kind != "self_harm":
                reply = (f"**Please stop training today.** {reply} {EMERGENCY} Otherwise, **see a doctor before your next "
                         "session** and tell them what you told me. I've noted this, and I'll check you've been cleared "
                         "before we plan anything hard.")
            return {"kind": kind, "reply": reply, "memory": MEMORY_TEXT[kind]}
    return None
