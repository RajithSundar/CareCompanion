# CareCompanion

A supervised learning demo for next-day recovery forecasting, plus a small coaching dashboard around it.

- **Task:** given one day's biometrics, forecast the recovery state for the next day (Ready, Moderate, or Deload Needed).
- **Model:** RandomForest, selected by nested leave-one-participant-out validation.
- **App:** a FastAPI backend and a static dashboard. An optional LLM (OpenRouter or Groq) writes the coaching text. Without a key, a rule-based reply is used.

## Results

Nested leave-one-participant-out on `data/pmdata_sample.csv` (464 forecast rows, 16 participants). The numbers come from `models/model_metrics.json`.

| Model | Accuracy | Macro F1 | Brier |
|---|---|---|---|
| RandomForest (selected on macro F1) | 0.591 | 0.510 | 0.506 |
| GradientBoosting | 0.647 | 0.483 | 0.488 |
| LogisticRegression | 0.565 | 0.488 | 0.550 |
| Baseline: majority class | 0.588 | 0.247 | |
| Baseline: persistence (tomorrow = today) | 0.629 | 0.542 | |

**No model beats the persistence baseline on macro F1.** GradientBoosting beats it on accuracy (0.647 vs 0.629) and has the best Brier score, but it rarely predicts Ready. Adding today's readiness, fatigue, soreness and stress as inputs did not close the gap. On this sample, today's state carries almost all the information about tomorrow. The sample is synthetic (see below), so the real test is the full PMData dataset.

Selected model, per class:

| Class | Precision | Recall | Support |
|---|---|---|---|
| Deload Needed | 0.78 | 0.70 | 273 |
| Moderate | 0.47 | 0.39 | 155 |
| Ready | 0.26 | 0.64 | 36 |

Ready is over-predicted (class weighting), which is why its precision is low.

Permutation importance puts `rhr_delta` first, then `fatigue`, `readiness` and `cardio_min`. The standard deviations are larger than the means, so the ranking is tentative.

## Method

- **Target.** The recovery state on day t+1, paired with day t's features. Pairs that are not on consecutive calendar days are dropped.
- **Labels are rule-based.** Ready if readiness >= 7 and fatigue <= 3; Deload Needed if readiness <= 4 or fatigue >= 4; otherwise Moderate. Not clinical ground truth.
- **Features** (`features.py`, shared by training and serving): steps, resting heart rate, sleep score, deep and REM minutes, cardio and peak minutes, today's readiness, fatigue, soreness and stress, `rhr_delta` (today minus the 7-day mean), and 3-day means of sleep, steps, readiness and fatigue. Windows are calendar-based, so a gap in the record does not stretch them. Today's readiness and fatigue define today's label, not tomorrow's, so they are fair inputs; the persistence baseline uses them too.
- **Validation.** The outer loop holds out one participant; the inner loop tunes hyperparameters with grouped 3-fold CV on the rest. Imputation is inside the pipeline.
- **Serving.** The backend builds features from stored day records by calling the same `features.py` code. A test checks they match.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python train_model.py            # optional: retrains and rewrites models/
uvicorn app:app --port 8000
```

Open `http://127.0.0.1:8000`.

To run the tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## LLM configuration

Copy `.env.example` to `.env` and set one key. The server uses OpenRouter if `OPENROUTER_API_KEY` is set, otherwise Groq if `GROQ_API_KEY` is set. Keys stay on the server; the browser never sees or sends one. Without a key the coach gives a rule-based reply built from the forecast and the saved memories.

Optional limits (defaults in brackets): `CC_COACH_PER_10MIN` (15) and `CC_COACH_PER_DAY` (60) LLM replies per IP, `CC_LLM_GLOBAL_PER_DAY` (500) in total, `CC_WRITES_PER_MIN` (60) writes per IP.

## Hosting

- Each browser gets a private sandbox: a random id in an HttpOnly, SameSite=Strict cookie and its own JSON file under `data/sandboxes/`. Visitors cannot see or change each other's data. Reads never create files; sandboxes unused for 14 days are deleted, and at most 2,000 are kept.
- Writes are atomic and locked per sandbox. Request bodies are capped at 256 KB, writes must be JSON, and every field is range-checked.
- The page sends a strict Content-Security-Policy (no inline or third-party scripts). Chart.js and Lucide are vendored in `static/vendor/`; Tailwind is built ahead of time (`frontend/README.md`).
- `/docs` and `/openapi.json` are disabled.
- On this machine the app and its Tailscale Funnel run as systemd user services: `systemctl --user status carecompanion carecompanion-tailscaled`. Logs: `journalctl --user -u carecompanion -f`.

## Data and limitations

- `data/pmdata_sample.csv` is a synthetic 480-row sample with the PMData schema (16 participants, 30 days each). The full PMData dataset is at [datasets.simula.no/pmdata](https://datasets.simula.no/pmdata) (non-commercial licence; cite the paper). Put participant folders under `data/pmdata_raw/` and rerun `train_model.py`. The loader reads `pmsys/wellness.csv` and the Fitbit files `sleep_score.csv`, `resting_heart_rate.json`, `steps.json`, `sleep.json` and `time_in_heart_rate_zones.json`, and raises if one is missing.
- The three demo personas (`demo_data.py`) are fictional and sit at higher readiness than most of the sample, so their forecasts are illustrations.
- Forecasts and coaching are a demo, not medical advice. Probabilities are not calibrated; the Brier score is reported so you can judge them.

## Files

- `train_model.py`: data loading, nested validation, baselines, permutation importance, and the saved model.
- `app.py`: FastAPI routes, validation, rate limits and the coach.
- `features.py`: feature definitions shared by training and serving.
- `sandbox.py`, `ratelimit.py`: per-visitor storage and rate limiting.
- `demo_data.py`: the three demo personas.
- `static/`: the dashboard (`index.html`, `app.js`, `tracker.js`, built `app.css`).
- `test_app.py`: API, sandbox, security, feature-parity and coach tests.
- `models/`: the saved model and its metrics.
