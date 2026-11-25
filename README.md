# CareCompanion

A supervised learning demo for next-day recovery forecasting, plus a small coaching dashboard around it.

- **Task:** given one day's biometrics, forecast the recovery state for the next day (Ready, Moderate, or Deload Needed).
- **Model:** RandomForest, selected by nested leave-one-participant-out validation.
- **App:** a FastAPI backend and a static dashboard. An optional LLM (OpenRouter or Groq) writes the coaching text. Without a key, a rule-based reply is used.

## The brief, and where each part lives

> Build a coaching agent that remembers a user's history across multiple sessions and adjusts its advice based on past interactions (not a fresh generic answer each time). Deliverable: demo on Azure AI showing advice changing across 2-3 simulated sessions. Dataset: PMData Fitbit dataset.

| Requirement | Where |
|---|---|
| Remembers history across sessions | `coach.py`: saved facts (goals, constraints, preferences) are extracted from what the user says and stored per profile; every reply also reads what the coach advised in the previous sessions and the numbers it saw then (`prior_sessions` in `app.py`). |
| Adjusts advice based on past interactions | The agent is told to open with what changed since the last session and whether its advice now differs, and to respect every saved goal and constraint by name. |
| Not a fresh generic answer each time | `/demo` puts each reply next to the answer a no-memory assistant gives to the same question on the same day. |
| Demo on Azure AI | Azure OpenAI is the first-choice provider (`AZURE_OPENAI_*` in `.env`); `docs/AZURE.md` covers the model deployment and hosting the app on Azure Container Apps (`Dockerfile`). |
| 2-3 simulated sessions | `/demo` (`demo.py`) runs three sessions in order, starting from a blank memory, for a fictional persona or any of the 16 real PMData participants on their own recorded days. |
| PMData Fitbit dataset | The forecast model is trained and tested on PMData (below); `data/pmdata_daily.csv` is the derived daily table the demo uses. |

Without an LLM key the agent still works: replies and memory extraction fall back to rules, and the demo shows the same structure.

## Results

Trained and evaluated on the real **PMData** dataset (Thambawita et al., 2020): 1,423 next-day forecast rows from 16 participants over five months. Nested leave-one-participant-out; numbers from `models/model_metrics.json`.

| Model | Accuracy | Macro F1 | Brier | Accuracy per participant |
|---|---|---|---|---|
| RandomForest (selected on macro F1) | 0.561 | 0.550 | 0.570 | 0.556 ± 0.151 |
| GradientBoosting | 0.569 | 0.532 | 0.567 | 0.569 ± 0.126 |
| LogisticRegression | 0.509 | 0.494 | 0.583 | 0.516 ± 0.128 |
| Baseline: majority class | 0.323 | 0.197 | | |
| Baseline: persistence (tomorrow = today) | 0.579 | 0.564 | | |

**The forecast does not beat the persistence baseline.** It is 0.014 macro F1 and 0.018 accuracy behind. People's recovery state is sticky from one day to the next, and today's biometrics add little on top of today's own state. A quick check of other set-ups (stronger regularisation, today's state as explicit inputs, histogram gradient boosting) did not close the gap either, and none of them replaced the model, because picking among them on the same folds would overfit the evaluation. Accuracy varies a lot between participants (standard deviation 0.15).

On the synthetic sample this project started with, the gap was larger (0.510 vs 0.542 macro F1).

### Final test on unseen future days

To check the realistic setting, where the app already has some of a user's history, the last 20% of each participant's timeline (292 days) was locked away before any feature exploration (`experiments/explore_features.py` only reads the first 80%). The model was tuned on the first 80% and scored once on the locked part (`experiments/evaluate_holdout.py`, results in `models/holdout_metrics.json`).

| Question | Model | Persistence |
|---|---|---|
| Will tomorrow need a deload? (yes/no) | **71.6% accuracy, F1 0.656, ROC AUC 0.763** | 69.9% accuracy, F1 0.627 |
| Tomorrow's state (3 classes) | 61.3% accuracy, macro F1 0.610 | 62.0% accuracy, macro F1 0.620 |

- The yes/no deload alert beats persistence, and unlike persistence it gives a probability (AUC 0.76), so the alert threshold can be tuned.
- On the 3-class task the model ties persistence: the 95% bootstrap interval for the macro-F1 difference is [-0.072, +0.050]. During exploration the model looked 0.05 ahead in this setting; the locked test did not confirm it.
- **Confident alerts.** Using the threshold chosen on the early 80% (`experiments/explore_more.py`), the alert is confident (probability at least 0.65 either way) on 51% of the locked days, and right on 77.2% of those, against 75.8% for persistence on the same days and 65.7% for the model on the remaining days. In exploration this looked like 85%; the locked test did not confirm it. This was the locked set's second use, stated here for that reason.
- Also tried and dropped in round two: readiness and fatigue from the previous two days and their trend (no gain), and weighting each person's own history three times (about one point of accuracy, within noise).
- Training-load features from `srpe.csv` (acute:chronic workload ratio), personal-baseline deviations, recent injuries and tomorrow's weekday were tried during exploration and did not help, so they are not used.

Selected model, per class:

| Class | Precision | Recall | Support |
|---|---|---|---|
| Deload Needed | 0.67 | 0.56 | 610 |
| Moderate | 0.55 | 0.55 | 580 |
| Ready | 0.42 | 0.60 | 233 |

Permutation importance puts the 3-day readiness average far ahead of everything else, then sleep score, stress and steps. The standard deviations are larger than the means, so only the first place is clear.

**Gaps in the data.** The release has no `resting_heart_rate.json` for p12 and p13, so resting heart rate is missing on 23% of rows; sleep fields on 12%. Missing values are imputed (median) inside the pipeline, and the metrics file lists the absent files and each feature's missing rate.

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

- The shipped model is trained on the real PMData dataset, which is not in the repository. `data/pmdata_sample.csv` is a synthetic 480-row sample with the same schema, used when `data/pmdata_raw/` is absent. The full PMData dataset is at [datasets.simula.no/pmdata](https://datasets.simula.no/pmdata) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); cite Thambawita et al., "PMData: A Sports Logging Dataset", MMSys 2020, [doi:10.1145/3339825.3394926](https://dl.acm.org/doi/10.1145/3339825.3394926). To retrain, put participant folders under `data/pmdata_raw/` (only `pmsys/wellness.csv` and the Fitbit files `sleep_score.csv`, `resting_heart_rate.json`, `steps.json`, `sleep.json` and `time_in_heart_rate_zones.json` are read, about 85 MB) and run `train_model.py`. A missing wellness file raises; a missing Fitbit file leaves its columns blank for imputation and is recorded.
- The three demo personas (`demo_data.py`) are fictional and sit at higher readiness than most of the sample, so their forecasts are illustrations.
- Forecasts and coaching are a demo, not medical advice. Probabilities are not calibrated; the Brier score is reported so you can judge them.

## Files

- `train_model.py`: data loading, nested validation, baselines, permutation importance, and the saved model.
- `app.py`: FastAPI routes, pages, validation and rate limits.
- `coach.py`: the agent: providers (Azure OpenAI first), prompts, memory extraction, offline replies.
- `demo.py`: the three-session simulation behind `/demo`.
- `features.py`: feature definitions shared by training and serving.
- `sandbox.py`, `ratelimit.py`: per-visitor storage and rate limiting.
- `demo_data.py`: the three demo personas.
- `static/`: the dashboard (`index.html`, `app.js`, `tracker.js`, built `app.css`).
- `test_app.py`: API, sandbox, security, feature-parity and coach tests.
- `models/`: the saved model and its metrics.
