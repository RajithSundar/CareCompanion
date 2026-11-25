# CareCompanion

A supervised learning demo for next-day recovery forecasting, plus a small coaching dashboard around it.

- **Task:** given one day's biometrics, forecast the recovery state for the next day (Ready, Moderate, or Deload Needed).
- **Model:** RandomForest, selected by nested leave-one-participant-out validation.
- **App:** a FastAPI backend and a static dashboard. An optional LLM (OpenRouter or Groq) writes the coaching text. Without a key, a rule-based reply is used.

## Results

Nested leave-one-participant-out on `data/pmdata_sample.csv` (464 forecast rows, 16 participants, 29 rows each). The numbers come from `models/model_metrics.json`.

| Model | Accuracy | Macro F1 | Brier |
|---|---|---|---|
| RandomForest (selected) | 0.593 | 0.511 | 0.513 |
| LogisticRegression | 0.573 | 0.490 | 0.542 |
| GradientBoosting | 0.636 | 0.468 | 0.491 |
| Baseline: majority class | 0.588 | 0.247 | |
| Baseline: persistence (tomorrow = today) | 0.629 | 0.542 | |

The forecasting model does not beat the persistence baseline on macro F1 or accuracy. That is the main finding of this project, and it is what the write-up should say.

Selected model, per class (leave-one-participant-out):

| Class | Precision | Recall | Support |
|---|---|---|---|
| Deload Needed | 0.79 | 0.69 | 273 |
| Moderate | 0.48 | 0.41 | 155 |
| Ready | 0.24 | 0.64 | 36 |

Ready is over-predicted, which is why its precision is low.

Permutation importance (drop in macro F1, averaged across held-out participants): `rhr_delta` is the largest, then `cardio_min`, `deep_sleep_min`, `sleep_score`, and `peak_min`. The standard deviations are as large as the means for most features, so treat the ranking as tentative.

## Method

- **Target.** The label is the recovery state on day t+1. Each day's features are paired with the next day's label. Pairs that aren't on consecutive calendar days are dropped.
- **Labels are rule-based.** `recovery_state` comes from fixed thresholds on readiness and fatigue. The labels are not clinical ground truth.
- **Features.** Ten features the app stores: steps, resting heart rate, sleep score, deep and REM sleep minutes, cardio and peak minutes, `rhr_delta` (today's resting heart rate minus the 7-day mean), and 3-day rolling sleep and steps averages. Rolling features use only the current and earlier days of the same participant.
- **Excluded.** Sedentary and very-active minutes are not stored by the app, so the model doesn't use them. Readiness and fatigue are excluded because they define the labels.
- **Validation.** The outer loop holds out one participant at a time. The inner loop tunes hyperparameters with grouped 3-fold CV on the remaining participants only. Imputation is inside the pipeline.
- **Serving.** The backend computes features from stored day records with the same definitions as training. The browser never builds model inputs. A test checks that the two agree.

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

Copy `.env.example` to `.env`. The server uses OpenRouter if `OPENROUTER_API_KEY` is set, otherwise Groq if `GROQ_API_KEY` is set. You can also paste a key in the dashboard settings. Without any key, the coach returns a rule-based reply that quotes the forecast and the stored memories.

## Data and limitations

- `data/pmdata_sample.csv` is a 480-row sample with the PMData schema (16 participants, 30 days each). The full PMData dataset is at [datasets.simula.no/pmdata](https://datasets.simula.no/pmdata). Its license is non-commercial, and the paper should be cited. To use the full data, place participant folders under `data/pmdata_raw/`. The loader raises an error if a required column is missing. It does not fill in defaults.
- The three demo personas in `app.py` are fictional. Their day records are illustrative and not tied to the sample CSV.
- Forecasts are a demo. They are not medical advice.
- The app has no authentication. Anyone who can reach it can change check-ins, memories, and imported data.
- `data/app_state.json` and `data/agent_memory.json` are written at runtime and are git-ignored.
- Probabilities are not calibrated. The Brier score is reported so you can judge them.
- Deep sleep, REM sleep, cardio, and peak minutes are not entered in the check-in form. A check-in keeps the stored values for those fields.

## Files

- `train_model.py`: data loading, nested validation, baselines, permutation importance, and the saved model.
- `app.py`: FastAPI routes, feature construction, and the coach.
- `static/index.html`: the dashboard.
- `test_app.py`: API, feature-parity, and memory tests.
- `models/`: the saved model and its metrics.
