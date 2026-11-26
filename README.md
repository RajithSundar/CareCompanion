# CareCompanion: an agentic health coach

Log how you slept and how you feel each day. CareCompanion predicts whether you're ready to train tomorrow, and an AI coach that remembers your goals, injuries and past conversations plans around it.

- **Task:** given one day's biometrics and how the person feels, forecast the recovery state for the next day: Ready, Moderate or Deload Needed (shown in the app as "ready to train", "take it easy" and "rest day").
- **Model:** ExtraTrees, picked from eleven supervised algorithms by nested leave-one-participant-out validation. On each person's locked later days it reaches 63.4% accuracy and macro F1 0.627 (3 classes), against 62.0% and 0.594 for the tomorrow-equals-today baseline.
- **Agent:** a tool-using coach (`agent.py`) that looks up saved facts, earlier sessions and trends, saves what you tell it and rewrites your plan. Azure OpenAI first, then OpenRouter or Groq; without a key, a rule-based coach.
- **App:** a FastAPI backend with sign-up, sign-in (three demo accounts ready to use) and a multi-page dashboard.

## The brief, and where each part lives

Problem 27, "CareCompanion: Agentic Health Coach". *Users need ongoing, personalized health coaching, not one-off advice.*

> Build a coaching agent that remembers a user's history across multiple sessions and adjusts its advice based on past interactions (not a fresh generic answer each time). Deliverable: demo on Azure AI showing advice changing across 2-3 simulated sessions. Dataset: PMData Fitbit dataset.

| Requirement | Where |
|---|---|
| Remembers history across sessions | `agent.py`: the agent saves goals, injuries and schedule limits with its `save_fact` tool (a second extraction pass catches any it misses), and reads them back with `get_saved_facts` and `get_past_sessions` (what it advised before, the numbers then, and what has changed since). Stored per account, so it survives sign-out. |
| Adjusts advice based on past interactions | It opens with what changed since the last session and whether its advice now differs, names every saved constraint, and rewrites the session plan (`update_plan`). Measured below: injuries from Session 1 come back unprompted in 97% of later replies, against 0% without memory. |
| Not a fresh generic answer each time | `/demo` puts each reply next to the answer a no-memory assistant gives to the same question on the same day. |
| Demo on Azure AI | Azure OpenAI is the first-choice provider (`AZURE_OPENAI_*` in `.env`); `docs/AZURE.md` covers the model deployment and hosting the app on Azure Container Apps (`Dockerfile`). |
| 2-3 simulated sessions | `/demo` (`demo.py`) runs three sessions in order, starting from a blank memory, for a fictional persona or any of the 16 real PMData participants on their own recorded days. |
| PMData Fitbit dataset | The forecast model is trained and tested on PMData (below); `data/pmdata_daily.csv` is the derived daily table the demo uses. |

Without an LLM key the agent still works: replies and memory extraction fall back to rules, and the demo shows the same structure.

## The agent

Each coach reply is a small tool-calling loop (`agent.py`), shown under the reply as "N steps before answering":

| Tool | What it does |
|---|---|
| `get_saved_facts` | Every goal, constraint, preference and rule on file for this account |
| `get_past_sessions` | The last three sessions: readings then, forecast then, advice given, and the changes since (with "better" or "worse" already judged) |
| `get_trends` | Average, latest and direction of steps, resting HR, sleep, readiness and fatigue over N days, against the earlier baseline |
| `save_fact` | Remember a new goal, injury or schedule limit (validated, deduplicated, at most 3 per message) |
| `update_plan` | Replace the session's action plan with 2 to 5 items |

The first model call must use a tool, the loop stops after 4 rounds, and tools only reach the signed-in account's data. Tool arguments from the model are validated like user input.

**Safety screen** (`safety.py`). Before the model sees a message, plain rules check for red flags: chest pain, fainting, trouble breathing, a racing heart at rest, stroke signs, coughing up blood, and thoughts of self-harm. Negations ("no chest pain") are ignored. A flagged message gets a fixed reply (stop training, see a doctor, emergency numbers; for self-harm, crisis lines such as Tele-MANAS 14416), no model call, and a saved constraint so later sessions ask about medical clearance.

### Does memory change the advice?

`experiments/evaluate_agent.py` runs the 3-session demo for all 19 profiles (3 personas, 16 real PMData participants) and asks the same questions of a stateless assistant (same model, same day, no memory). It checks, with keywords, whether a reply brings back what the person said in Session 1 when the new message does not mention it. The deployed coach runs on **Azure OpenAI, gpt-4.1-mini** (`models/agent_eval.json`); the earlier final run on GPT-4o-mini via OpenRouter is kept for comparison (`models/agent_eval_openrouter_4o-mini.json`). The stateless assistant scored 0 to 3% on every recall row with both models.

| Recalled unprompted in Sessions 2 and 3 | Agent, Azure gpt-4.1-mini | Agent, GPT-4o-mini | Without memory |
|---|---|---|---|
| Injury or constraint from Session 1 (37 sessions) | **100%** | 97% | 0% |
| Goal (37) | **84%** | 92% | 0 to 3% |
| Schedule limit (36) | **92%** | 11% | 0% |
| Refers back to the previous session (38) | **87%** | 100% | 0% |

| Agent behaviour (57 sessions) | Azure gpt-4.1-mini | GPT-4o-mini |
|---|---|---|
| Read saved facts and earlier sessions before answering | 98% | 100% |
| Facts saved in Session 1: injury / goal / schedule | 100% / 95% / 100% | 100% / 95% / 100% |
| Tool calls per reply | 3.8 | 4.2 |
| Looked at trends / updated the plan | 7% / 70% | 90% / 28% |
| Resting-HR changes described in the wrong direction | 1 of 10 | 0 of 24 |

Safety screen on 24 labelled messages (14 red flags, 10 look-alikes): 100% recall, 100% precision. It is rule-based, so the model does not affect it.

How it got here, honestly: the first run (`models/agent_eval_v1.json`) found the agent recalled goals in 68% and schedules in 3% of sessions and never updated the plan; the prompt now names schedule limits and asks for a plan update when advice changes (`agent_eval_v2.json`: 86% and 19%). That run also showed the model sometimes called a rising resting HR an improvement (about 3 in 10 claims), so `get_past_sessions` now returns the changes with the direction already judged; the GPT-4o-mini final run had none and the gpt-4.1-mini run had one. One stroke-sign phrasing in the safety set was missed at first and the pattern was widened, so the safety score is not from an untouched test set. The two models trade strengths: gpt-4.1-mini uses schedule limits and updates the plan far more often, but rarely calls `get_trends` and refers back less often. Recall numbers also move by several points between runs of the same model. Keyword checks measure recall, not whether the advice is good.

## Results

Trained and evaluated on the real **PMData** dataset (Thambawita et al., 2020): 1,423 next-day forecast rows from 16 participants over five months.

> **Correction (26 Nov 2026).** Earlier versions of this project labelled days with the PMSys wellness scales the wrong way round. In PMSys, fatigue, soreness and stress run from 1 (worst) to 5 (best): fatigue 5 means very fresh. In PMData these scores rise with sleep quality, mood and readiness, which confirms it. The old rule treated fatigue 4 or 5 as "Deload Needed", so 21% of days were mislabelled (77 well-rested days were labelled as needing rest). The rule, the demo personas and every result below have been redone with the correct direction. The old results are kept in `models/before-label-fix/` and should not be compared with these, because the labels differ.

### Eleven supervised algorithms, new-person setting

Nested leave-one-participant-out: each participant is held out in turn, and every algorithm is tuned by grouped 3-fold CV on the other 15 only. Numbers from `models/model_metrics.json`, sorted by macro F1.

| Model | Accuracy | Macro F1 | Brier | Accuracy per participant |
|---|---|---|---|---|
| **ExtraTrees (selected)** | **0.573** | **0.557** | 0.558 | 0.579 ± 0.148 |
| RandomForest | 0.575 | 0.555 | 0.550 | 0.577 ± 0.153 |
| Soft voting (LR + RF + ET) | 0.577 | 0.550 | 0.538 | 0.584 ± 0.149 |
| HistGradientBoosting | 0.547 | 0.528 | 0.557 | 0.556 ± 0.167 |
| SVM (RBF kernel) | 0.543 | 0.521 | 0.554 | 0.544 ± 0.155 |
| Gaussian naive Bayes | 0.535 | 0.520 | 0.755 | 0.539 ± 0.163 |
| GradientBoosting | 0.581 | 0.516 | 0.532 | 0.595 ± 0.152 |
| LogisticRegression | 0.500 | 0.494 | 0.603 | 0.512 ± 0.143 |
| DecisionTree | 0.502 | 0.479 | 0.610 | 0.508 ± 0.158 |
| KNN | 0.528 | 0.475 | 0.583 | 0.529 ± 0.151 |
| MLP (one hidden layer) | 0.490 | 0.416 | 0.609 | 0.495 ± 0.163 |
| Baseline: persistence (tomorrow = today) | 0.574 | 0.543 | | |
| Baseline: majority class | 0.508 | 0.225 | | |

For someone the model has never seen, the selected model is ahead of persistence on macro F1 (0.557 vs 0.543) and level on accuracy (0.573 vs 0.574). The bagged tree models lead; KNN and the MLP do worst on 1,400 noisy rows. Choosing among eleven models on the same folds flatters the winner a little, which is why the locked test below matters more. Accuracy varies a lot between participants (standard deviation 0.15).

### How the features were chosen

`experiments/explore_techniques.py` compared techniques on the **early 80%** of each participant's days only (`models/technique_sweep.json`, rerun on the corrected labels). Persistence scores 0.529 macro F1 on those rows.

| Technique (random forest unless stated) | Macro F1 | Kept? |
|---|---|---|
| Original features | 0.522 | baseline |
| **+ lag features**: yesterday's readiness, fatigue, soreness, stress and sleep; change since yesterday; today's state and how many days it has lasted; 7-day readiness and fatigue mean and spread | **0.545** | yes |
| + personal normalisation (each value against the person's own running mean and spread) | 0.542 | no: no gain over lags alone |
| Soft voting, stacking, MLP, KNN on the same features | 0.45 to 0.54 | voting kept as a candidate only |
| Regress tomorrow's readiness and fatigue, then apply the label rule | 0.438 | no |

"Yesterday" means the previous calendar day: after a gap in the record the lag features are missing and imputed, in training and in the app alike.

### Final test on unseen future days

To check the realistic setting, where the app already has some of a user's history, the last 20% of each participant's timeline (292 days) was locked away before any feature exploration. All eleven candidates were tuned on the first 80% (1,131 rows), the best by inner CV was picked (ExtraTrees), and it was scored once on the locked part (`experiments/evaluate_holdout.py`, `models/holdout_metrics.json`).

| Question | Model | Persistence |
|---|---|---|
| Tomorrow's state (3 classes) | **63.4% accuracy, macro F1 0.627** | 62.0%, macro F1 0.594 |
| Will tomorrow need a rest day? (yes/no, 48% of days) | **69.9% accuracy, F1 0.677, ROC AUC 0.757** | 68.2%, F1 0.669 |
| Days the model is confident (probability at least 0.65 either way, 45% of days) | 80.0% right | 80.0% on the same days |

- The 3-class forecast is ahead of persistence by 0.033 macro F1. The 95% bootstrap interval for the difference is [-0.024, +0.086], so it is likely but not certain that the model is better.
- The yes/no rest-day alert is 1.7 points ahead and, unlike persistence, gives a probability (AUC 0.76), so its threshold can be tuned. On the days it is confident it is right 80% of the time, but so is persistence: those are the easy days.
- **Use of the locked set.** It was scored three times on the old labels (the original model, the confident-alert threshold, the lag features) and now once on the corrected labels. Each time, the choices were fixed on the early 80% before the locked part was read, but repeated use still makes these numbers slightly optimistic.
- Tried and dropped earlier: training load from `srpe.csv` (acute:chronic workload ratio), recent injuries, tomorrow's weekday, and weighting each person's own history three times.

Selected model (new-person setting), per class:

| Class | Precision | Recall | Support |
|---|---|---|---|
| Deload Needed | 0.70 | 0.58 | 723 |
| Moderate | 0.51 | 0.54 | 449 |
| Ready | 0.44 | 0.62 | 251 |

Permutation importance puts today's state first, then today's readiness and its 3-day and 7-day averages, then the fatigue averages. In other words, how people say they feel carries most of the signal; the wearable numbers add a little. The standard deviations are larger than the means, so only the first place is clear.

**Gaps in the data.** The release has no `resting_heart_rate.json` for p12 and p13, so resting heart rate is missing on 23% of rows; sleep fields on 12%. Missing values are imputed (median) inside the pipeline, and the metrics file lists the absent files and each feature's missing rate.

## Method

- **Target.** The recovery state on day t+1, paired with day t's features. Pairs that are not on consecutive calendar days are dropped.
- **Labels are rule-based** (`features.state_rule`). PMSys scales run 1 (worst) to 5 (best), readiness 0 to 10. Ready if readiness >= 7 and fatigue >= 3 (not tired); Deload Needed if readiness <= 4 or fatigue <= 2 (tired); otherwise Moderate. Not clinical ground truth.
- **Features** (`features.py`, shared by training and serving), 28 in all: steps, resting heart rate, sleep score, deep and REM minutes, cardio and peak minutes, today's readiness, fatigue, soreness and stress, `rhr_delta` (today minus the 7-day mean), 3-day means of sleep, steps, readiness and fatigue, and the lag features above (yesterday's self-reports and sleep, the change since yesterday, today's state and its streak, 7-day readiness and fatigue mean and readiness spread). Windows are calendar-based, so a gap in the record does not stretch them. Today's readiness and fatigue define today's label, not tomorrow's, so they are fair inputs; the persistence baseline uses them too.
- **Algorithms.** Logistic regression, KNN, SVM (RBF), Gaussian naive Bayes, decision tree, random forest, extra trees, gradient boosting, histogram gradient boosting, an MLP and a soft-voting ensemble, with class weighting where the algorithm supports it (Ready is the smallest class, 16%).
- **Validation.** The outer loop holds out one participant; the inner loop tunes hyperparameters with grouped 3-fold CV on the rest. Imputation and scaling are inside the pipeline, so they are fit on training folds only. The selection metric is macro F1, so the small Ready class counts as much as the others.
- **Serving.** The backend builds features from stored day records by calling the same `features.py` code. A test checks they match.

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python train_model.py            # optional: retrains and rewrites models/
uvicorn app:app --port 8000
```

Open `http://127.0.0.1:8000` and sign in with a demo account:

| Username | Profile | Password |
|---|---|---|
| `alex` | Alex Rivera, half-marathon runner | `demo1234` |
| `maya` | Maya Lin, desk-bound tech lead | `demo1234` |
| `jordan` | Jordan Patel, strength athlete | `demo1234` |

Or create your own account at `/signup`. A new account starts either **fresh** (no days; log your own check-ins or import a CSV, and the forecast and coach start after the first day) or with **sample data** (the last 14 complete days of one real PMData participant, labelled as sample data in the app).

The landing, sign-in and sign-up pages are public (the watch shows sample stats until you sign in); every other page redirects to `/login`.

To run the tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## LLM configuration

Copy `.env.example` to `.env` and set one provider. The server uses Azure OpenAI if `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_DEPLOYMENT` are all set (see `docs/AZURE.md`), otherwise OpenRouter if `OPENROUTER_API_KEY` is set, otherwise Groq if `GROQ_API_KEY` is set. Keys stay on the server; the browser never sees or sends one. Without a key the coach gives a rule-based reply built from the forecast and the saved memories.

Optional limits (defaults in brackets): `CC_COACH_PER_10MIN` (15) and `CC_COACH_PER_DAY` (60) LLM replies per IP, `CC_LLM_GLOBAL_PER_DAY` (500) in total, `CC_WRITES_PER_MIN` (60) writes per IP.

## Accounts and sessions

- Sign-in (`auth.py`) issues an HS256 JWT (PyJWT) with `iss`, `aud`, `iat`, `nbf`, `exp` (8 hours) and a random `jti`; decoding pins the algorithm and requires every claim, so `alg: none`, wrong-key, expired and tampered tokens are rejected.
- The browser keeps the token in an HttpOnly, Secure, SameSite=Strict cookie that page scripts cannot read. API clients can send `Authorization: Bearer <token>` instead (`POST /api/auth/login` returns it).
- Sign-out revokes the token's `jti` and clears the cookie. Sign-in attempts are rate-limited per IP (`CC_LOGINS_PER_5MIN`, default 10).
- Passwords are scrypt-hashed and compared in constant time, and unknown usernames take as long as known ones. Set `CC_DEMO_PASSWORD` to change the shared demo password and `CC_SHOW_DEMO_PASSWORD=0` to stop the sign-in page showing it.
- The signing key is `CC_JWT_SECRET` (32+ characters) or, if unset, a random key generated once into `data/.jwt_secret` (mode 600).
- Sign-up (`POST /api/auth/signup`) takes a name, a username (3 to 20 of `a-z 0-9 _`), a password (8+ characters) and how to start. Registered accounts are kept in `data/users.json` (mode 600, scrypt salt and hash only), sign-ups are rate-limited per IP (`CC_SIGNUPS_PER_HOUR`, default 5) and capped in total (`CC_MAX_USERS`, default 500). PMData participants p12 and p13 have no resting heart rate in the release, so they are not offered as sample data; other gaps in a sample day are filled with that participant's median (`onboarding.py`).
- Each account can only read and change its own profile (403 otherwise). Its records live in one JSON file under `data/accounts/` and persist across sign-ins and devices, which is what lets the coach remember earlier sessions.

## Hosting

- Writes are atomic and locked per account. Request bodies are capped at 256 KB, writes must be JSON, and every field is range-checked.
- The page sends a strict Content-Security-Policy (no inline or third-party scripts). Chart.js is vendored in `static/vendor/`; the 17 Lucide icons in use are inlined in `static/icons.js` (3 KB instead of 358 KB); Tailwind is built ahead of time (`frontend/README.md`).
- `/docs` and `/openapi.json` are disabled.
- On this machine the app and its Tailscale Funnel run as systemd user services: `systemctl --user status carecompanion carecompanion-tailscaled`. Logs: `journalctl --user -u carecompanion -f`.

## Data and limitations

- The shipped model is trained on the real PMData dataset, which is not in the repository. `data/pmdata_sample.csv` is a synthetic 480-row sample with the same schema, used when `data/pmdata_raw/` is absent. The full PMData dataset is at [datasets.simula.no/pmdata](https://datasets.simula.no/pmdata) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); cite Thambawita et al., "PMData: A Sports Logging Dataset", MMSys 2020, [doi:10.1145/3339825.3394926](https://dl.acm.org/doi/10.1145/3339825.3394926). To retrain, put participant folders under `data/pmdata_raw/` (only `pmsys/wellness.csv` and the Fitbit files `sleep_score.csv`, `resting_heart_rate.json`, `steps.json`, `sleep.json` and `time_in_heart_rate_zones.json` are read, about 85 MB) and run `train_model.py`. A missing wellness file raises; a missing Fitbit file leaves its columns blank for imputation and is recorded.
- The three demo personas (`demo_data.py`) are fictional and sit at higher readiness than most of the sample, so their forecasts are illustrations.
- Forecasts and coaching are a demo, not medical advice. Probabilities are not calibrated; the Brier score is reported so you can judge them.

## Files

- `train_model.py`: data loading, nested validation, baselines, permutation importance, and the saved model.
- `app.py`: FastAPI routes, pages, validation and rate limits.
- `agent.py`: the tool-using agent loop and its five tools.
- `safety.py`: the red-flag screen that runs before the agent.
- `coach.py`: providers (Azure OpenAI first), prompts, memory extraction, offline replies.
- `experiments/evaluate_agent.py`: the memory and safety evaluation above.
- `demo.py`: the three-session simulation behind `/demo`.
- `features.py`: feature definitions shared by training and serving.
- `auth.py`: demo and registered accounts, sign-up, password hashing and JWT issue, verify and revoke.
- `onboarding.py`: what a new account starts with (fresh, or sample PMData days).
- `sandbox.py`, `ratelimit.py`: per-account storage and rate limiting.
- `demo_data.py`: the three demo personas.
- `static/`: the dashboard (`index.html`, `app.js`, `tracker.js`, built `app.css`).
- `test_app.py`: API, sandbox, security, feature-parity and coach tests.
- `models/`: the saved model and its metrics.
