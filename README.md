# CareCompanion — Supervised ML + Agentic Health Coach (PMData + Groq)

A longitudinal health coaching application that combines a **supervised Machine Learning model** trained on the **PMData Fitbit + PMSys dataset** with a **Groq-powered Agentic Health Coach** (`llama-3.3-70b-versatile`) that remembers user history across multiple sessions.

## Project Structure
- `train_model.py` — Supervised ML pipeline (ingests raw Kaggle PMData folders or `data/pmdata_sample.csv`, engineers 3-day rolling biometric features, compares Random Forest, Gradient Boosting, and Logistic Regression, and saves the best model to `models/readiness_model.joblib`).
- `app.py` — FastAPI backend serving ML predictions (`/api/predict`), Groq-powered memory-aware coaching (`/api/coach`), and persistent session memory (`/api/memory`).
- `static/index.html` — Modern, responsive dashboard with ML Readiness Predictor, Feature Importance view, PMData charts, and multi-session coaching chat.
- `data/pmdata_sample.csv` — 480-row PMData-schema dataset (16 participants × 30 days) for immediate training & testing.

## Quickstart
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **(Optional) Use the full Kaggle PMData dataset**
   Download PMData from Kaggle / Simula and place participant folders (`p01/`, `p02/`, ...) inside `data/pmdata_raw/`. If `data/pmdata_raw/` is empty, `train_model.py` automatically uses `data/pmdata_sample.csv`.
3. **Train the Supervised ML Model**
   ```bash
   python train_model.py
   ```
   This trains and evaluates 3 classifiers, prints the 5-fold CV accuracy & F1 score, and saves `models/readiness_model.joblib` + `models/model_metrics.json`.
4. **Configure your Groq API Key**
   Copy `.env.example` to `.env` and paste your Groq API key:
   ```bash
   cp .env.example .env
   ```
   (You can also paste your Groq API key directly in the web UI settings modal.)
5. **Run the Application**
   ```bash
   uvicorn app:app --reload --port 8000
   ```
   Open `http://127.0.0.1:8000` in your browser.
