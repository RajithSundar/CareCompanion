"""Tests for the CareCompanion FastAPI backend, the next-day forecast, and agentic memory."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app
import train_model

CLASSES = {"Deload Needed", "Moderate", "Ready"}

FULL_BIOMETRICS = {
    "steps": 9650,
    "resting_heart_rate": 58,
    "sleep_score": 82,
    "deep_sleep_min": 88,
    "rem_sleep_min": 95,
    "cardio_min": 26,
    "peak_min": 8,
    "rhr_delta": 0.0,
    "sleep_3d_avg": 81.0,
    "steps_3d_avg": 9400.0,
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "MEMORY_FILE", tmp_path / "agent_memory.json")
    monkeypatch.setattr(app, "STATE_FILE", tmp_path / "app_state.json")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return TestClient(app.app)


def test_model_info_reports_forecast_and_baselines(client):
    info = client.get("/api/model-info").json()
    assert info["trained"] is True
    assert info["task"] == "Next-day recovery state forecast"
    assert info["selected_model"] in {"RandomForest", "GradientBoosting", "LogisticRegression"}
    assert {"majority_class", "persistence_today_state"} <= set(info["baselines"])
    assert "sedentary_min" not in info["feature_importances"]
    assert "very_active_min" not in info["feature_importances"]


def test_predict_returns_forecast_for_next_day(client):
    res = client.post("/api/predict", json=FULL_BIOMETRICS)
    assert res.status_code == 200
    body = res.json()
    assert body["predicted_recovery_state"] in CLASSES
    assert body["forecast_for"] == "next_day"
    assert set(body["probabilities"]) == CLASSES
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3


def test_predict_rejects_missing_fields(client):
    partial = {k: v for k, v in FULL_BIOMETRICS.items() if k != "deep_sleep_min"}
    assert client.post("/api/predict", json=partial).status_code == 422


def test_server_features_match_training_definitions(client):
    """The app's rhr_delta and rolling means must equal train_model.engineer_features on the same days."""
    raw = pd.read_csv(train_model.SAMPLE_CSV)
    raw["recovery_state"] = train_model.label_recovery_state(raw)
    feats = train_model.engineer_features(raw)
    p01 = feats[feats["participant_id"] == "p01"].reset_index(drop=True)

    days = []
    for _, r in p01.iterrows():
        days.append({
            "day": r["date"], "steps": r["steps"], "rhr": r["resting_heart_rate"],
            "sleepScore": r["sleep_score"], "deepMin": r["deep_sleep_min"], "remMin": r["rem_sleep_min"],
            "cardio": r["cardio_min"], "peak": r["peak_min"], "readiness": r["readiness"],
            "fatigue": r["fatigue"], "soreness": r["soreness"], "stress": r["stress"],
        })
    for idx in (0, 1, 6, 15):
        bio = app.build_biometrics_from_days(days, idx)
        row = p01.iloc[idx]
        assert bio.rhr_delta == pytest.approx(row["rhr_delta"], abs=0.01)
        assert bio.sleep_3d_avg == pytest.approx(row["sleep_3d_avg"], abs=0.01)
        assert bio.steps_3d_avg == pytest.approx(row["steps_3d_avg"], abs=0.1)


def test_day_forecast_endpoint_uses_stored_days(client):
    res = client.get("/api/predict/p01", params={"day_index": 4})
    assert res.status_code == 200
    body = res.json()
    assert body["day_index"] == 4
    assert body["forecast_for"] == "next_day"
    assert body["predicted_recovery_state"] in CLASSES
    assert set(body["biometrics"]) == set(app.BiometricInput.model_fields)


def test_checkin_rejects_missing_sleep_and_activity_fields(client):
    res = client.post("/api/checkin/p01", json={
        "dayIndex": 0, "steps": 4100, "rhr": 65, "sleepScore": 58,
        "readiness": 3, "fatigue": 5, "soreness": 4, "stress": 4,
    })
    assert res.status_code == 422


def test_checkin_stores_entered_values_not_defaults(client):
    res = client.post("/api/checkin/p01", json={
        "steps": 4100, "rhr": 65, "sleepScore": 58, "deepMin": 40, "remMin": 50,
        "cardio": 3, "peak": 0, "readiness": 3, "fatigue": 5, "soreness": 4, "stress": 4,
    })
    assert res.status_code == 200
    record = res.json()["day_record"]
    assert record["deepMin"] == 40
    assert record["remMin"] == 50
    assert record["cardio"] == 3
    assert record["peak"] == 0
    assert "fatBurn" not in record


def test_landing_links_point_to_existing_sections(client):
    html = client.get("/").text
    for target in ("forecast", "coach", "checkin", "memory", "telemetry"):
        assert f'href="#{target}"' in html, target
        assert f'id="{target}"' in html, target
    assert 'id="themeToggleBtn"' in html


def test_no_cors_middleware_is_installed():
    names = [m.cls.__name__ for m in app.app.user_middleware]
    assert "CORSMiddleware" not in names


def test_memory_crud_and_reset(client):
    res = client.get("/api/memory/P01")
    assert res.status_code == 200
    assert len(res.json()["memories"]) == 3

    add_res = client.post(
        "/api/memory/P01",
        json={"category": "Constraint", "text": "Avoid steep hill repeats", "session": "S2"},
    )
    assert add_res.status_code == 200
    added_id = add_res.json()["added"]["id"]
    assert len(add_res.json()["memories"]) == 4

    del_res = client.delete(f"/api/memory/P01/{added_id}")
    assert del_res.status_code == 200
    assert len(del_res.json()["memories"]) == 3

    reset_res = client.post("/api/memory/P01/reset")
    assert reset_res.status_code == 200
    assert len(reset_res.json()["memories"]) == 3


def test_offline_coach_uses_forecast_wording_and_memory(client):
    fatigued = dict(FULL_BIOMETRICS, steps=4500, resting_heart_rate=64, sleep_score=59,
                    deep_sleep_min=48, rem_sleep_min=65, cardio_min=6, peak_min=0,
                    rhr_delta=5.0, sleep_3d_avg=63.0, steps_3d_avg=5100.0)
    res = client.post("/api/coach", json={
        "participant_id": "p01",
        "session_label": "Session 2",
        "message": "Should I still do my tempo run?",
        "biometrics": fatigued,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "offline_fallback"
    assert "tomorrow" in body["reply"].lower()
    assert "left knee" in body["reply"].lower()


def test_groq_live_with_autonomous_memory_extraction(client):
    mock_reply = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Protect your **right ankle** today and stick to **20 min flat mobility**."))]
    )
    mock_extract = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({"memories": [{"category": "Constraint", "text": "Mild right ankle soreness after trail run"}]})
        ))]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [mock_reply, mock_extract]

    with patch("app.Groq", return_value=mock_client):
        res = client.post("/api/coach", json={
            "participant_id": "p02",
            "session_label": "Session 2",
            "message": "My right ankle feels sore after yesterday's trail run.",
            "biometrics": dict(FULL_BIOMETRICS, rhr_delta=2.0, sleep_score=65),
            "groq_api_key": "gsk_test_key",
        })

    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "groq_live"
    assert len(body["newly_extracted_memories"]) == 1
    assert body["newly_extracted_memories"][0]["category"] == "Constraint"

    mem_res = client.get("/api/memory/p02")
    assert any("right ankle" in m["text"].lower() for m in mem_res.json()["memories"])


def test_ui_palette_light_and_dark_tokens(client):
    html = client.get("/").text
    # Light theme is the default: red accent on white. Dark theme keeps a red accent for contrast.
    assert "--color-accent: #C62828;" in html
    assert '--color-paper: #FFFFFF;' in html
    assert ':root[data-theme="dark"]' in html
    assert "--color-accent: #FF6B6B;" in html
    assert "teal-" not in html
    assert "indigo-" not in html

    for pid in ("P01", "P07", "P12"):
        assert len(client.get(f"/api/memory/{pid}").json()["memories"]) >= 3

    assert "theme-color" in html
    assert "Simulated Sessions" not in html
    assert "Compare Generic AI" not in html
    assert "Offline Demo" not in html
    assert "sk-or-v1-" not in html
    assert ".env" in (app.BASE_DIR / ".gitignore").read_text(encoding="utf-8")


def test_full_system_state_checkin_and_sessions(client):
    state_data = client.get("/api/state/p01").json()
    assert len(state_data["days"]) == 10
    assert len(state_data["sessions"]) == 4

    checkin_res = client.post("/api/checkin/p01", json={
        "dayIndex": 0, "steps": 4100, "rhr": 65, "sleepScore": 58,
        "deepMin": 44, "remMin": 58, "cardio": 2, "peak": 0,
        "readiness": 3, "fatigue": 5, "soreness": 4, "stress": 4,
    })
    assert checkin_res.status_code == 200
    assert checkin_res.json()["day_record"]["steps"] == 4100
    assert checkin_res.json()["ml_prediction"]["forecast_for"] == "next_day"

    sess_res = client.post("/api/sessions/p01", json={})
    assert sess_res.status_code == 200
    assert len(sess_res.json()["sessions"]) == 5
    assert sess_res.json()["session"]["id"] == "S5"

    reset_res = client.post("/api/state/p01/reset")
    assert reset_res.status_code == 200
    assert len(reset_res.json()["state"]["sessions"]) == 4
