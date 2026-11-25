"""Tests for CareCompanion FastAPI backend, ML prediction, and multi-session agentic memory."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    mem_file = tmp_path / "agent_memory.json"
    state_file = tmp_path / "app_state.json"
    monkeypatch.setattr(app, "MEMORY_FILE", mem_file)
    monkeypatch.setattr(app, "STATE_FILE", state_file)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return TestClient(app.app)


def test_model_info_and_predict(client):
    info_res = client.get("/api/model-info")
    assert info_res.status_code == 200
    info = info_res.json()
    assert info["trained"] is True
    assert info["selected_model"] in {"RandomForest", "GradientBoosting", "LogisticRegression"}
    assert "sleep_score" in info["feature_importances"]

    # High recovery profile
    ready_res = client.post(
        "/api/predict",
        json={
            "steps": 9500,
            "resting_heart_rate": 57,
            "sleep_score": 85,
            "deep_sleep_min": 92,
            "rem_sleep_min": 100,
            "sedentary_min": 600,
            "very_active_min": 28,
            "cardio_min": 24,
            "peak_min": 6,
            "rhr_delta": -1.0,
            "sleep_3d_avg": 83.0,
            "steps_3d_avg": 9200.0,
        },
    )
    assert ready_res.status_code == 200
    assert ready_res.json()["predicted_recovery_state"] in {"Ready", "Moderate", "Deload Needed"}

    # Fatigued profile
    deload_res = client.post(
        "/api/predict",
        json={
            "steps": 4200,
            "resting_heart_rate": 64,
            "sleep_score": 58,
            "deep_sleep_min": 48,
            "rem_sleep_min": 65,
            "sedentary_min": 720,
            "very_active_min": 8,
            "cardio_min": 6,
            "peak_min": 0,
            "rhr_delta": 5.0,
            "sleep_3d_avg": 61.0,
            "steps_3d_avg": 5100.0,
        },
    )
    assert deload_res.status_code == 200
    assert deload_res.json()["predicted_recovery_state"] == "Deload Needed"


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


def test_multi_session_offline_adaptation(client):
    # Session 1: Baseline Ready/Moderate
    s1 = client.post(
        "/api/coach",
        json={
            "participant_id": "p01",
            "session_label": "Session 1",
            "message": "What should my workout intensity be today?",
            "biometrics": {"steps": 9650, "resting_heart_rate": 58, "sleep_score": 82, "rhr_delta": 0.0},
        },
    )
    assert s1.status_code == 200
    assert s1.json()["mode"] == "offline_fallback"
    assert "half-marathon" in s1.json()["reply"].lower()

    # Session 2: Fatigue & RHR spike -> advice adapts to Deload Needed + knee constraint
    s2 = client.post(
        "/api/coach",
        json={
            "participant_id": "p01",
            "session_label": "Session 2",
            "message": "Should I still do my tempo run?",
            "biometrics": {
                "steps": 4500,
                "resting_heart_rate": 64,
                "sleep_score": 59,
                "rhr_delta": 5.0,
                "sleep_3d_avg": 63.0,
            },
        },
    )
    assert s2.status_code == 200
    assert s2.json()["ml_prediction"]["predicted_recovery_state"] == "Deload Needed"
    assert "mobility" in s2.json()["reply"].lower()
    assert "left knee" in s2.json()["reply"].lower()


def test_groq_live_with_autonomous_memory_extraction(client):
    mock_reply = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Great catch—protect your **right ankle** today and stick to **20 min flat mobility**."))]
    )
    mock_extract = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({"memories": [{"category": "Constraint", "text": "Mild right ankle soreness after trail run"}]})
                )
            )
        ]
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [mock_reply, mock_extract]

    with patch("app.Groq", return_value=mock_client):
        res = client.post(
            "/api/coach",
            json={
                "participant_id": "p02",
                "session_label": "Session 2",
                "message": "My right ankle feels sore after yesterday's trail run.",
                "biometrics": {"steps": 6200, "resting_heart_rate": 66, "sleep_score": 65, "rhr_delta": 2.0},
                "groq_api_key": "gsk_test_key",
            },
        )

    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "groq_live"
    assert len(body["newly_extracted_memories"]) == 1
    assert body["newly_extracted_memories"][0]["category"] == "Constraint"

    # Verify memory persisted for Session 3
    mem_res = client.get("/api/memory/p02")
    assert any("right ankle" in m["text"].lower() for m in mem_res.json()["memories"])


def test_ui_brand_palette_and_personas(client):
    root_res = client.get("/")
    assert root_res.status_code == 200
    html = root_res.text
    html_upper = html.upper()
    assert "#3E92CC" in html_upper
    assert "#F1D246" in html_upper
    assert "#090C08" in html_upper
    assert "teal-" not in html
    assert "indigo-" not in html

    for pid in ("P01", "P07", "P12"):
        mem_res = client.get(f"/api/memory/{pid}")
        assert mem_res.status_code == 200
        assert len(mem_res.json()["memories"]) >= 3

    assert "Simulated Sessions" not in html
    assert "Compare Generic AI" not in html
    assert "Offline Demo" not in html
    assert "sk-or-v1-" not in html
    gitignore_text = (app.BASE_DIR / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore_text


def test_full_system_state_checkin_and_sessions(client):
    state_res = client.get("/api/state/p01")
    assert state_res.status_code == 200
    state_data = state_res.json()
    assert len(state_data["days"]) == 10
    assert len(state_data["sessions"]) == 4

    checkin_res = client.post(
        "/api/checkin/p01",
        json={
            "dayIndex": 0,
            "steps": 4100,
            "rhr": 65,
            "sleepScore": 58,
            "readiness": 3,
            "fatigue": 5,
            "soreness": 4,
            "stress": 4,
        },
    )
    assert checkin_res.status_code == 200
    assert checkin_res.json()["day_record"]["steps"] == 4100
    assert checkin_res.json()["ml_prediction"]["predicted_recovery_state"] == "Deload Needed"

    sess_res = client.post("/api/sessions/p01", json={})
    assert sess_res.status_code == 200
    assert len(sess_res.json()["sessions"]) == 5
    assert sess_res.json()["session"]["id"] == "S5"

    reset_res = client.post("/api/state/p01/reset")
    assert reset_res.status_code == 200
    assert len(reset_res.json()["state"]["sessions"]) == 4


