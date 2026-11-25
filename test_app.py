"""Tests for the CareCompanion backend: forecast, sandboxes, validation, security and the coach."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app
import features
import train_model
from sandbox import SandboxStore

CLASSES = {"Deload Needed", "Moderate", "Ready"}

FULL_BIOMETRICS = {
    "steps": 9650, "resting_heart_rate": 58, "sleep_score": 82, "deep_sleep_min": 88,
    "rem_sleep_min": 95, "cardio_min": 26, "peak_min": 8, "readiness": 8, "fatigue": 2,
    "soreness": 2, "stress": 2, "rhr_delta": 0.0, "sleep_3d_avg": 81.0, "steps_3d_avg": 9400.0,
    "readiness_3d_avg": 7.7, "fatigue_3d_avg": 2.3,
}

DAY = {"steps": 4100, "rhr": 65, "sleepScore": 58, "deepMin": 40, "remMin": 50,
       "cardio": 3, "peak": 0, "readiness": 3, "fatigue": 5, "soreness": 4, "stress": 4}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "store", SandboxStore(tmp_path / "sandboxes"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    app.limiter.reset()
    return TestClient(app.app)


def fresh_client():
    return TestClient(app.app)


# ----------------------------------------------------------------- model + features

def test_model_info_reports_forecast_and_baselines(client):
    info = client.get("/api/model-info").json()
    assert info["trained"] is True
    assert info["task"] == "Next-day recovery state forecast"
    assert {"majority_class", "persistence_today_state"} <= set(info["baselines"])
    assert set(info["feature_importances"]) == set(features.FEATURE_COLS)
    assert all("beats_persistence_macro_f1" in row for row in info["comparison"])
    assert info["llm_configured"] is False


def test_predict_returns_forecast_for_next_day(client):
    res = client.post("/api/predict", json=FULL_BIOMETRICS)
    assert res.status_code == 200
    body = res.json()
    assert body["predicted_recovery_state"] in CLASSES
    assert body["forecast_for"] == "next_day"
    assert set(body["probabilities"]) == CLASSES
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3


def test_predict_rejects_missing_and_out_of_range_fields(client):
    partial = {k: v for k, v in FULL_BIOMETRICS.items() if k != "deep_sleep_min"}
    assert client.post("/api/predict", json=partial).status_code == 422
    assert client.post("/api/predict", json=dict(FULL_BIOMETRICS, steps=-5)).status_code == 422
    assert client.post("/api/predict", json=dict(FULL_BIOMETRICS, readiness=99)).status_code == 422


def test_server_features_match_training_pipeline():
    """Serving and training share features.py; check the app path against the training frame."""
    raw = pd.read_csv(train_model.SAMPLE_CSV)
    feats = features.engineer_features(raw)
    p01 = feats[feats["participant_id"] == "p01"].reset_index(drop=True)
    days = [{
        "day": str(r["date"].date()), "date": str(r["date"].date()), "steps": r["steps"],
        "rhr": r["resting_heart_rate"], "sleepScore": r["sleep_score"], "deepMin": r["deep_sleep_min"],
        "remMin": r["rem_sleep_min"], "cardio": r["cardio_min"], "peak": r["peak_min"],
        "readiness": r["readiness"], "fatigue": r["fatigue"], "soreness": r["soreness"], "stress": r["stress"],
    } for _, r in p01.iterrows()]
    for idx in (0, 1, 6, 15):
        bio = app.build_biometrics_from_days(days, idx)
        for col in features.FEATURE_COLS:
            assert getattr(bio, col) == pytest.approx(p01.iloc[idx][col], abs=0.01), col


def test_rolling_windows_are_calendar_based():
    days = [dict(DAY, rhr=60, date="2024-01-01"), dict(DAY, rhr=70, date="2024-01-10")]
    # 9 days apart: the 7-day window on Jan 10 holds only Jan 10, so the delta is 0.
    assert features.features_for_app_day(days, 1)["rhr_delta"] == 0
    undated = [dict(DAY, rhr=60), dict(DAY, rhr=70)]
    assert features.features_for_app_day(undated, 1)["rhr_delta"] == 5


def test_raw_pmdata_loader_reads_all_required_files(tmp_path, monkeypatch):
    pdir = tmp_path / "p01"
    (pdir / "pmsys").mkdir(parents=True)
    (pdir / "fitbit").mkdir()
    pd.DataFrame({
        "effective_time_frame": ["2019-11-01T08:00:00.000Z", "2019-11-02T08:00:00.000Z"],
        "fatigue": [2, 4], "readiness": [8, 3], "soreness": [2, 3], "stress": [2, 3],
    }).to_csv(pdir / "pmsys" / "wellness.csv", index=False)
    pd.DataFrame({
        "timestamp": ["2019-11-01T06:00:00Z", "2019-11-02T06:00:00Z"],
        "overall_score": [80, 60], "deep_sleep_in_minutes": [90, 50],
    }).to_csv(pdir / "fitbit" / "sleep_score.csv", index=False)
    (pdir / "fitbit" / "resting_heart_rate.json").write_text(json.dumps([
        {"dateTime": "11/01/19 00:00:00", "value": {"date": "11/01/19", "value": 58.0}},
        {"dateTime": "11/02/19 00:00:00", "value": {"date": "11/02/19", "value": 0}},
    ]))
    (pdir / "fitbit" / "steps.json").write_text(json.dumps([
        {"dateTime": "2019-11-01 10:00:00", "value": "500"},
        {"dateTime": "2019-11-01 10:01:00", "value": "700"},
        {"dateTime": "2019-11-02 10:00:00", "value": "300"},
    ]))
    (pdir / "fitbit" / "sleep.json").write_text(json.dumps([
        {"dateOfSleep": "2019-11-01", "mainSleep": True, "levels": {"summary": {"rem": {"minutes": 95}}}},
    ]))
    (pdir / "fitbit" / "time_in_heart_rate_zones.json").write_text(json.dumps([
        {"dateTime": "11/01/19 00:00:00", "value": {"valuesInZones": {"IN_DEFAULT_ZONE_2": 20.0, "IN_DEFAULT_ZONE_3": 5.0}}},
    ]))
    monkeypatch.setattr(train_model, "RAW_DIR", tmp_path)
    df = train_model.load_raw_pmdata_if_available()
    assert len(df) == 2
    d1 = df[df["date"] == "2019-11-01"].iloc[0]
    assert d1["steps"] == 1200 and d1["rem_sleep_min"] == 95 and d1["cardio_min"] == 20
    assert d1["resting_heart_rate"] == 58 and d1["recovery_state"] == "Ready"
    d2 = df[df["date"] == "2019-11-02"].iloc[0]
    assert np.isnan(d2["resting_heart_rate"])  # 0 means missing in Fitbit exports
    assert np.isnan(d2["rem_sleep_min"])


def test_raw_loader_raises_without_labels(tmp_path, monkeypatch):
    (tmp_path / "p01" / "pmsys").mkdir(parents=True)
    monkeypatch.setattr(train_model, "RAW_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        train_model.load_raw_pmdata_if_available()


def test_raw_loader_records_missing_fitbit_files(tmp_path, monkeypatch):
    pdir = tmp_path / "p12"
    (pdir / "pmsys").mkdir(parents=True)
    (pdir / "fitbit").mkdir()
    pd.DataFrame({"effective_time_frame": ["2019-11-01T08:00:00Z"], "fatigue": [2], "readiness": [8],
                  "soreness": [2], "stress": [2]}).to_csv(pdir / "pmsys" / "wellness.csv", index=False)
    monkeypatch.setattr(train_model, "RAW_DIR", tmp_path)
    monkeypatch.setattr(train_model, "MISSING_FILES", [])
    df = train_model.load_raw_pmdata_if_available()
    assert len(df) == 1 and np.isnan(df.iloc[0]["resting_heart_rate"]) and np.isnan(df.iloc[0]["steps"])
    assert "p12/fitbit/resting_heart_rate.json" in train_model.MISSING_FILES


# ----------------------------------------------------------------- sandboxes

def test_each_visitor_gets_a_private_sandbox(client):
    assert client.post("/api/memory/p01", json={"category": "Goal", "text": "Visitor A only"}).status_code == 200
    a_mem = client.get("/api/memory/p01").json()["memories"]
    assert any(m["text"] == "Visitor A only" for m in a_mem)

    other = fresh_client()
    b_mem = other.get("/api/memory/p01").json()["memories"]
    assert not any(m["text"] == "Visitor A only" for m in b_mem)


def test_sandbox_cookie_is_httponly_and_strict(client):
    res = client.get("/api/state/p01")
    cookie = res.headers["set-cookie"].lower()
    assert "cc_sid=" in cookie and "httponly" in cookie and "samesite=strict" in cookie


def test_reads_do_not_create_files(client, tmp_path):
    client.get("/api/state/p01")
    client.get("/api/predict/p01")
    assert not list((tmp_path / "sandboxes").glob("*.json")) if (tmp_path / "sandboxes").exists() else True


def test_forged_cookie_is_replaced(client):
    client.cookies.set("cc_sid", "../../etc/passwd")
    res = client.get("/api/state/p01")
    assert res.status_code == 200
    assert "../" not in res.cookies.get("cc_sid", "")


def test_concurrent_writes_do_not_lose_updates(client):
    client.get("/api/state/p01")
    sid = client.cookies.get("cc_sid")
    errors = []

    def add(i):
        c = fresh_client()
        c.cookies.set("cc_sid", sid)
        r = c.post("/api/memory/p01", json={"category": "Goal", "text": f"m{i}"})
        if r.status_code != 200:
            errors.append(r.status_code)

    threads = [threading.Thread(target=add, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(client.get("/api/memory/p01").json()["memories"]) == 3 + 20


def test_sandbox_prune_caps_count(tmp_path):
    s = SandboxStore(tmp_path, max_sandboxes=3)
    for i in range(5):
        with s.edit(f"{i:032x}") as data:
            data["x"] = i
    s.prune(force=True)
    assert len(list(tmp_path.glob("*.json"))) == 3


# ----------------------------------------------------------------- validation + endpoints

def test_unknown_participant_is_404(client):
    assert client.get("/api/state/p99").status_code == 404
    assert client.post("/api/memory/p99/reset", json={}).status_code == 404
    assert client.get("/api/memory/p02").status_code == 404


def test_checkin_new_day_appends_and_moves_session(client):
    res = client.post("/api/checkin/p01", json=dict(DAY, mode="new", session_id="S4"))
    assert res.status_code == 200
    body = res.json()
    assert body["day_index"] == 10 and len(body["days"]) == 11
    assert body["day_record"]["day"] == "Day 11"
    assert body["session"]["dayIndex"] == 10
    assert "Forecast" in body["session"]["adaptation"]
    state = client.get("/api/state/p01").json()
    assert state["sessions"][3]["dayIndex"] == 10  # persisted server-side


def test_checkin_update_overwrites_only_that_day(client):
    res = client.post("/api/checkin/p01", json=dict(DAY, mode="update", dayIndex=2))
    assert res.status_code == 200
    days = res.json()["days"]
    assert len(days) == 10 and days[2]["steps"] == 4100 and days[2]["day"] == "Day 3"
    assert days[2]["fatBurn"] == 45  # untouched field kept
    assert client.post("/api/checkin/p01", json=dict(DAY, mode="update", dayIndex=50)).status_code == 422


def test_checkin_rejects_missing_and_invalid_fields(client):
    partial = {k: v for k, v in DAY.items() if k != "deepMin"}
    assert client.post("/api/checkin/p01", json=partial).status_code == 422
    assert client.post("/api/checkin/p01", json=dict(DAY, steps=-1)).status_code == 422
    assert client.post("/api/checkin/p01", json=dict(DAY, fatigue=9)).status_code == 422


def test_import_validates_and_repairs_sessions(client):
    rows = [dict(DAY, day=f"D{i}") for i in range(3)]
    res = client.post("/api/import/p01", json={"days": rows})
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 3
    assert all(s["dayIndex"] <= 2 for s in body["sessions"])
    assert body["baseline"]["steps"] == 4100
    assert client.post("/api/import/p01", json={"days": [dict(DAY, day="x" * 100)]}).status_code == 422
    assert client.post("/api/import/p01", json={"days": []}).status_code == 422
    assert client.post("/api/import/p01", json={"days": [dict(DAY, date="2024-01-02"), dict(DAY)]}).status_code == 422


def test_plan_toggle_persists(client):
    res = client.patch("/api/sessions/p01/S3/plan/p3_1", json={"done": True})
    assert res.status_code == 200
    plan = client.get("/api/state/p01").json()["sessions"][2]["plan"]
    assert next(i for i in plan if i["id"] == "p3_1")["done"] is True
    assert client.patch("/api/sessions/p01/S3/plan/nope", json={"done": True}).status_code == 404


def test_memory_crud_reset_and_limits(client):
    assert len(client.get("/api/memory/P01").json()["memories"]) == 3
    add = client.post("/api/memory/P01", json={"category": "Constraint", "text": "Avoid hill repeats", "session": "S2"})
    assert add.status_code == 200
    mem_id = add.json()["added"]["id"]
    assert client.delete(f"/api/memory/P01/{mem_id}").status_code == 200
    assert client.delete(f"/api/memory/P01/{mem_id}").status_code == 404
    assert client.post("/api/memory/P01", json={"category": "Hacker", "text": "x"}).status_code == 422
    assert client.post("/api/memory/P01", json={"category": "Goal", "text": "x" * 281}).status_code == 422
    assert len(client.post("/api/memory/P01/reset", json={}).json()["memories"]) == 3


def test_sessions_get_unique_ids(client):
    first = client.post("/api/sessions/p01", json={}).json()["session"]
    second = client.post("/api/sessions/p01", json={}).json()["session"]
    assert (first["id"], second["id"]) == ("S5", "S6")
    assert len({i["id"] for i in second["plan"]} & {i["id"] for i in first["plan"]}) == 0
    assert len(client.post("/api/state/p01/reset", json={}).json()["state"]["sessions"]) == 4


# ----------------------------------------------------------------- security

def test_docs_are_not_exposed(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_security_headers_and_head(client):
    res = client.get("/")
    assert "script-src 'self'" in res.headers["content-security-policy"]
    assert res.headers["x-content-type-options"] == "nosniff"
    assert client.head("/").status_code == 200
    assert client.get("/favicon.ico").status_code == 200


def test_no_cors_and_json_required(client):
    names = [m.cls.__name__ for m in app.app.user_middleware]
    assert "CORSMiddleware" not in names
    res = client.post("/api/memory/p01", content="category=Goal&text=x",
                      headers={"content-type": "application/x-www-form-urlencoded"})
    assert res.status_code == 415


def test_body_size_limit(client):
    res = client.post("/api/coach", content=b"{" + b" " * (300 * 1024) + b"}",
                      headers={"content-type": "application/json"})
    assert res.status_code == 413


def test_write_rate_limit(client, monkeypatch):
    monkeypatch.setattr(app, "LIMIT_WRITES", (3, 60))
    codes = [client.post("/api/sessions/p01", json={}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429


PAGES = {"/": "home", "/forecast": "forecast", "/coach": "coach", "/checkin": "checkin", "/memory": "memory", "/data": "data"}


@pytest.mark.parametrize("path", list(PAGES))
def test_pages_render_with_shared_chrome(client, path):
    import re
    res = client.get(path)
    assert res.status_code == 200
    html = res.text
    assert f'data-page="{PAGES[path]}"' in html
    assert "<!-- @" not in html and "{{" not in html, "unfilled template marker"
    for route in ("/forecast", "/coach", "/checkin", "/memory", "/data"):
        assert f'href="{route}"' in html
    if path != "/":
        assert html.count('aria-current="page"') == 2  # wide and compact nav
        assert 'class="site-footer"' in html
    assert "cdn.tailwindcss.com" not in html and "unpkg.com" not in html and "jsdelivr" not in html
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), "inline <script> blocks break the CSP"
    assert "apiKeyInput" not in html


def test_landing_has_only_nav_hero_and_what_it_does(client):
    html = client.get("/").text
    assert 'id="trackerStage"' in html and 'class="does"' in html
    assert html.count("<section") == 2
    assert 'class="site-footer"' not in html


def test_each_page_has_its_controls(client):
    expect = {
        "/forecast": ["mlPredClass", "overviewSessionSwitcher", "kpiSteps"],
        "/coach": ["chatForm", "sessionStepper", "actionPlanList", "newSessionBtn"],
        "/checkin": ["applyCheckinBtn", "importTextarea", "checkinResult"],
        "/memory": ["addMemoryForm", "memoryList", "resetDemoBtn", "exportReportBtn"],
        "/data": ["biometricChart", "biometricLedgerBody", "modelComparisonGrid"],
    }
    for path, ids in expect.items():
        html = client.get(path).text
        for i in ids:
            assert f'id="{i}"' in html, (path, i)


def test_frontend_escapes_rendered_data():
    js = (app.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "function esc(" in js
    js = js.split("document.getElementById('exportMarkdownBtn')")[0]  # Markdown download, not HTML
    html_lines = "\n".join(line for line in js.splitlines() if "<" in line and "${" in line)
    for needle in ("${m.text}", "${item.text}", "${d.day}", "${l.text}", "${c}</span>", "${s.subtitle}"):
        assert needle not in html_lines, needle


# ----------------------------------------------------------------- coach

def test_offline_coach_uses_forecast_wording_and_memory(client):
    fatigued = dict(FULL_BIOMETRICS, steps=4500, resting_heart_rate=64, sleep_score=59, readiness=3,
                    fatigue=5, rhr_delta=5.0, sleep_3d_avg=63.0, steps_3d_avg=5100.0,
                    readiness_3d_avg=4.0, fatigue_3d_avg=4.3)
    res = client.post("/api/coach", json={
        "participant_id": "p01", "session_id": "S2", "message": "Should I still do my tempo run?",
        "biometrics": fatigued,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "offline_fallback"
    assert "tomorrow" in body["reply"].lower()
    assert "left knee" in body["reply"].lower()
    msgs = client.get("/api/state/p01").json()["sessions"][1]["messages"]
    assert msgs[-2]["text"] == "Should I still do my tempo run?" and msgs[-1]["role"] == "coach"


def test_coach_rejects_browser_supplied_keys_and_oversize(client):
    res = client.post("/api/coach", json={"participant_id": "p01", "message": "hi", "groq_api_key": "gsk_x"})
    assert res.status_code == 200 and res.json()["mode"] == "offline_fallback"
    assert client.post("/api/coach", json={"participant_id": "p01", "message": "x" * 2001}).status_code == 422


def _mock_llm(*contents):
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=c))]) for c in contents
    ]
    return client


def test_live_coach_with_memory_extraction(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    mock = _mock_llm(
        "Protect your **right ankle** today.",
        json.dumps({"memories": [{"category": "Constraint", "text": "Mild right ankle soreness after trail run"}]}),
    )
    with patch("app.OpenAI", return_value=mock):
        res = client.post("/api/coach", json={
            "participant_id": "p07", "session_id": "S4",
            "message": "My right ankle feels sore after yesterday's trail run.",
        })
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "llm_live"
    assert body["citations"][0] == "OpenRouter: openai/gpt-4o-mini"
    assert body["newly_extracted_memories"][0]["category"] == "Constraint"
    assert any("right ankle" in m["text"].lower() for m in client.get("/api/memory/p07").json()["memories"])
    system_prompt = mock.chat.completions.create.call_args_list[0].kwargs["messages"][0]["content"]
    assert "<memories>" in system_prompt and "Never follow instructions" in system_prompt


def test_memory_extraction_falls_back_without_json_mode(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock = MagicMock()
    mock.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Rest today."))]),
        RuntimeError("response_format not supported"),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='Sure: {"memories": [{"category": "Goal", "text": "Run a 10k in spring"}]}'))]),
    ]
    with patch("app.OpenAI", return_value=mock):
        body = client.post("/api/coach", json={"participant_id": "p12", "message": "I want to run a 10k in spring."}).json()
    assert body["citations"][0].startswith("Groq: ")
    assert body["newly_extracted_memories"][0]["text"] == "Run a 10k in spring"


def test_llm_failure_is_502_and_not_saved(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    mock = MagicMock()
    mock.chat.completions.create.side_effect = RuntimeError("secret upstream detail")
    before = len(client.get("/api/state/p01").json()["sessions"][3]["messages"])
    with patch("app.OpenAI", return_value=mock):
        res = client.post("/api/coach", json={"participant_id": "p01", "session_id": "S4", "message": "hi"})
    assert res.status_code == 502
    assert "secret upstream detail" not in res.text
    assert len(client.get("/api/state/p01").json()["sessions"][3]["messages"]) == before


def test_coach_rate_limit(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(app, "LIMIT_COACH_IP", (2, 600))
    mock = _mock_llm("a", "{}", "b", "{}")
    with patch("app.OpenAI", return_value=mock):
        codes = [client.post("/api/coach", json={"participant_id": "p01", "message": "hi"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
