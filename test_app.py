"""Tests for the CareCompanion backend: forecast, accounts and JWT, validation, security and the coach."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app
import auth
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
def anon(tmp_path, monkeypatch):
    """A signed-out browser."""
    monkeypatch.setattr(app, "store", SandboxStore(tmp_path / "accounts"))
    for var in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"):
        monkeypatch.delenv(var, raising=False)
    auth.set_secret("test-secret-" + "x" * 40)
    auth.revoked.clear()
    real_users = auth.users.path
    auth.users.reset(tmp_path / "users.json")
    app.limiter.reset()
    yield TestClient(app.app)
    auth.set_secret(None)
    auth.users.reset(real_users)


def sign_in(c, username="alex", password=auth.DEMO_PASSWORD):
    return c.post("/api/auth/login", json={"username": username, "password": password})


@pytest.fixture()
def client(anon):
    """Signed in as the demo account alex (profile p01)."""
    assert sign_in(anon).status_code == 200
    app.limiter.reset()  # the sign-in itself counts as a write
    return anon


def fresh_client(username=None):
    c = TestClient(app.app)
    if username:
        assert sign_in(c, username).status_code == 200
    return c


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


# ----------------------------------------------------------------- accounts + JWT

def test_login_sets_httponly_strict_jwt_cookie(anon):
    res = sign_in(anon)
    assert res.status_code == 200
    body = res.json()
    assert body["user"] == {"username": "alex", "pid": "P01", "name": "Alex Rivera", "role": "Half-Marathon Prep, Morning Runner"}
    assert body["token_type"] == "bearer" and body["expires_in"] == auth.TOKEN_TTL_S
    cookie = res.headers["set-cookie"].lower()
    assert "cc_token=" in cookie and "httponly" in cookie and "samesite=strict" in cookie
    claims = auth.decode_token(body["access_token"])
    assert claims["sub"] == "alex" and claims["pid"] == "p01" and claims["exp"] - claims["iat"] == auth.TOKEN_TTL_S
    assert anon.get("/api/auth/me").json()["user"]["name"] == "Alex Rivera"


def test_wrong_password_and_unknown_user_are_401(anon):
    assert sign_in(anon, "alex", "nope").status_code == 401
    assert sign_in(anon, "mallory", "demo1234").status_code == 401
    assert sign_in(anon, "ALEX ", auth.DEMO_PASSWORD).status_code == 200  # usernames are case-insensitive
    assert anon.post("/api/auth/login", json={"username": "alex", "password": "x", "admin": True}).status_code == 422


def test_login_is_rate_limited(anon, monkeypatch):
    monkeypatch.setattr(app, "LIMIT_LOGIN", (3, 300))
    codes = [sign_in(anon, "alex", "bad").status_code for _ in range(4)]
    assert codes == [401, 401, 401, 429]


def test_api_requires_a_token(anon):
    for path in ("/api/state/p01", "/api/memory/p01", "/api/predict/p01", "/api/auth/me", "/api/demo/profiles"):
        res = anon.get(path)
        assert res.status_code == 401 and res.headers["www-authenticate"] == "Bearer"
    assert anon.post("/api/coach", json={"message": "hi"}).status_code == 401
    assert anon.post("/api/demo/run", json={"profile": "p01"}).status_code == 401
    # Public: health, model card, the stateless model endpoint.
    assert anon.get("/healthz").status_code == 200 and anon.get("/api/model-info").status_code == 200


def test_bearer_header_works_without_cookie(anon):
    token = sign_in(anon).json()["access_token"]
    other = TestClient(app.app)
    assert other.get("/api/state/p01", headers={"Authorization": f"Bearer {token}"}).status_code == 200


@pytest.mark.parametrize("make", [
    lambda: "not-a-jwt",
    lambda: __import__("jwt").encode({"sub": "alex", "pid": "p01", "iat": 0, "nbf": 0, "exp": 9999999999, "jti": "x",
                                      "iss": auth.ISSUER, "aud": auth.AUDIENCE}, "wrong-key-" + "y" * 40, algorithm="HS256"),
    lambda: __import__("jwt").encode({"sub": "alex", "pid": "p01", "iat": 0, "nbf": 0, "exp": 1, "jti": "x",
                                      "iss": auth.ISSUER, "aud": auth.AUDIENCE}, auth.secret(), algorithm="HS256"),
    lambda: __import__("jwt").encode({"sub": "alex", "pid": "p07", "iat": 0, "nbf": 0, "exp": 9999999999, "jti": "x",
                                      "iss": auth.ISSUER, "aud": auth.AUDIENCE}, auth.secret(), algorithm="HS256"),
    lambda: __import__("jwt").encode({"sub": "alex", "pid": "p01", "iat": 0, "nbf": 0, "exp": 9999999999, "jti": "x",
                                      "iss": auth.ISSUER, "aud": "someone-else"}, auth.secret(), algorithm="HS256"),
    lambda: "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJhbGV4IiwicGlkIjoicDAxIn0.",  # alg=none
], ids=["garbage", "wrong-key", "expired", "pid-mismatch", "wrong-audience", "alg-none"])
def test_forged_or_expired_tokens_are_rejected(anon, make):
    anon.cookies.set(auth.COOKIE, make())
    assert anon.get("/api/state/p01").status_code == 401


def test_logout_revokes_the_token(anon):
    token = sign_in(anon).json()["access_token"]
    assert anon.post("/api/auth/logout", json={}).status_code == 200
    assert "cc_token=" in anon.post("/api/auth/logout", json={}).headers.get("set-cookie", "")  # cookie cleared
    other = TestClient(app.app)
    assert other.get("/api/state/p01", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_an_account_cannot_touch_another_profile(client):
    assert client.get("/api/state/p07").status_code == 403
    assert client.post("/api/memory/p07", json={"category": "Goal", "text": "x"}).status_code == 403
    assert client.post("/api/coach", json={"participant_id": "p12", "message": "hi"}).status_code == 403


def test_account_data_persists_across_logins_and_is_private(client):
    assert client.post("/api/memory/p01", json={"category": "Goal", "text": "Alex only"}).status_code == 200
    client.post("/api/auth/logout", json={})
    again = fresh_client("alex")  # a new browser, same account
    assert any(m["text"] == "Alex only" for m in again.get("/api/memory/p01").json()["memories"])
    maya = fresh_client("maya")
    assert maya.get("/api/state/p07").json()["name"] == "Maya Lin"
    assert not any(m["text"] == "Alex only" for m in maya.get("/api/memory/p07").json()["memories"])


def test_reads_do_not_create_files(client, tmp_path):
    client.get("/api/state/p01")
    client.get("/api/predict/p01")
    assert not list((tmp_path / "accounts").glob("*.json")) if (tmp_path / "accounts").exists() else True


def test_concurrent_writes_do_not_lose_updates(client):
    token = client.cookies.get(auth.COOKIE)
    errors = []

    def add(i):
        c = TestClient(app.app)
        r = c.post("/api/memory/p01", json={"category": "Goal", "text": f"m{i}"}, headers={"Authorization": f"Bearer {token}"})
        if r.status_code != 200:
            errors.append(r.status_code)

    threads = [threading.Thread(target=add, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(client.get("/api/memory/p01").json()["memories"]) == 3 + 20


def test_safe_next_blocks_open_redirects():
    assert auth.safe_next("/coach") == "/coach"
    for bad in ("//evil.com", "https://evil.com", "/\\evil.com", "", None):
        assert auth.safe_next(bad) == "/"


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
    assert client.get("/api/state/p07").status_code == 403  # exists, but belongs to maya


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


PAGES = {"/": "home", "/demo": "demo", "/forecast": "forecast", "/coach": "coach", "/checkin": "checkin", "/memory": "memory", "/data": "data"}


@pytest.mark.parametrize("path", list(PAGES))
def test_pages_render_with_shared_chrome(client, path):
    import re
    res = client.get(path)
    assert res.status_code == 200
    html = res.text
    assert f'data-page="{PAGES[path]}"' in html
    assert "<!-- @" not in html and "{{" not in html, "unfilled template marker"
    for route in ("/demo", "/forecast", "/coach", "/checkin", "/memory", "/data"):
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
        "/demo": ["demoForm", "demoProfile", "demoSessions"],
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


def _reply(item):
    """A fake chat completion: a string is a text reply, a list of (name, args) is a tool-call turn."""
    if isinstance(item, (list, tuple)):
        calls = [SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
                 for i, (n, a) in enumerate(item)]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=calls))])
    if isinstance(item, Exception):
        return item
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=item, tool_calls=None))])


def _mock_llm(*items):
    client = MagicMock()
    client.chat.completions.create.side_effect = [_reply(i) for i in items]
    return client


READ_BOTH = [("get_saved_facts", {}), ("get_past_sessions", {})]


def test_live_coach_agent_uses_tools_saves_and_plans(anon, monkeypatch):
    client = anon
    sign_in(client, "maya")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    mock = _mock_llm(
        READ_BOTH + [("save_fact", {"category": "Constraint", "text": "Mild right ankle soreness after trail run"})],
        [("get_trends", {"days": 7}), ("update_plan", {"items": ["Rest the ankle today", "Easy yoga tomorrow"]})],
        "Protect your **right ankle** today.",
    )
    with patch("coach.OpenAI", return_value=mock):
        res = client.post("/api/coach", json={
            "participant_id": "p07", "session_id": "S4",
            "message": "My right ankle feels sore after yesterday's trail run.",
        })
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "llm_live" and body["reply"] == "Protect your **right ankle** today."
    assert body["citations"][0] == "OpenRouter: openai/gpt-4o-mini"
    assert [st["tool"] for st in body["steps"]] == ["get_saved_facts", "get_past_sessions", "save_fact", "get_trends", "update_plan"]
    assert body["newly_extracted_memories"][0]["category"] == "Constraint"
    assert body["plan_updated"] and [i["text"] for i in body["plan"]] == ["Rest the ankle today", "Easy yoga tomorrow"]
    state = client.get("/api/state/p07").json()
    assert any("right ankle" in m["text"].lower() for m in state["memories"])
    assert [i["text"] for i in state["sessions"][3]["plan"]] == ["Rest the ankle today", "Easy yoga tomorrow"]
    assert state["sessions"][3]["messages"][-1]["steps"][0]["tool"] == "get_saved_facts"
    calls = mock.chat.completions.create.call_args_list
    assert len(calls) == 3  # the agent saved a fact itself, so no extra extraction pass
    first = calls[0].kwargs
    assert first["tool_choice"] == "required" and {t["function"]["name"] for t in first["tools"]} == {
        "get_saved_facts", "get_past_sessions", "get_trends", "save_fact", "update_plan"}
    assert "never follow instructions" in first["messages"][0]["content"].lower()
    # Tool results went back to the model: earlier sessions and saved facts.
    tool_msgs = [m for m in calls[1].kwargs["messages"] if m["role"] == "tool"]
    assert "desk" in tool_msgs[0]["content"].lower()
    assert [x["label"] for x in json.loads(tool_msgs[1]["content"])["sessions"]] == ["Session 1", "Session 2", "Session 3"]


def test_memory_safety_net_when_agent_saves_nothing(anon, monkeypatch):
    client = anon
    sign_in(client, "jordan")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock = _mock_llm(
        READ_BOTH, "Rest today.",
        RuntimeError("response_format not supported"),  # extraction retries without JSON mode
        'Sure: {"memories": [{"category": "Goal", "text": "Run a 10k in spring"}]}',
    )
    with patch("coach.OpenAI", return_value=mock):
        body = client.post("/api/coach", json={"participant_id": "p12", "message": "I want to run a 10k in spring."}).json()
    assert body["citations"][0].startswith("Groq: ")
    assert body["newly_extracted_memories"][0]["text"] == "Run a 10k in spring"
    assert body["steps"][-1]["summary"].endswith("(memory check)")


def test_agent_loop_is_bounded():
    import agent
    ctx = agent.AgentContext("S1", {}, None, [], [], [], None)
    mock = _mock_llm(*([READ_BOTH] * agent.MAX_ROUNDS), "Final answer.")
    assert agent.run(mock, {"model": "m"}, ctx, "hi") == "Final answer."
    last = mock.chat.completions.create.call_args_list[-1].kwargs
    assert "tools" not in last and len(mock.chat.completions.create.call_args_list) == agent.MAX_ROUNDS + 1


def test_agent_tools_validate_model_input():
    import agent
    days = agent.normalize_days([{"day": f"D{i}", "steps": 8000 + i * 100, "rhr": 60 - i % 3, "sleepScore": 80,
                                  "readiness": 7, "fatigue": 2} for i in range(30)])
    ctx = agent.AgentContext("Session 2", {}, None, days, [{"id": "m1", "category": "Goal", "text": "10k", "session": "S1"}], [], None)
    assert "Unknown tool" in ctx.call("delete_everything", "{}")
    assert "Bad arguments" in ctx.call("save_fact", "not json")
    assert "Bad arguments" in ctx.call("get_saved_facts", '{"category": "Goal"}')  # no such parameter
    assert json.loads(ctx.call("save_fact", json.dumps({"category": "Goal", "text": "10k"})))["saved"] is False  # duplicate
    for i in range(agent.MAX_NEW_FACTS):
        assert json.loads(ctx.call("save_fact", json.dumps({"category": "Hacker", "text": f"fact {i}"})))["saved"]
    assert json.loads(ctx.call("save_fact", json.dumps({"category": "Goal", "text": "one too many"})))["saved"] is False
    assert {m["category"] for m in ctx.new_memories} == {"Baseline"}  # unknown categories are coerced
    assert json.loads(ctx.call("update_plan", json.dumps({"items": ["a"]})))["updated"] is False  # no open session
    trends = json.loads(ctx.call("get_trends", json.dumps({"days": 500})))
    assert trends["days_covered"] == 30 and trends["steps"]["direction"] == "rising"
    assert trends["steps"]["latest"] == 10900


def test_llm_failure_is_502_and_not_saved(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    mock = MagicMock()
    mock.chat.completions.create.side_effect = RuntimeError("secret upstream detail")
    before = len(client.get("/api/state/p01").json()["sessions"][3]["messages"])
    with patch("coach.OpenAI", return_value=mock):
        res = client.post("/api/coach", json={"participant_id": "p01", "session_id": "S4", "message": "hi"})
    assert res.status_code == 502
    assert "secret upstream detail" not in res.text
    assert len(client.get("/api/state/p01").json()["sessions"][3]["messages"]) == before


def test_coach_rate_limit(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(app, "LIMIT_COACH_IP", (2, 600))
    mock = _mock_llm(READ_BOTH, "a", "{}", READ_BOTH, "b", "{}")
    with patch("coach.OpenAI", return_value=mock):
        codes = [client.post("/api/coach", json={"participant_id": "p01", "message": "hi"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


# ----------------------------------------------------------------- Azure, agent memory, the 3-session demo

def test_azure_openai_is_preferred(monkeypatch):
    import coach
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    cfg = coach.llm_config()
    assert cfg["provider"] == "Azure OpenAI" and cfg["model"] == "gpt-4o-mini"
    client = coach.make_llm_client(cfg)
    assert type(client).__name__ == "AzureOpenAI"
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT")
    assert coach.llm_config()["provider"] == "OpenRouter"  # incomplete Azure config falls through


def test_live_coach_uses_azure_client(client, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    mock = _mock_llm(READ_BOTH, "Rest today.", "{}")
    with patch("coach.AzureOpenAI", return_value=mock) as az:
        body = client.post("/api/coach", json={"participant_id": "p01", "session_id": "S4", "message": "hi"}).json()
    assert az.called and body["citations"][0] == "Azure OpenAI: gpt-4o-mini"


def test_offline_memory_extraction():
    import coach
    got = coach.extract_memories_offline(
        "I'm training for a 10 km race. My right calf gets tight after intervals. I can only train before work. Should I run?",
        "Session 1", [])
    cats = [m["category"] for m in got]
    assert cats == ["Goal", "Constraint", "Baseline"]
    assert not coach.extract_memories_offline("Should I run today?", "S1", [])


def test_offline_coach_refers_to_previous_session(client):
    body = client.post("/api/coach", json={"participant_id": "p01", "session_id": "S3", "message": "Ready to ramp up?"}).json()
    assert body["reply"].startswith("Since Session 2 (Day 5)")
    assert [st["summary"] for st in body["steps"]] == ["Read 3 saved facts", "Read 2 earlier sessions"]
    assert "Agent steps: 2" in body["citations"]


@pytest.mark.parametrize("profile", ["p01", "p07", "p12", "pmdata:p05"])
def test_demo_offline_shows_advice_changing(client, profile):
    res = client.post("/api/demo/run", json={"profile": profile})
    assert res.status_code == 200
    data = res.json()
    sessions = data["sessions"]
    assert len(sessions) == 3 and data["mode"] == "offline_fallback"
    assert sessions[0]["memories_used"] == []                       # starts blank
    assert sessions[0]["new_memories"], "session 1 should save facts"
    assert len(sessions[1]["memories_used"]) >= len(sessions[0]["new_memories"])
    assert sessions[1]["advice"].startswith("Since Session 1")      # reads the previous session
    assert len({s["advice"] for s in sessions}) == 3                 # advice differs session to session
    for s in sessions:
        assert s["advice"] != s["stateless_advice"]
        assert "Since" not in s["stateless_advice"]


def test_demo_live_runs_the_agent_with_growing_memory(client, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    replies = []
    for n in range(3):  # per session: tool turn, advice, memory check (agent saved nothing), stateless reply
        replies += [READ_BOTH, f"advice {n}", json.dumps({"memories": [{"category": "Goal", "text": f"fact {n}"}]}), f"generic {n}"]
    mock = _mock_llm(*replies)
    with patch("coach.AzureOpenAI", return_value=mock):
        data = client.post("/api/demo/run", json={"profile": "p07"}).json()
    assert data["provider"] == "Azure OpenAI: gpt-4o-mini"
    sessions = data["sessions"]
    assert [s["advice"] for s in sessions] == ["advice 0", "advice 1", "advice 2"]
    assert [s["stateless_advice"] for s in sessions] == ["generic 0", "generic 1", "generic 2"]
    assert [m["text"] for m in sessions[2]["memories_used"]] == ["fact 0", "fact 1"]
    assert sessions[2]["steps"][1]["summary"] == "Read 2 earlier sessions"
    # The third session's agent read the second session's advice through its tool.
    tool_results = [m["content"] for m in mock.chat.completions.create.call_args_list[9].kwargs["messages"] if m["role"] == "tool"]
    assert any("advice 1" in t for t in tool_results) and any("fact 1" in t for t in tool_results)


def test_demo_rejects_unknown_profile(client):
    assert client.post("/api/demo/run", json={"profile": "pmdata:p99"}).status_code == 404
    assert client.post("/api/demo/run", json={"profile": "../etc"}).status_code == 422


# ----------------------------------------------------------------- sign-in pages

@pytest.mark.parametrize("path", ["/demo", "/forecast", "/coach", "/checkin", "/memory", "/data"])
def test_signed_out_pages_redirect_to_login(anon, path):
    res = anon.get(path, follow_redirects=False)
    assert res.status_code == 303 and res.headers["location"] == f"/login?next={path}"


def test_landing_and_login_are_public_and_show_sign_in(anon):
    home = anon.get("/")
    assert home.status_code == 200 and 'href="/login"' in home.text and "logoutBtn" not in home.text
    login = anon.get("/login")
    assert login.status_code == 200
    for needle in ('id="loginForm"', 'autocomplete="current-password"', 'data-username="alex"', 'data-username="maya"',
                   'data-username="jordan"', "/static/login.js"):
        assert needle in login.text
    assert "<script>" not in login.text  # CSP: no inline scripts


def test_signed_in_header_has_name_and_sign_out(client):
    page = client.get("/coach").text
    assert 'id="accountChip" data-pid="P01"' in page and "Alex Rivera" in page and 'id="logoutBtn"' in page
    assert 'id="personaSelect"' not in page and 'openApiModalBtn' not in page
    header = page[page.index("<header"):page.index("</header>")]
    assert header.index('id="themeToggleBtn"') < header.index('id="accountChip"') < header.index('id="logoutBtn"')  # account on the far right
    res = client.get("/login?next=/memory", follow_redirects=False)  # already signed in
    assert res.status_code == 303 and res.headers["location"] == "/memory"
    assert client.get("/login?next=//evil.com", follow_redirects=False).headers["location"] == "/"


def test_demo_password_can_be_hidden(anon, monkeypatch):
    monkeypatch.setenv("CC_SHOW_DEMO_PASSWORD", "0")
    assert auth.DEMO_PASSWORD not in anon.get("/login").text


# ----------------------------------------------------------------- performance

def test_text_responses_are_gzipped(anon):
    for path in ("/", "/static/app.css", "/static/vendor/chart.umd.min.js"):
        res = anon.get(path, headers={"Accept-Encoding": "gzip"})
        assert res.status_code == 200 and res.headers.get("content-encoding") == "gzip", path


def test_static_assets_are_versioned_and_cached(anon):
    page = anon.get("/").text
    assert "lucide.min.js" not in page and "/static/icons.js?v=" in page
    assert "/static/app.css?v=" in page and "/static/app.js?v=" in page
    v = page.split("/static/app.css?v=")[1].split('"')[0]
    assert "immutable" in anon.get(f"/static/app.css?v={v}").headers["cache-control"]
    assert anon.get("/static/fonts/fonts.css").headers["cache-control"] == "public, max-age=604800"
    assert anon.get("/api/model-info").headers["cache-control"] == "no-store"


# ----------------------------------------------------------------- safety screen

@pytest.mark.parametrize("msg,kind", [
    ("I had chest pain during my run this morning", "chest_pain"),
    ("Felt pressure in my chest on the stairs", "chest_pain"),
    ("I passed out after intervals yesterday", "fainting"),
    ("My heart is racing at rest and I have palpitations", "heart_rhythm"),
    ("I can't breathe properly even sitting down", "breathing"),
    ("Sometimes I want to kill myself", "self_harm"),
])
def test_safety_flags_red_flags(msg, kind):
    import safety
    assert safety.check(msg)["kind"] == kind


@pytest.mark.parametrize("msg", [
    "No chest pain, legs just heavy. Should I run?",
    "I've never fainted, but I feel tired",
    "My calf is tight after intervals",
    "Heart rate zones for tempo runs?",
])
def test_safety_ignores_negations_and_ordinary_aches(msg):
    import safety
    assert safety.check(msg) is None


def test_red_flag_skips_the_llm_and_is_remembered(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    mock = MagicMock()
    with patch("coach.OpenAI", return_value=mock):
        body = client.post("/api/coach", json={"session_id": "S4", "message": "Got chest pain on my tempo run, should I finish the plan?"}).json()
    assert not mock.chat.completions.create.called
    assert body["mode"] == "safety" and body["safety_flag"] == "chest_pain"
    assert "stop training" in body["reply"].lower() and "112" in body["reply"]
    assert body["steps"][0]["tool"] == "safety_check"
    assert any("medical clearance" in m["text"] for m in client.get("/api/memory/p01").json()["memories"])


# ----------------------------------------------------------------- sign-up

def sign_up(c, username="sam_k", password="correct horse", name="Sam K", start="fresh"):
    return c.post("/api/auth/signup", json={"username": username, "password": password, "name": name, "start": start})


def test_signup_fresh_account_starts_empty_and_signs_in(anon):
    res = sign_up(anon)
    assert res.status_code == 201
    user = res.json()["user"]
    assert user["username"] == "sam_k" and user["pid"].startswith("U")
    pid = user["pid"].lower()
    state = anon.get(f"/api/state/{pid}").json()
    assert state["days"] == [] and state["memories"] == [] and state["sample"] is False
    assert state["baseline"] is None and len(state["sessions"]) == 1
    # Nothing that needs a day works yet, and says why.
    res = anon.get(f"/api/predict/{pid}")
    assert res.status_code == 409 and "check-in" in res.json()["detail"]
    # The first check-in creates Day 1 and the baseline.
    res = anon.post(f"/api/checkin/{pid}", json={**DAY, "session_id": "S1"})
    assert res.status_code == 200
    body = res.json()
    assert body["day_index"] == 0 and body["baseline"]["rhr"] == 65 and body["session"]["dayLabel"] == "Day 1"
    assert anon.get(f"/api/predict/{pid}").status_code == 200
    # Sign out and back in with the new password: the day is still there.
    anon.post("/api/auth/logout", json={})
    assert sign_in(anon, "sam_k", "correct horse").status_code == 200
    assert len(anon.get(f"/api/state/{pid}").json()["days"]) == 1


def test_signup_sample_account_gets_real_pmdata_days(anon):
    res = sign_up(anon, start="pmdata:p05")
    assert res.status_code == 201
    pid = res.json()["user"]["pid"].lower()
    state = anon.get(f"/api/state/{pid}").json()
    assert state["sample"] is True and "P05" in state["role"]
    assert len(state["days"]) == 14 and all(d["date"] for d in state["days"])
    assert anon.get(f"/api/predict/{pid}").status_code == 200
    # Reset returns to the same sample days, not to a demo persona.
    anon.post(f"/api/checkin/{pid}", json=DAY)
    reset = anon.post(f"/api/state/{pid}/reset", json={}).json()
    assert len(reset["state"]["days"]) == 14 and reset["state"]["name"] == "Sam K"


@pytest.mark.parametrize("payload, status", [
    ({"username": "alex"}, 409),                 # demo account name
    ({"username": "ab"}, 422),                   # too short
    ({"username": "Bad Name!"}, 422),
    ({"password": "short"}, 422),
    ({"password": "sam_k"}, 422),                # same as the username
    ({"name": "   "}, 422),
    ({"start": "pmdata:p12"}, 422),              # no resting HR in the release, so not offered
    ({"start": "p01"}, 422),
])
def test_signup_rejects_bad_input(anon, payload, status):
    base = {"username": "sam_k", "password": "correct horse", "name": "Sam K", "start": "fresh"}
    res = anon.post("/api/auth/signup", json={**base, **payload})
    assert res.status_code == status
    assert auth.users.count() == 0


def test_signup_duplicate_username_is_rejected(anon):
    assert sign_up(anon).status_code == 201
    other = TestClient(app.app)
    res = sign_up(other, password="another password")
    assert res.status_code == 409 and "taken" in res.json()["detail"]
    assert sign_in(other, "sam_k", "another password").status_code == 401


def test_registered_accounts_are_isolated(anon):
    a = sign_up(anon).json()["user"]["pid"].lower()
    other = TestClient(app.app)
    b = sign_up(other, username="riya").json()["user"]["pid"].lower()
    assert a != b
    assert anon.get(f"/api/state/{b}").status_code == 403
    assert other.get(f"/api/state/{a}").status_code == 403
    assert other.get("/api/state/p01").status_code == 403


def test_signup_password_is_hashed_on_disk(anon):
    sign_up(anon)
    text = auth.users.path.read_text()
    assert "correct horse" not in text
    rec = json.loads(text)["sam_k"]
    assert len(rec["salt"]) == 32 and len(rec["hash"]) == 64
    assert (auth.users.path.stat().st_mode & 0o777) == 0o600


def test_signup_is_rate_limited(anon, monkeypatch):
    monkeypatch.setattr(app, "LIMIT_SIGNUP", (2, 3600))
    codes = [sign_up(TestClient(app.app), username=f"user_{i}").status_code for i in range(3)]
    assert codes == [201, 201, 429]


def test_signup_respects_user_cap(anon, monkeypatch):
    monkeypatch.setattr(auth, "MAX_USERS", 1)
    assert sign_up(anon).status_code == 201
    res = sign_up(TestClient(app.app), username="riya")
    assert res.status_code == 409 and "full" in res.json()["detail"]


def test_signup_page_is_public_and_lists_samples(anon):
    page = anon.get("/signup")
    assert page.status_code == 200
    assert 'id="signupForm"' in page.text and 'value="pmdata:p05"' in page.text and "pmdata:p12" not in page.text
    assert 'href="/signup"' in anon.get("/login").text
    sign_up(anon)
    assert anon.get("/signup", follow_redirects=False).status_code == 303
