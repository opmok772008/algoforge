"""
tests/test_api_endpoints.py — Direct route handler tests for FastAPI backend matching the frontend.
Does not require httpx or external test servers.
"""

from __future__ import annotations

import asyncio
import pytest

from biosig_fuzzy_pso.api.main import (
    health,
    serve_frontend,
    serve_root_frontend,
    run_tests_endpoint,
    analyze_ecg,
    classify,
    simulate_ecg,
    get_params,
    update_params,
    AnalyzeRequest,
    ClassifyRequest,
    SimulateRequest,
    ParamsRequest,
)
from biosig_fuzzy_pso.api.sim_engine import Sim


def test_health_endpoint():
    res = asyncio.run(health())
    assert res.status_code == 200
    import json
    data = json.loads(res.body)
    assert data["status"] == "ok"
    assert data["service"] == "biosig_fuzzy_pso"
    assert data["particle_dim"] == 74


def test_frontend_serving():
    res_root = asyncio.run(serve_frontend())
    assert res_root.status_code == 200
    assert "Fuzzy-Evolutionary ICU Arrhythmia Pipeline" in res_root.body.decode("utf-8")

    res_html = asyncio.run(serve_root_frontend())
    assert res_html.status_code == 200
    assert "Lethal arrhythmias, found through the noise" in res_html.body.decode("utf-8")


def test_tests_endpoint():
    import json
    res = asyncio.run(run_tests_endpoint())
    assert res.status_code == 200
    data = json.loads(res.body)
    assert data["total"] == 9
    assert data["passed"] == 9
    assert "9 / 9 passed" in data["summary"]
    assert len(data["tests"]) == 9


def test_analyze_endpoint():
    import json
    sim = Sim(seed=42)
    samples = [sim.next({"scn": "normal"}) for _ in range(300 * 5)]
    req = AnalyzeRequest(samples=samples, fs=300)
    res = asyncio.run(analyze_ecg(req))
    assert res.status_code == 200
    data = json.loads(res.body)
    assert "p" in data
    assert "hr" in data
    assert "rhythm" in data
    assert "advice" in data
    assert "items" in data
    assert len(data["items"]) == 10


def test_classify_endpoint_diagnostics():
    sim = Sim(seed=42)
    samples = [sim.next({"scn": "normal"}) for _ in range(250 * 4)]
    req = ClassifyRequest(samples=samples, fs=250, scn="normal")
    res = asyncio.run(classify(req))
    assert res.class_name in ["Normal", "VT", "VF", "Artifact", "Other"]
    assert hasattr(res, "lethal_score")
    assert hasattr(res, "memberships")
    assert "hrHigh" in res.memberships
    assert "regular" in res.memberships
    assert "conf" in res.memberships
    assert hasattr(res, "banner_text")
    assert hasattr(res, "status_class")


def test_simulate_endpoint():
    import json
    req = SimulateRequest(scn="vt", wander=True, duration_sec=2.0, fs=250)
    res = asyncio.run(simulate_ecg(req))
    assert res.status_code == 200
    data = json.loads(res.body)
    assert len(data["samples"]) == 500
    assert data["scn"] == "vt"


def test_params_endpoints():
    import json
    res_get = asyncio.run(get_params())
    assert res_get.status_code == 200
    data_get = json.loads(res_get.body)
    assert "active_detector" in data_get

    req_post = ParamsRequest(bw=210, sm=6)
    res_post = asyncio.run(update_params(req_post))
    assert res_post.status_code == 200
    data_post = json.loads(res_post.body)
    assert data_post["active_detector"]["bw"] == 210
    assert data_post["active_detector"]["sm"] == 6
