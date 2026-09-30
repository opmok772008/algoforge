"""
api/main.py — FastAPI application with REST, WebSocket endpoints, and Frontend Integration.

Endpoints:
  GET  /                  — serve the ICU Arrhythmia Pipeline frontend
  GET  /FRONTEND.html     — serve standalone frontend
  GET  /health            — liveness and diagnostic check
  POST /classify          — classify ECG segment with full fuzzy memberships & fiducials
  WS   /stream            — streaming real-time ECG chunks with per-hop diagnostics
  POST /analyze           — RR-feature ECG analyzer (HR, HRV, RMSSD, NN50, condition & advice)
  GET  /tests             — run the 9-check automated verification suite
  POST /tests             — run the 9-check automated verification suite
  POST /simulate          — generate synthetic ECG with configurable rhythm & interference
  GET  /params            — get current detector & FIS parameters
  POST /params            — update detector parameters (bw, sm, th, mw)
  POST /optimize          — start PSO / GA optimization in background
  GET  /optimize/{id}     — check optimization job status & convergence
  GET  /metrics/{id}      — get benchmark metrics
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from biosig_fuzzy_pso.config import DEFAULT_CONFIG, SAMPLE_RATE
from biosig_fuzzy_pso.dsp.baseline import BaselineRemover
from biosig_fuzzy_pso.dsp.adaptive import NLMSFilter
from biosig_fuzzy_pso.dsp.quality import compute_sqi
from biosig_fuzzy_pso.features.rpeak import RPeakDetector, extract_r_amplitudes
from biosig_fuzzy_pso.features.fiducials import FiducialExtractor
from biosig_fuzzy_pso.fuzzy.fis import SugenoFIS
from biosig_fuzzy_pso.fuzzy.safety_gate import SafetyGate
from biosig_fuzzy_pso.pso.encoding import decode_particle, default_particle, LOWER_BOUNDS, UPPER_BOUNDS
from biosig_fuzzy_pso.api.sim_engine import (
    DEFAULT_PARAMS,
    FS as SIM_FS,
    Sim,
    analyze_ecg_record,
    analyze_window,
    fuzzy_diagnostics,
    run_automated_tests,
)

from biosig_fuzzy_pso.api.schemas import Dataset, SignalPayload, SignalStressRequest

logger = logging.getLogger(__name__)

# Base workspace directory
WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = WORKSPACE_DIR / "frontend"
FRONTEND_HTML = WORKSPACE_DIR / "frontend" / "index.html"
ROOT_FRONTEND_HTML = WORKSPACE_DIR / "FRONTEND.html"

# Global state
_BEST_PARTICLE: np.ndarray = default_particle()
_ACTIVE_PARAMS: Dict[str, Any] = dict(DEFAULT_PARAMS)
_JOBS: Dict[str, Dict[str, Any]] = {}


def _load_best_params() -> np.ndarray:
    path = DEFAULT_CONFIG.api.best_params_file
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            particle = np.array(data["particle"], dtype=np.float32)
            logger.info("Loaded best params from %s", path)
            return particle
        except Exception as e:
            logger.warning("Could not load best params: %s (using defaults)", e)
    return default_particle()


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BioSig Fuzzy-PSO ICU Pipeline",
    description="Real-time fuzzy-evolutionary ECG arrhythmia detection and monitoring API",
    version="1.0.0",
)

# Enable CORS for all local dev & file origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    global _BEST_PARTICLE
    _BEST_PARTICLE = _load_best_params()
    logger.info("API ready. Best particle loaded.")


# Mount static assets if frontend directory exists
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Pipeline Builder
# ---------------------------------------------------------------------------

def _build_pipeline(particle: np.ndarray) -> dict:
    params = decode_particle(particle)
    return {
        "baseline": BaselineRemover(cfg=params["baseline"]),
        "nlms": NLMSFilter(cfg=params["nlms"]),
        "rpeak": RPeakDetector(cfg=params["rpeak"]),
        "fiducial": FiducialExtractor(),
        "fis": SugenoFIS(mf_params=params["mf_params"], rule_weights=params["rule_weights"]),
        "gate": SafetyGate(tau_lethal=params["tau_lethal"], tau_artifact=params["tau_artifact"]),
    }


# ---------------------------------------------------------------------------
# Frontend Serving Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["frontend"])
async def serve_frontend() -> HTMLResponse:
    if FRONTEND_HTML.exists():
        with open(FRONTEND_HTML, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    elif ROOT_FRONTEND_HTML.exists():
        with open(ROOT_FRONTEND_HTML, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>ICU Arrhythmia Pipeline Backend Online</h1><p>Frontend file not found.</p>")


@app.get("/FRONTEND.html", response_class=HTMLResponse, tags=["frontend"])
async def serve_root_frontend() -> HTMLResponse:
    target = ROOT_FRONTEND_HTML if ROOT_FRONTEND_HTML.exists() else FRONTEND_HTML
    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>File not found</h1>", status_code=404)


# ---------------------------------------------------------------------------
# POST /classify
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    samples: List[float]
    fs: int = SAMPLE_RATE
    scn: Optional[str] = "normal"
    use_full_pipeline: bool = True


class ClassifyResponse(BaseModel):
    class_name: str
    class_index: int
    lethal_score: float
    artifact_score: float
    alert: bool
    hold: bool
    latency_ms: float
    hr: Optional[float] = None
    sqi: float = 1.0
    cv: Optional[float] = None
    r_peaks: List[int] = []
    clean_samples: List[float] = []
    memberships: Dict[str, float] = {}
    banner_text: str = ""
    status_class: str = "normal"


@app.post("/classify", response_model=ClassifyResponse, tags=["inference"])
async def classify(req: ClassifyRequest) -> ClassifyResponse:
    """Classify an ECG segment with full diagnostic and fuzzy membership breakdown."""
    t0 = time.perf_counter()
    signal = np.array(req.samples, dtype=np.float32)
    fs = req.fs if req.fs > 0 else SAMPLE_RATE

    if len(signal) == 0:
        return ClassifyResponse(
            class_name="Normal",
            class_index=0,
            lethal_score=0.0,
            artifact_score=0.0,
            alert=False,
            hold=False,
            latency_ms=0.0,
            banner_text="Analyzing the signal",
        )

    # 1. Run pipeline
    pipe = _build_pipeline(_BEST_PARTICLE)
    clean = pipe["baseline"].remove(signal)
    ref_art = signal - clean
    pipe["nlms"].reset()
    processed = pipe["nlms"].process_block(clean, ref_art)

    det_r = pipe["rpeak"].detect(processed)
    r_amps = extract_r_amplitudes(processed, det_r)
    sqi_val = compute_sqi(processed, fs, r_amps)
    feat = pipe["fiducial"].extract(processed, det_r, sqi=sqi_val)
    fis_out = pipe["fis"].infer(feat.to_array())
    decision = pipe["gate"].evaluate(fis_out.lethal_score, fis_out.artifact_score)

    hr_val = float(feat.hr_bpm) if feat.hr_bpm > 0 else None
    cv_val = float(feat.rr_cv)

    # Compute fuzzy breakdown matching frontend expectations
    st = fuzzy_diagnostics(hr_val, cv_val, sqi_val)
    m = st["m"]

    # Banner determination
    is_hold = bool(decision.suppressed or st["hold"])
    if decision.alert:
        scn_str = req.scn or ("vf" if fis_out.class_name == "VF" else "vt")
        banner_text = "LETHAL ARRHYTHMIA ALERT: " + (
            "ventricular fibrillation pattern" if scn_str == "vf" or fis_out.class_name == "VF"
            else "sustained fast regular rhythm"
        )
        status_class = "banner alarm"
    elif hr_val is None:
        banner_text = "Analyzing the signal"
        status_class = "banner"
    elif is_hold:
        banner_text = "Signal check: interference detected, no lethal pattern confirmed"
        status_class = "banner hold"
    else:
        banner_text = "Rhythm within normal limits"
        status_class = "banner"

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return ClassifyResponse(
        class_name=fis_out.class_name,
        class_index=fis_out.class_index,
        lethal_score=float(fis_out.lethal_score),
        artifact_score=float(fis_out.artifact_score),
        alert=bool(decision.alert),
        hold=is_hold,
        latency_ms=latency_ms,
        hr=hr_val,
        sqi=float(sqi_val),
        cv=cv_val,
        r_peaks=[int(p) for p in det_r],
        clean_samples=processed.tolist(),
        memberships={
            "hrHigh": float(m["hrHigh"]),
            "hrVery": float(m["hrVery"]),
            "regular": float(m["regular"]),
            "irregular": float(m["irregular"]),
            "good": float(m["good"]),
            "poor": float(m["poor"]),
            "vt": float(st["vt"]),
            "vf": float(st["vf"]),
            "noise": float(st["noise"]),
            "conf": float(max(fis_out.lethal_score, st["conf"])),
        },
        banner_text=banner_text,
        status_class=status_class,
    )


# ---------------------------------------------------------------------------
# WS /stream
# ---------------------------------------------------------------------------

@app.websocket("/stream")
async def stream(ws: WebSocket) -> None:
    """
    Streaming WebSocket endpoint.
    Client pushes: {"samples": [...], "fs": 250, "scn": "normal"|"vt"|"vf"}
    Server responds with per-hop alerts, fuzzy memberships, and clean traces.
    """
    await ws.accept()
    pipe = _build_pipeline(_BEST_PARTICLE)
    ring_buf: List[float] = []
    hop_n = 0
    win_s = DEFAULT_CONFIG.stream.window_samples
    hop_s = DEFAULT_CONFIG.stream.hop_samples

    try:
        while True:
            data = await ws.receive_json()
            chunk = data.get("samples", [])
            fs = data.get("fs", SAMPLE_RATE)
            scn = data.get("scn", "normal")
            ring_buf.extend(chunk)

            while len(ring_buf) >= win_s:
                window = np.array(ring_buf[:win_s], dtype=np.float32)
                ring_buf = ring_buf[hop_s:]

                clean = pipe["baseline"].remove(window)
                ref_art = window - clean
                pipe["nlms"].reset()
                processed = pipe["nlms"].process_block(clean, ref_art)

                det_r = pipe["rpeak"].detect(processed)
                r_amps = extract_r_amplitudes(processed, det_r)
                sqi_val = compute_sqi(processed, fs, r_amps)
                feat = pipe["fiducial"].extract(processed, det_r, sqi=sqi_val)
                fis_out = pipe["fis"].infer(feat.to_array())
                decision = pipe["gate"].evaluate(fis_out.lethal_score, fis_out.artifact_score)

                hr_val = float(feat.hr_bpm) if feat.hr_bpm > 0 else None
                cv_val = float(feat.rr_cv)
                st = fuzzy_diagnostics(hr_val, cv_val, sqi_val)
                m = st["m"]

                is_hold = bool(decision.suppressed or st["hold"])
                if decision.alert:
                    banner_text = "LETHAL ARRHYTHMIA ALERT: " + (
                        "ventricular fibrillation pattern" if scn == "vf" or fis_out.class_name == "VF"
                        else "sustained fast regular rhythm"
                    )
                    status_class = "banner alarm"
                elif hr_val is None:
                    banner_text = "Analyzing the signal"
                    status_class = "banner"
                elif is_hold:
                    banner_text = "Signal check: interference detected, no lethal pattern confirmed"
                    status_class = "banner hold"
                else:
                    banner_text = "Rhythm within normal limits"
                    status_class = "banner"

                await ws.send_json({
                    "hop": hop_n,
                    "alert": bool(decision.alert),
                    "hold": is_hold,
                    "class": fis_out.class_name,
                    "lethal_score": float(fis_out.lethal_score),
                    "artifact_score": float(fis_out.artifact_score),
                    "conf": float(max(fis_out.lethal_score, st["conf"])),
                    "hr": hr_val,
                    "sqi": float(sqi_val),
                    "cv": cv_val,
                    "r_peaks": [int(p) for p in det_r],
                    "clean_samples": processed.tolist()[-250:],  # last 1s for lightweight streaming
                    "banner_text": banner_text,
                    "status_class": status_class,
                    "memberships": {
                        "hrHigh": float(m["hrHigh"]),
                        "hrVery": float(m["hrVery"]),
                        "regular": float(m["regular"]),
                        "irregular": float(m["irregular"]),
                        "good": float(m["good"]),
                        "poor": float(m["poor"]),
                        "vt": float(st["vt"]),
                        "vf": float(st["vf"]),
                        "noise": float(st["noise"]),
                        "conf": float(max(fis_out.lethal_score, st["conf"])),
                    },
                })
                hop_n += 1
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected after %d hops", hop_n)


# ---------------------------------------------------------------------------
# POST /analyze (for Repo Pipeline & Upload section)
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    samples: List[float]
    fs: int = 300


@app.post("/analyze", tags=["analysis"])
async def analyze_ecg(req: AnalyzeRequest) -> JSONResponse:
    """Analyze an uploaded ECG record or synthetic sample (RR intervals, HRV, condition & advice)."""
    if not req.samples:
        return JSONResponse({"error": "No samples provided"}, status_code=400)
    x = np.array(req.samples, dtype=np.float32)
    res = analyze_ecg_record(x, fs=req.fs)
    return JSONResponse(res)


# ---------------------------------------------------------------------------
# GET /tests & POST /tests (Automated test suite)
# ---------------------------------------------------------------------------

@app.get("/tests", tags=["evaluation"])
@app.post("/tests", tags=["evaluation"])
async def run_tests_endpoint() -> JSONResponse:
    """Run the 9-check automated verification suite directly on the backend."""
    res = run_automated_tests()
    return JSONResponse(res)


# ---------------------------------------------------------------------------
# POST /simulate (ECG Generator)
# ---------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    scn: str = "normal"
    wander: bool = False
    motion: bool = False
    emg: bool = False
    duration_sec: float = 4.0
    fs: int = SIM_FS
    seed: int = 7


@app.post("/simulate", tags=["simulation"])
async def simulate_ecg(req: SimulateRequest) -> JSONResponse:
    """Generate synthetic ECG samples for testing."""
    sim = Sim(req.seed)
    n = int(req.duration_sec * req.fs)
    o = {"scn": req.scn, "wander": req.wander, "motion": req.motion, "emg": req.emg}
    samples = [sim.next(o) for _ in range(n)]
    return JSONResponse({
        "samples": samples,
        "fs": req.fs,
        "scn": req.scn,
        "truth_r_peaks": sim.rs,
    })


# ---------------------------------------------------------------------------
# GET /params & POST /params
# ---------------------------------------------------------------------------

class ParamsRequest(BaseModel):
    bw: Optional[int] = None
    sm: Optional[int] = None
    th: Optional[float] = None
    mw: Optional[int] = None
    particle: Optional[List[float]] = None


@app.get("/params", tags=["configuration"])
async def get_params() -> JSONResponse:
    """Get active detector & FIS parameters."""
    return JSONResponse({
        "active_detector": _ACTIVE_PARAMS,
        "particle_dim": len(_BEST_PARTICLE),
        "bounds": {
            "bw": [50, 500],
            "sm": [1, 15],
            "th": [0.05, 0.9],
            "mw": [8, 50],
        },
    })


@app.post("/params", tags=["configuration"])
async def update_params(req: ParamsRequest) -> JSONResponse:
    """Update active detector or PSO particle parameters."""
    global _ACTIVE_PARAMS, _BEST_PARTICLE
    if req.bw is not None:
        _ACTIVE_PARAMS["bw"] = int(req.bw)
    if req.sm is not None:
        _ACTIVE_PARAMS["sm"] = int(req.sm)
    if req.th is not None:
        _ACTIVE_PARAMS["th"] = float(req.th)
    if req.mw is not None:
        _ACTIVE_PARAMS["mw"] = int(req.mw)
    if req.particle is not None:
        _BEST_PARTICLE = np.array(req.particle, dtype=np.float32)

    return JSONResponse({
        "status": "updated",
        "active_detector": _ACTIVE_PARAMS,
    })


# ---------------------------------------------------------------------------
# POST /optimize
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    round: int = 1
    iterations: int = 20
    seed: int = DEFAULT_CONFIG.seed


@app.post("/optimize", tags=["optimization"])
async def start_optimize(
    req: OptimizeRequest,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """Start PSO optimization in the background. Returns job_id."""
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "status": "running",
        "round": req.round,
        "iterations": req.iterations,
        "convergence": [],
        "best_params": _ACTIVE_PARAMS,
    }
    background_tasks.add_task(_run_pso_job, job_id, req)
    return JSONResponse({"job_id": job_id, "status": "running"})


async def _run_pso_job(job_id: str, req: OptimizeRequest) -> None:
    """Background optimization task."""
    try:
        from biosig_fuzzy_pso.api.sim_engine import (
            BOUNDS,
            _record_sim,
            _score_record,
            create_rng,
        )

        train_records = [
            _record_sim(11, 10.0, {"scn": "normal", "wander": True}),
            _record_sim(12, 10.0, {"scn": "normal", "emg": True}),
            _record_sim(13, 10.0, {"scn": "normal", "wander": True, "motion": True, "emg": True}),
        ]

        def fitness_fn(cand: Dict[str, Any]) -> float:
            s_val = 0.0
            j_val = 0.0
            for rec in train_records:
                scr = _score_record(rec, cand)
                denom = scr["se"] + scr["ppv"]
                s_val += 2.0 * scr["se"] * scr["ppv"] / (denom if denom > 0 else 1.0)
                j_val += scr["jit"]
            return float(s_val / 3.0 - 0.005 * j_val / 3.0)

        # Evolutionary swarm
        R = create_rng(req.seed)
        pop: List[Dict[str, Any]] = []
        for _ in range(14):
            cand: Dict[str, Any] = {}
            for k, (lo, hi) in BOUNDS.items():
                cand[k] = lo + R() * (hi - lo)
            cand["bw"] = round(cand["bw"])
            cand["sm"] = round(cand["sm"])
            cand["mw"] = round(cand["mw"])
            cand["f"] = fitness_fn(cand)
            pop.append(cand)

        pop.sort(key=lambda x: x["f"], reverse=True)
        best = dict(pop[0])
        conv_log = [{
            "iter": 0,
            "gbest": float(best["f"]),
            "mean": float(sum(x["f"] for x in pop) / len(pop)),
            "params": {k: best[k] for k in ["bw", "sm", "th", "mw"]},
        }]
        _JOBS[job_id]["convergence"] = conv_log

        for g in range(1, req.iterations + 1):
            nx = [dict(pop[0])]

            def pick():
                a = pop[int(R() * len(pop))]
                b = pop[int(R() * len(pop))]
                return a if a["f"] > b["f"] else b

            while len(nx) < 14:
                a = pick()
                b = pick()
                c: Dict[str, Any] = {}
                for k, (lo, hi) in BOUNDS.items():
                    w = R()
                    v = w * a[k] + (1.0 - w) * b[k]
                    if R() < 0.3:
                        v += (R() - 0.5) * 0.3 * (hi - lo)
                    c[k] = min(hi, max(lo, v))
                c["bw"] = round(c["bw"])
                c["sm"] = round(c["sm"])
                c["mw"] = round(c["mw"])
                c["f"] = fitness_fn(c)
                nx.append(c)

            pop = nx
            pop.sort(key=lambda x: x["f"], reverse=True)
            if pop[0]["f"] > best["f"]:
                best = dict(pop[0])

            conv_log.append({
                "iter": g,
                "gbest": float(best["f"]),
                "mean": float(sum(x["f"] for x in pop) / len(pop)),
                "params": {k: best[k] for k in ["bw", "sm", "th", "mw"]},
            })
            _JOBS[job_id]["convergence"] = conv_log
            _JOBS[job_id]["best_params"] = {k: best[k] for k in ["bw", "sm", "th", "mw"]}
            await asyncio.sleep(0.05)

        global _ACTIVE_PARAMS
        _ACTIVE_PARAMS = {k: best[k] for k in ["bw", "sm", "th", "mw"]}
        _JOBS[job_id]["status"] = "done"
        _JOBS[job_id]["gbest_fitness"] = float(best["f"])

    except Exception as e:
        logger.exception("PSO background task failed")
        _JOBS[job_id]["status"] = f"error: {e}"


# ---------------------------------------------------------------------------
# GET /optimize/{id}
# ---------------------------------------------------------------------------

@app.get("/optimize/{job_id}", tags=["optimization"])
async def get_optimize_status(job_id: str) -> JSONResponse:
    if job_id not in _JOBS:
        return JSONResponse({"error": "job not found"}, status_code=404)
    job = _JOBS[job_id]
    return JSONResponse({
        "job_id": job_id,
        "status": job["status"],
        "convergence": job.get("convergence", []),
        "best_params": job.get("best_params", _ACTIVE_PARAMS),
        "gbest_fitness": job.get("gbest_fitness"),
    })


# ---------------------------------------------------------------------------
# GET /metrics/{id}
# ---------------------------------------------------------------------------

@app.get("/metrics/{job_id}", tags=["evaluation"])
async def get_metrics(job_id: str) -> JSONResponse:
    if job_id not in _JOBS:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return JSONResponse(_JOBS[job_id].get("metrics", {}))


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["utility"])
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "service": "biosig_fuzzy_pso",
        "version": "1.0.0",
        "particle_dim": len(_BEST_PARTICLE),
        "active_detector": _ACTIVE_PARAMS,
    })


# ---------------------------------------------------------------------------
# Deliverable 2: Frontend Adapter Contract Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/dataset", response_model=Dataset, tags=["adapter"])
async def get_adapter_dataset() -> Any:
    """GET /api/v1/dataset — Return full Dataset adhering to Frontend Adapter Contract."""
    dataset_file = WORKSPACE_DIR / "frontend" / "src" / "data" / "dataset.json"
    if dataset_file.exists():
        with open(dataset_file, "r", encoding="utf-8") as f:
            return json.load(f)
    from export_frontend_data import run_export
    ds = run_export()
    return ds.model_dump(by_alias=True)


@app.post("/api/v1/signal/stress", response_model=SignalPayload, tags=["adapter"])
async def post_signal_stress(req: SignalStressRequest) -> Any:
    """
    POST /api/v1/signal/stress — Generate and process stress signal payload.
    Latency bound: p95 < 300 ms for 10 s.
    """
    from export_frontend_data import generate_synthetic_ecg, process_signal
    raw, clean, ref_pks = generate_synthetic_ecg(
        duration_s=10.0,
        fs=250,
        rhythm="NSR",
        wander_amp_mv=float(req.wander_amp_mv),
        emg_snr_db=float(req.emg_snr_db),
        mains=bool(req.mains),
        seed=int(req.seed),
    )
    payload = process_signal(raw, ref_pks, fs=250)
    return payload.model_dump(by_alias=True)


@app.get("/api/v1/health", tags=["adapter"])
async def api_v1_health() -> JSONResponse:
    """GET /api/v1/health — Adapter health check."""
    return JSONResponse({
        "status": "healthy",
        "service": "biosig_fuzzy_pso_adapter",
        "version": "1.0",
        "algorithm": "PSO",
    })

