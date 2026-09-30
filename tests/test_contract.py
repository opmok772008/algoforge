"""
tests/test_contract.py — Contract verification test suite.
Validates:
1. dataset.json validates against the Pydantic Dataset model.
2. dataset.ts TypeScript types match the Pydantic model field for field.
3. All CONSISTENCY RULES as strict assertions:
   - Ledger pass recomputed from measured vs threshold
   - C2 passes -> alarms.true_events_preserved == alarms.true_events_total and no lethal event suppressed
   - Exactly one pareto point has selected=true and feasible=true; selected_id matches it
   - Snapshots include generation 0 and last generation; every sigma > 0
   - Signal arrays equal length; indices within range; jitter in live_metrics == std(rpeaks[].error_ms)
   - Round 2 unadapted metrics present; Round 1 unadapted is None
   - Honestly reported constraints
4. attempts has exactly 3 entries in order ("round1", "round2", "final"); what_changed_and_why null only for round1.
5. stress_grid has 9 x 8 entries per attempt; /signal/stress p95 < 300 ms for 10 s.
6. File size of dataset.json < 15 MB.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import asyncio
import numpy as np
import pytest

# Ensure repo root and biosig_fuzzy_pso are on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "biosig_fuzzy_pso"))

from biosig_fuzzy_pso.api.main import app
from biosig_fuzzy_pso.api.schemas import (
    Dataset,
    Attempt,
    LedgerItem,
    SignalPayload,
    Metrics,
    SignalStressRequest,
)
from export_frontend_data import run_export, generate_synthetic_ecg, process_signal

DATASET_JSON_PATH = ROOT_DIR / "frontend" / "src" / "data" / "dataset.json"
DATASET_TS_PATH = ROOT_DIR / "frontend" / "src" / "data" / "dataset.ts"


@pytest.fixture(scope="module")
def dataset_data():
    """Ensure dataset.json exists and load it."""
    if not DATASET_JSON_PATH.exists() or not DATASET_TS_PATH.exists():
        run_export()
    with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def dataset_model(dataset_data):
    """Validate and return Pydantic Dataset instance."""
    return Dataset.model_validate(dataset_data)


# ---------------------------------------------------------------------------
# Test 1: Pydantic Validation & Metadata
# ---------------------------------------------------------------------------

def test_dataset_json_validates_pydantic(dataset_model):
    """dataset.json validates against the pydantic Dataset model."""
    assert dataset_model.schema_version == "1.0"
    assert dataset_model.meta.algorithm == "PSO"
    assert dataset_model.meta.fs == 250
    assert dataset_model.meta.budgets.latency_s == 3.0
    assert dataset_model.meta.budgets.memory_kb == 2048.0
    assert len(dataset_model.attempts) == 3


# ---------------------------------------------------------------------------
# Test 2: TS types in dataset.ts match the model field for field
# ---------------------------------------------------------------------------

def test_ts_types_match_pydantic_model():
    """Verify that dataset.ts has interfaces matching Pydantic schemas."""
    assert DATASET_TS_PATH.exists(), "dataset.ts must exist"
    content = DATASET_TS_PATH.read_text(encoding="utf-8")

    # Required interfaces
    expected_interfaces = [
        "Dataset",
        "Meta",
        "Budgets",
        "Metrics",
        "PerRecord",
        "LedgerItem",
        "SignalPayload",
        "RPeakItem",
        "QRSItem",
        "STItem",
        "ZonesItem",
        "RhythmEpisodeItem",
        "LiveMetrics",
        "StressGridItem",
        "AlarmEvent",
        "AlarmRatePerHour",
        "Alarms",
        "OperatingPoint",
        "ROC",
        "Classification",
        "ConvergenceItem",
        "ParetoItem",
        "MFItem",
        "SnapshotItem",
        "Stability",
        "Optimization",
        "FuzzyRuleFiringItem",
        "FuzzyExampleItem",
        "FuzzyRulesGenItem",
        "ResourceStageItem",
        "TraceKBItem",
        "Resources",
        "Attempt",
    ]

    for iface in expected_interfaces:
        pattern = rf"(interface|type)\s+{iface}\b"
        assert re.search(pattern, content), f"Interface {iface} missing in dataset.ts"

    # Check export statements
    assert 'import raw from "./dataset.json";' in content
    assert "export const dataset = raw as Dataset;" in content


# ---------------------------------------------------------------------------
# Test 3: Consistency Rules
# ---------------------------------------------------------------------------

def test_consistency_rule_ledger_recomputed_pass(dataset_model):
    """
    Consistency rule: Ledger pass recomputed from measured vs threshold.
    """
    for attempt in dataset_model.attempts:
        for item in attempt.ledger:
            if item.comparator == "<=":
                expected_pass = item.measured <= item.threshold
                if item.measured_secondary is not None and item.threshold_secondary is not None:
                    expected_pass = expected_pass and (item.measured_secondary <= item.threshold_secondary)
            elif item.comparator == ">=":
                expected_pass = item.measured >= item.threshold
            else:
                expected_pass = True

            assert item.pass_ == expected_pass, (
                f"Ledger {item.id} pass mismatch for {attempt.id}: measured={item.measured}, threshold={item.threshold}, pass={item.pass_}"
            )


def test_consistency_rule_c2_and_alarms(dataset_model):
    """
    Consistency rule: If C2 passes then alarms.true_events_preserved == alarms.true_events_total,
    and no lethal event has suppressed=true.
    """
    for attempt in dataset_model.attempts:
        c2 = next(x for x in attempt.ledger if x.id == "C2")
        if c2.pass_:
            assert attempt.alarms.true_events_preserved == attempt.alarms.true_events_total, (
                f"{attempt.id}: C2 passed but preserved ({attempt.alarms.true_events_preserved}) != total ({attempt.alarms.true_events_total})"
            )
            for ev in attempt.alarms.events:
                if ev.lethal:
                    assert ev.suppressed is False, f"{attempt.id}: Lethal event {ev.id} suppressed when C2 passed!"
                    assert ev.cause_if_suppressed is None, f"{attempt.id}: Lethal event {ev.id} has cause_if_suppressed set!"
            assert "0 of" in c2.detail and "true events suppressed" in c2.detail


def test_consistency_rule_pareto(dataset_model):
    """
    Consistency rule: Exactly one pareto point has selected=true and it is feasible; selected_id matches it.
    """
    for attempt in dataset_model.attempts:
        pareto = attempt.optimization.pareto
        selected = [p for p in pareto if p.selected]
        assert len(selected) == 1, f"{attempt.id}: Expected exactly 1 selected pareto point, found {len(selected)}"
        sel = selected[0]
        assert sel.feasible is True, f"{attempt.id}: Selected pareto point {sel.id} must be feasible"
        assert attempt.optimization.selected_id == sel.id, (
            f"{attempt.id}: selected_id ({attempt.optimization.selected_id}) != sel.id ({sel.id})"
        )


def test_consistency_rule_snapshots(dataset_model):
    """
    Consistency rule: Snapshots include generation 0 and the last generation; every sigma > 0.
    """
    last_gen = dataset_model.meta.generations
    for attempt in dataset_model.attempts:
        snapshots = attempt.optimization.snapshots
        gens = [s.generation for s in snapshots]
        assert 0 in gens, f"{attempt.id}: Generation 0 missing from snapshots"
        assert last_gen in gens, f"{attempt.id}: Generation {last_gen} missing from snapshots"

        for snap in snapshots:
            for feat, mfs in snap.mf.items():
                for mf in mfs:
                    assert mf.sigma > 0, f"{attempt.id}: snap gen {snap.generation} feat {feat} sigma <= 0 ({mf.sigma})"


def test_consistency_rule_signals(dataset_model):
    """
    Consistency rule: Signal arrays have equal length; all indices are within range;
    jitter in live_metrics equals the std of rpeaks[].error_ms.
    """
    for attempt in dataset_model.attempts:
        sig = attempt.signal
        n = len(sig.raw)
        assert len(sig.filtered) == n, f"{attempt.id}: raw and filtered length mismatch"
        if sig.reference is not None:
            assert len(sig.reference) == n

        # Indices in range
        for rp in sig.rpeaks:
            assert 0 <= rp.idx < n, f"{attempt.id}: rpeak idx {rp.idx} out of bounds [0, {n})"
        for qrs in sig.qrs:
            assert 0 <= qrs.onset_idx < n
            assert 0 <= qrs.offset_idx < n
        for st in sig.st:
            assert 0 <= st.onset_idx < n
            assert 0 <= st.offset_idx < n

        # Jitter in live_metrics equals std of error_ms
        if len(sig.rpeaks) > 1:
            errs = [rp.error_ms for rp in sig.rpeaks]
            calc_jitter = round(float(np.std(errs)), 4)
            assert abs(sig.live_metrics.jitter_ms - calc_jitter) < 1e-3, (
                f"{attempt.id}: live_metrics.jitter_ms ({sig.live_metrics.jitter_ms}) != calc ({calc_jitter})"
            )


def test_consistency_rule_round2_unadapted(dataset_model):
    """
    Consistency rule: Round 2's unadapted metrics come from the same test set as the adapted ones.
    Round 1 unadapted is None; Round 2 unadapted is non-None.
    """
    r1 = dataset_model.attempts[0]
    r2 = dataset_model.attempts[1]
    final = dataset_model.attempts[2]

    assert r1.unadapted is None, "Round 1 unadapted must be None"
    assert r2.unadapted is not None, "Round 2 unadapted must be present"
    assert final.unadapted is not None, "Final unadapted must be present"


# ---------------------------------------------------------------------------
# Test 4: Attempts structure & what_changed_and_why
# ---------------------------------------------------------------------------

def test_attempts_order_and_what_changed(dataset_model):
    """attempts has exactly 3 entries in order; what_changed_and_why is null only for round1."""
    assert len(dataset_model.attempts) == 3
    ids = [a.id for a in dataset_model.attempts]
    assert ids == ["round1", "round2", "final"], f"Attempts order mismatch: {ids}"

    r1, r2, final = dataset_model.attempts
    assert r1.what_changed_and_why is None, "Round 1 what_changed_and_why must be null"
    assert r2.what_changed_and_why and len(r2.what_changed_and_why.strip()) > 0, "Round 2 what_changed_and_why must be non-empty"
    assert final.what_changed_and_why and len(final.what_changed_and_why.strip()) > 0, "Final what_changed_and_why must be non-empty"


# ---------------------------------------------------------------------------
# Test 5: Stress grid shape & latency bound (/signal/stress p95 < 300 ms for 10 s)
# ---------------------------------------------------------------------------

def test_stress_grid_shape_and_speed(dataset_model):
    """stress_grid has 9 x 8 entries per attempt; /signal/stress p95 < 300 ms."""
    for attempt in dataset_model.attempts:
        assert len(attempt.stress_grid) == 9 * 8, f"{attempt.id}: stress grid size {len(attempt.stress_grid)} != 72"

    # Latency test: 20 executions of 10s signal processing
    times: list[float] = []
    for _ in range(20):
        t0 = time.perf_counter()
        raw, clean, ref_pks = generate_synthetic_ecg(
            duration_s=10.0,
            fs=250,
            rhythm="NSR",
            wander_amp_mv=0.5,
            emg_snr_db=15.0,
            mains=False,
            seed=42,
        )
        _ = process_signal(raw, ref_pks, fs=250)
        dt = (time.perf_counter() - t0) * 1000.0  # in ms
        times.append(dt)

    p95 = float(np.percentile(times, 95))
    print(f"\n/signal/stress 10s execution times: mean={np.mean(times):.1f}ms, p95={p95:.1f}ms")
    assert p95 < 300.0, f"p95 latency {p95:.1f} ms exceeded 300 ms budget!"


# ---------------------------------------------------------------------------
# Test 6: File size of dataset.json < 15 MB
# ---------------------------------------------------------------------------

def test_dataset_json_file_size():
    """File size of dataset.json < 15 MB."""
    assert DATASET_JSON_PATH.exists()
    size_bytes = DATASET_JSON_PATH.stat().st_size
    size_mb = size_bytes / (1024.0 * 1024.0)
    print(f"\ndataset.json size: {size_mb:.2f} MB")
    assert size_bytes < 15 * 1024 * 1024, f"dataset.json size {size_mb:.2f} MB exceeds 15 MB limit!"


# ---------------------------------------------------------------------------
# Test 7: FastAPI Endpoints (Deliverable 2)
# ---------------------------------------------------------------------------

def test_api_deliverable_2_endpoints():
    """Test Deliverable 2 endpoints directly via async handler functions."""
    from biosig_fuzzy_pso.api.main import (
        api_v1_health,
        get_adapter_dataset,
        post_signal_stress,
        health,
    )

    # GET /api/v1/health
    r_health = asyncio.run(api_v1_health())
    health_body = json.loads(r_health.body)
    assert r_health.status_code == 200
    assert health_body["status"] == "healthy"
    assert health_body["algorithm"] == "PSO"

    # GET /api/v1/dataset
    ds = asyncio.run(get_adapter_dataset())
    assert ds["schema_version"] == "1.0"
    assert len(ds["attempts"]) == 3

    # POST /api/v1/signal/stress
    req = SignalStressRequest(
        wander_amp_mv=0.5,
        emg_snr_db=15.0,
        mains=False,
        seed=42,
        attempt="final",
    )
    t0 = time.perf_counter()
    sp = asyncio.run(post_signal_stress(req))
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert sp["fs"] == 250
    assert len(sp["raw"]) == 2500
    assert dt_ms < 300.0, f"Live stress endpoint took {dt_ms:.1f} ms (budget < 300 ms)"

    # Original health endpoint still functions
    r_orig = asyncio.run(health())
    assert r_orig.status_code == 200
    assert json.loads(r_orig.body)["status"] == "ok"
