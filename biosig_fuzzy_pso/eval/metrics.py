"""
eval/metrics.py — Comprehensive metrics computation for the benchmark.

Computes:
  Se, Sp, PPV, lethal-event Se, alert latency (mean & worst-case),
  false alarms per hour, FAFI, R-peak jitter (ms), SNR improvement (dB),
  QRS amplitude distortion, ST error (mV), ms per window, peak memory (KB).
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from biosig_fuzzy_pso.config import DEFAULT_CONFIG, SAMPLE_RATE, StreamConfig
from biosig_fuzzy_pso.dsp.baseline import BaselineRemover, measure_qrs_amplitude_distortion, measure_st_error
from biosig_fuzzy_pso.dsp.adaptive import NLMSFilter, compute_snr_improvement
from biosig_fuzzy_pso.dsp.quality import compute_sqi
from biosig_fuzzy_pso.features.rpeak import RPeakDetector, extract_r_amplitudes
from biosig_fuzzy_pso.features.fiducials import FiducialExtractor
from biosig_fuzzy_pso.fuzzy.fis import SugenoFIS
from biosig_fuzzy_pso.fuzzy.safety_gate import SafetyGate
from biosig_fuzzy_pso.pso.encoding import decode_particle
from biosig_fuzzy_pso.pso.fitness import _compute_jitter, _lethal_sensitivity


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkMetrics:
    # Classification
    Se: float               # macro-averaged sensitivity
    Sp: float               # macro-averaged specificity
    PPV: float              # macro-averaged positive predictive value
    lethal_Se: float        # C2: lethal sensitivity (must == 1.0)
    # Alarm
    false_alarms_per_hr: float
    FAFI: float
    alert_latency_mean_s: float
    alert_latency_worst_s: float
    # Signal quality
    r_jitter_ms: float
    dSNR_db: float
    qrs_amplitude_distortion: float
    st_error_mv: float
    # Performance
    ms_per_window: float
    peak_memory_kb: float
    # Feasibility flags
    c1_ok: bool     # latency <= 3.0 s
    c2_ok: bool     # lethal_Se == 1.0
    c3_ok: bool     # distortion <= 5%, ST <= 0.02 mV
    c4_ok: bool     # memory <= 2 MB

    def print_table(self) -> None:
        print("\n" + "=" * 60)
        print("BENCHMARK METRICS")
        print("=" * 60)
        print(f"  Se (macro):              {self.Se:.4f}")
        print(f"  Sp (macro):              {self.Sp:.4f}")
        print(f"  PPV (macro):             {self.PPV:.4f}")
        print(f"  Lethal Se (C2):          {self.lethal_Se:.4f}  {'[PASS]' if self.c2_ok else '[FAIL]'}")
        print(f"  FA/hr:                   {self.false_alarms_per_hr:.2f}")
        print(f"  FAFI:                    {self.FAFI:.4f}")
        print(f"  Alert latency (mean):    {self.alert_latency_mean_s*1000:.1f} ms")
        print(f"  Alert latency (worst):   {self.alert_latency_worst_s*1000:.1f} ms  {'[PASS]' if self.c1_ok else '[FAIL]'}")
        print(f"  R-peak jitter:           {self.r_jitter_ms:.2f} ms")
        print(f"  dSNR:                    {self.dSNR_db:.2f} dB")
        print(f"  QRS distortion (C3):     {self.qrs_amplitude_distortion*100:.2f}%  {'[PASS]' if self.c3_ok else '[FAIL]'}")
        print(f"  ST error (C3):           {self.st_error_mv*1000:.2f} uV")
        print(f"  ms/window:               {self.ms_per_window:.2f} ms")
        print(f"  Peak memory (C4):        {self.peak_memory_kb:.1f} KB  {'[PASS]' if self.c4_ok else '[FAIL]'}")
        print("=" * 60)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate_on_records(
    records: List[dict],
    best_particle: np.ndarray,
    stream: StreamConfig = DEFAULT_CONFIG.stream,
    weights=DEFAULT_CONFIG.weights,
) -> BenchmarkMetrics:
    """
    Evaluate a decoded parameter set on a list of records and return
    a BenchmarkMetrics object with all required metrics.
    """
    params = decode_particle(best_particle)
    baseline = BaselineRemover(cfg=params["baseline"])
    nlms = NLMSFilter(cfg=params["nlms"])
    rpeak_det = RPeakDetector(cfg=params["rpeak"])
    fiducial = FiducialExtractor()
    fis = SugenoFIS(
        mf_params=params["mf_params"],
        rule_weights=params["rule_weights"],
    )
    gate = SafetyGate(
        tau_lethal=params["tau_lethal"],
        tau_artifact=params["tau_artifact"],
    )

    labels_all, preds_all, alerts_all = [], [], []
    latencies, jitters, dsnrs = [], [], []
    qrs_dists, st_errs = [], []
    times_ms = []
    peak_mem_kb = 0.0

    tracemalloc.start()

    for record in records:
        signal = record["signal"]
        ref_r_peaks = record["r_peaks"]
        true_label = record["label"]
        fs = record.get("fs", SAMPLE_RATE)

        win_s = stream.window_samples
        hop_s = stream.hop_samples
        n = len(signal)
        pos = 0

        while pos + win_s <= n:
            win = signal[pos: pos + win_s].copy()
            t0 = time.perf_counter()

            clean = baseline.remove(win)
            ref_art = win - clean
            nlms.reset()
            processed = nlms.process_block(clean, ref_art)

            det_r = rpeak_det.detect(processed)
            r_amps = extract_r_amplitudes(processed, det_r)
            sqi_val = compute_sqi(processed, fs, r_amps)
            feat = fiducial.extract(processed, det_r, sqi=sqi_val)
            fis_out = fis.infer(feat.to_array())
            decision = gate.evaluate(fis_out.lethal_score, fis_out.artifact_score)

            elapsed = time.perf_counter() - t0
            latency = stream.window_sec + elapsed

            ref_in_win = ref_r_peaks[
                (ref_r_peaks >= pos) & (ref_r_peaks < pos + win_s)
            ] - pos
            jitter = _compute_jitter(det_r, ref_in_win, fs)
            dsnr = compute_snr_improvement(clean, win, processed)
            qd = measure_qrs_amplitude_distortion(win, processed, ref_in_win)
            se = measure_st_error(win, processed, ref_in_win)

            labels_all.append(true_label)
            preds_all.append(fis_out.class_index)
            alerts_all.append(decision.alert)
            latencies.append(latency)
            jitters.append(jitter)
            dsnrs.append(dsnr)
            qrs_dists.append(qd)
            st_errs.append(se)
            times_ms.append(elapsed * 1000.0)

            _, peak = tracemalloc.get_traced_memory()
            peak_mem_kb = max(peak_mem_kb, peak / 1024.0)

            pos += hop_s

    tracemalloc.stop()

    labels = np.array(labels_all)
    preds = np.array(preds_all)
    alerts = np.array(alerts_all)

    # Classification
    se_list, sp_list, ppv_list = [], [], []
    for cls in range(3):
        tp = int(np.sum((labels == cls) & (preds == cls)))
        fn = int(np.sum((labels == cls) & (preds != cls)))
        tn = int(np.sum((labels != cls) & (preds != cls)))
        fp = int(np.sum((labels != cls) & (preds == cls)))
        se_list.append(tp / max(tp + fn, 1))
        sp_list.append(tn / max(tn + fp, 1))
        ppv_list.append(tp / max(tp + fp, 1))

    Se = float(np.mean(se_list))
    Sp = float(np.mean(sp_list))
    PPV = float(np.mean(ppv_list))
    lethal_Se = _lethal_sensitivity(labels, alerts)

    # Alarms
    total_hrs = sum(
        len(rec["signal"]) / rec.get("fs", SAMPLE_RATE) / 3600.0
        for rec in records
    )
    false_alarms = int(np.sum(alerts & (labels == 0)))
    fahr = false_alarms / max(total_hrs, 1e-6)
    FAFI = float(np.clip(fahr / weights.FA_ref_per_hr, 0.0, 1.0))

    mean_lat = float(np.mean(latencies))
    worst_lat = float(np.max(latencies))

    c1_ok = worst_lat <= DEFAULT_CONFIG.stream.max_latency_sec
    c2_ok = lethal_Se == 1.0
    mean_qd = float(np.mean(qrs_dists))
    mean_se = float(np.mean(st_errs))
    c3_ok = (mean_qd <= 0.05) and (mean_se <= 0.02)
    c4_ok = peak_mem_kb <= DEFAULT_CONFIG.evl.max_memory_kb

    return BenchmarkMetrics(
        Se=Se, Sp=Sp, PPV=PPV, lethal_Se=lethal_Se,
        false_alarms_per_hr=fahr, FAFI=FAFI,
        alert_latency_mean_s=mean_lat, alert_latency_worst_s=worst_lat,
        r_jitter_ms=float(np.mean(jitters)),
        dSNR_db=float(np.mean(dsnrs)),
        qrs_amplitude_distortion=mean_qd,
        st_error_mv=mean_se,
        ms_per_window=float(np.mean(times_ms)),
        peak_memory_kb=peak_mem_kb,
        c1_ok=c1_ok, c2_ok=c2_ok, c3_ok=c3_ok, c4_ok=c4_ok,
    )
