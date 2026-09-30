"""
pso/fitness.py — Multi-objective scalarized fitness function.

Fitness formula (minimize):
  F = a1*(1 - Se) + a2*(1 - Sp) + a3*FAFI + a4*(J / J_max)
    + a5*(1 - dSNR / dSNR_max) + a6*(t_run / t_budget) + P

where P = sum of hard-constraint penalties (1e3 * violation magnitude):
  C1: latency violation (worst_latency > 3.0 s)
  C2: lethal Se < 1.0
  C3: QRS amplitude distortion > 5%, ST error > 0.02 mV
  C4: peak memory > 2 MB

Feasibility: a particle is 'feasible' only if lethal_Se == 1.0 (C2).
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import List

import numpy as np

from biosig_fuzzy_pso.config import (
    FitnessWeights, DEFAULT_CONFIG, SAMPLE_RATE, StreamConfig
)
from biosig_fuzzy_pso.dsp.baseline import BaselineRemover, measure_qrs_amplitude_distortion, measure_st_error
from biosig_fuzzy_pso.dsp.adaptive import NLMSFilter, FuzzyStepController, compute_snr_improvement
from biosig_fuzzy_pso.dsp.quality import compute_sqi
from biosig_fuzzy_pso.features.rpeak import RPeakDetector, extract_r_amplitudes
from biosig_fuzzy_pso.features.fiducials import FiducialExtractor
from biosig_fuzzy_pso.fuzzy.fis import SugenoFIS
from biosig_fuzzy_pso.fuzzy.safety_gate import SafetyGate
from biosig_fuzzy_pso.pso.encoding import decode_particle


# ---------------------------------------------------------------------------
# Per-window classification result
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    label: int              # ground-truth label (0=Normal, 1=VT, 2=VF)
    predicted: int          # predicted class index
    lethal_score: float
    artifact_score: float
    alert: bool
    latency_s: float        # seconds from window start to decision
    r_jitter_ms: float      # R-peak jitter for this window (ms)
    dsnr: float             # SNR improvement (dB)
    qrs_distortion: float   # C3
    st_error_mv: float      # C3


# ---------------------------------------------------------------------------
# Fitness evaluator
# ---------------------------------------------------------------------------

class FitnessEvaluator:
    """
    Evaluates the PSO fitness for one particle on the training set.

    Parameters
    ----------
    records : list of record dicts (from loader)
    weights : FitnessWeights
    stream  : StreamConfig
    max_windows_per_record : int or None
        Subsample limit for PSO speed. None = use all windows.
        Set to ~60 for PSO iterations; use None for final benchmark.
    """

    def __init__(
        self,
        records: List[dict],
        weights: FitnessWeights = DEFAULT_CONFIG.weights,
        stream: StreamConfig = DEFAULT_CONFIG.stream,
        max_windows_per_record: int = 12,
    ) -> None:
        self.records = records
        self.weights = weights
        self.stream = stream
        self.max_windows_per_record = max_windows_per_record

    # ------------------------------------------------------------------
    def evaluate(self, particle: np.ndarray) -> tuple[float, dict]:
        """
        Evaluate fitness for one particle.

        Returns
        -------
        (fitness, metrics_dict)
        """
        params = decode_particle(particle)

        # Build pipeline components from decoded params
        baseline = BaselineRemover(cfg=params["baseline"])
        nlms = NLMSFilter(cfg=params["nlms"])
        nlms_ctrl = FuzzyStepController(cfg=params["nlms"])
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

        results: List[WindowResult] = []

        tracemalloc.start()
        peak_mem_kb = 0.0

        for record in self.records:
            signal = record["signal"]
            ref_r_peaks = record["r_peaks"]
            true_label = record["label"]
            fs = record.get("fs", SAMPLE_RATE)

            window_s = self.stream.window_samples
            hop_s = self.stream.hop_samples

            n = len(signal)

            # Subsample for PSO speed: evenly sample positions from all windows
            all_positions = list(range(0, n - window_s + 1, hop_s))
            if self.max_windows_per_record and len(all_positions) > self.max_windows_per_record:
                step = max(1, len(all_positions) // self.max_windows_per_record)
                positions = all_positions[::step][: self.max_windows_per_record]
            else:
                positions = all_positions

            for pos in positions:
                win = signal[pos: pos + window_s].copy()
                t_start = time.perf_counter()

                # DSP
                clean = baseline.remove(win)
                # No reference channel in single-lead — use high-pass residual as reference
                ref_artifact = win - clean
                nlms.reset()
                processed = nlms.process_block(clean, ref_artifact)

                # R-peaks
                det_r = rpeak_det.detect(processed)

                # SQI
                r_amps = extract_r_amplitudes(processed, det_r)
                sqi_val = compute_sqi(processed, fs, r_amps)

                # Features
                feat = fiducial.extract(processed, det_r, sqi=sqi_val)

                # FIS
                fis_out = fis.infer(feat.to_array())
                decision = gate.evaluate(fis_out.lethal_score, fis_out.artifact_score)

                t_elapsed = time.perf_counter() - t_start
                latency_s = self.stream.window_sec + t_elapsed  # C1 measure

                # R-peak jitter vs reference peaks in this window
                ref_in_window = ref_r_peaks[
                    (ref_r_peaks >= pos) & (ref_r_peaks < pos + window_s)
                ] - pos
                jitter_ms = _compute_jitter(det_r, ref_in_window, fs)

                # C3 measures
                qrs_dist = measure_qrs_amplitude_distortion(win, processed, ref_in_window)
                st_err = measure_st_error(win, processed, ref_in_window)

                # SNR
                dsnr = compute_snr_improvement(clean, win, processed)

                results.append(WindowResult(
                    label=true_label,
                    predicted=fis_out.class_index,
                    lethal_score=fis_out.lethal_score,
                    artifact_score=fis_out.artifact_score,
                    alert=decision.alert,
                    latency_s=latency_s,
                    r_jitter_ms=jitter_ms,
                    dsnr=dsnr,
                    qrs_distortion=qrs_dist,
                    st_error_mv=st_err,
                ))

        _, peak = tracemalloc.get_traced_memory()
        peak_mem_kb = peak / 1024.0
        tracemalloc.stop()

        return self._compute_fitness(results, peak_mem_kb)

    # ------------------------------------------------------------------
    def _compute_fitness(
        self, results: List[WindowResult], peak_mem_kb: float
    ) -> tuple[float, dict]:
        """Compute scalarized fitness and metrics dict from window results."""
        if not results:
            return 1e6, {}

        w = self.weights
        labels = np.array([r.label for r in results])
        preds = np.array([r.predicted for r in results])
        alerts = np.array([r.alert for r in results])
        latencies = np.array([r.latency_s for r in results])
        jitters = np.array([r.r_jitter_ms for r in results])
        dsnrs = np.array([r.dsnr for r in results])
        qrs_dists = np.array([r.qrs_distortion for r in results])
        st_errs = np.array([r.st_error_mv for r in results])
        times = np.array([r.latency_s - self.stream.window_sec for r in results])

        # --- Classification metrics ---
        Se, Sp = _macro_se_sp(labels, preds)
        lethal_Se = _lethal_sensitivity(labels, alerts)

        # --- FAFI ---
        total_hrs = sum(
            len(rec["signal"]) / rec.get("fs", SAMPLE_RATE) / 3600.0
            for rec in self.records
        )
        false_alarms = int(np.sum(alerts & (labels == 0)))
        fahr = false_alarms / max(total_hrs, 1e-6)
        FAFI = float(np.clip(fahr / w.FA_ref_per_hr, 0.0, 1.0))

        # --- Jitter ---
        J = float(np.mean(jitters))

        # --- dSNR ---
        dSNR = float(np.mean(dsnrs))

        # --- Compute time ---
        t_run = float(np.mean(times)) * 1000.0  # ms

        # --- Penalties ---
        P = 0.0

        # C1: latency
        worst_latency = float(np.max(latencies))
        if worst_latency > self.stream.max_latency_sec:
            P += w.penalty_coeff * (worst_latency - self.stream.max_latency_sec)

        # C2: lethal sensitivity MUST be 1.0
        if lethal_Se < 1.0:
            P += w.penalty_coeff * (1.0 - lethal_Se)

        # C3: QRS amplitude distortion
        mean_qrs_dist = float(np.mean(qrs_dists))
        if mean_qrs_dist > 0.05:
            P += w.penalty_coeff * (mean_qrs_dist - 0.05)

        # C3: ST error
        mean_st_err = float(np.mean(st_errs))
        if mean_st_err > 0.02:
            P += w.penalty_coeff * (mean_st_err - 0.02)

        # C4: Memory
        if peak_mem_kb > DEFAULT_CONFIG.evl.max_memory_kb:
            P += w.penalty_coeff * (peak_mem_kb - DEFAULT_CONFIG.evl.max_memory_kb) / 1024.0

        # --- Fitness ---
        # F = a1*(1-Se) + a2*(1-Sp) + a3*FAFI + a4*(J/J_max)
        #   + a5*(1-dSNR/dSNR_max) + a6*(t_run/t_budget) + P
        F = (
            w.a1 * (1.0 - Se)
            + w.a2 * (1.0 - Sp)
            + w.a3 * FAFI
            + w.a4 * (J / max(w.J_max_ms, 1.0))
            + w.a5 * (1.0 - min(dSNR, w.dSNR_max_db) / w.dSNR_max_db)
            + w.a6 * (t_run / w.t_budget_ms)
            + P
        )

        metrics = {
            "Se": Se, "Sp": Sp, "lethal_Se": lethal_Se,
            "FAFI": FAFI, "J_ms": J, "dSNR_db": dSNR,
            "t_run_ms": t_run, "P": P, "fitness": F,
            "worst_latency_s": worst_latency,
            "peak_mem_kb": peak_mem_kb,
            "mean_qrs_distortion": mean_qrs_dist,
            "mean_st_error_mv": mean_st_err,
            "false_alarms_per_hr": fahr,
        }
        return F, metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _macro_se_sp(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    """Macro-averaged sensitivity and specificity across classes 0,1,2."""
    se_list, sp_list = [], []
    for cls in range(3):
        tp = int(np.sum((labels == cls) & (preds == cls)))
        fn = int(np.sum((labels == cls) & (preds != cls)))
        tn = int(np.sum((labels != cls) & (preds != cls)))
        fp = int(np.sum((labels != cls) & (preds == cls)))
        se_list.append(tp / max(tp + fn, 1))
        sp_list.append(tn / max(tn + fp, 1))
    return float(np.mean(se_list)), float(np.mean(sp_list))


def _lethal_sensitivity(labels: np.ndarray, alerts: np.ndarray) -> float:
    """
    Lethal-event sensitivity: fraction of windows with lethal label (1=VT, 2=VF)
    that produced an alert. Must equal 1.0 for C2 compliance.
    """
    lethal_mask = (labels == 1) | (labels == 2)
    n_lethal = int(np.sum(lethal_mask))
    if n_lethal == 0:
        return 1.0  # no lethal events -> trivially satisfied
    detected = int(np.sum(alerts[lethal_mask]))
    return float(detected / n_lethal)


def _compute_jitter(
    detected: np.ndarray, reference: np.ndarray, fs: int
) -> float:
    """
    R-peak jitter: std of (detected_R - nearest_reference_R) in ms.
    Returns 0 if insufficient peaks.
    """
    if len(detected) == 0 or len(reference) == 0:
        return 0.0
    diffs = []
    for d in detected:
        nearest = reference[np.argmin(np.abs(reference - d))]
        diff_ms = abs(int(d) - int(nearest)) / fs * 1000.0
        if diff_ms < 150.0:   # ignore grossly mismatched
            diffs.append(diff_ms)
    return float(np.std(diffs)) if len(diffs) > 1 else 0.0
