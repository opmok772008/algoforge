"""
data/synthetic.py — Deterministic synthetic ECG/EMG/PPG generator.

Morphology: sum-of-Gaussians for P, Q, R, S, T waves.
Rhythm modes: Normal Sinus, VT (wide-QRS, fast), VF (sinusoid mix).
Perturbation injectors: baseline wander, motion artifact, EMG burst, mains.

All arrays use float32 and are pre-allocated (no growing lists).
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Optional, Tuple

from biosig_fuzzy_pso.config import SyntheticConfig, SAMPLE_RATE, DEFAULT_CONFIG

RhythmMode = Literal["normal", "vt", "vf"]


# ---------------------------------------------------------------------------
# Core morphology helpers
# ---------------------------------------------------------------------------

def _gaussian(t: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    """Single Gaussian component (float32)."""
    return np.float32(amplitude) * np.exp(
        -0.5 * ((t - np.float32(center)) / np.float32(sigma)) ** 2
    ).astype(np.float32)


def _beat_template(
    t_ms: np.ndarray,
    cfg: SyntheticConfig,
    qrs_width_ms: float,
    st_offset_mv: float = 0.0,
) -> np.ndarray:
    """
    Build one beat template (sum of Gaussians) centred at t=0 ms.

    Parameters
    ----------
    t_ms        : time axis in milliseconds, relative to R peak
    cfg         : SyntheticConfig
    qrs_width_ms: QRS width (scales Q/S sigma)
    st_offset_mv: ST-segment elevation/depression in mV
    """
    width_scale = qrs_width_ms / cfg.qrs_normal_ms

    beat = (
        _gaussian(t_ms, *cfg.p_wave)
        + _gaussian(t_ms, cfg.q_wave[0], cfg.q_wave[1], cfg.q_wave[2] * width_scale)
        + _gaussian(t_ms, *cfg.r_wave)
        + _gaussian(t_ms, cfg.s_wave[0], cfg.s_wave[1], cfg.s_wave[2] * width_scale)
        + _gaussian(t_ms, cfg.t_wave[0], cfg.t_wave[1] + st_offset_mv * 20, cfg.t_wave[2])
    )
    # Add ST level by shifting the baseline between J-point and T-wave onset
    j_point_idx = np.searchsorted(t_ms, cfg.r_wave[1] + 40.0)
    t_onset_idx = np.searchsorted(t_ms, cfg.t_wave[1] - cfg.t_wave[2] * 2)
    beat[j_point_idx:t_onset_idx] = (
        beat[j_point_idx:t_onset_idx] + np.float32(st_offset_mv)
    )
    return beat.astype(np.float32)


# ---------------------------------------------------------------------------
# ECG generator
# ---------------------------------------------------------------------------

class SyntheticECG:
    """
    Deterministic synthetic ECG generator using sum-of-Gaussians morphology.

    Usage
    -----
    >>> gen = SyntheticECG(seed=42)
    >>> ecg, ann = gen.generate(duration_sec=10.0, mode="normal")
    """

    def __init__(
        self,
        cfg: SyntheticConfig = DEFAULT_CONFIG.synthetic,
        seed: int = 42,
    ) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def generate(
        self,
        duration_sec: float,
        mode: RhythmMode = "normal",
        st_offset_mv: float = 0.0,
        add_noise: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ECG signal.

        Returns
        -------
        ecg : float32 array of shape (n_samples,)
        r_peaks : int array of R-peak sample indices
        """
        fs = self.cfg.fs
        n = int(duration_sec * fs)
        ecg = np.zeros(n, dtype=np.float32)
        r_peaks = []

        if mode == "vf":
            ecg = self._vf_signal(n)
            r_peaks = np.array([], dtype=np.int32)
        else:
            hr = self.cfg.hr_vt_bpm if mode == "vt" else self.cfg.hr_normal_bpm
            qrs_ms = self.cfg.qrs_vt_ms if mode == "vt" else self.cfg.qrs_normal_ms
            rr_mean = 60.0 / hr  # seconds

            # Pre-build beat template (1-second window)
            t_ms = np.arange(-500, 501, 1.0 / fs * 1000).astype(np.float32)
            template = _beat_template(t_ms, self.cfg, qrs_ms, st_offset_mv)
            half = len(template) // 2

            pos = int(rr_mean * fs * 0.5)  # first beat
            while pos < n:
                # RR variability: ±5 ms jitter for normal, ±2 ms for VT
                jitter_ms = 5.0 if mode == "normal" else 2.0
                jitter = int(self.rng.normal(0, jitter_ms * 1e-3 * fs))
                pos = pos + jitter
                if pos < 0 or pos >= n:
                    break
                r_peaks.append(pos)
                # Overlay template
                start = max(0, pos - half)
                end = min(n, pos + half + 1)
                ts = start - (pos - half)
                te = ts + (end - start)
                ecg[start:end] += template[ts:te]
                # Advance by RR
                rr_samples = int(rr_mean * fs)
                pos = pos + rr_samples

            r_peaks = np.array(r_peaks, dtype=np.int32)

        if add_noise:
            noise = self.rng.standard_normal(n).astype(np.float32) * self.cfg.noise_std
            ecg += noise

        return ecg, r_peaks

    # ------------------------------------------------------------------
    def _vf_signal(self, n: int) -> np.ndarray:
        """
        VF: superposition of sinusoids in [3–7 Hz] with random AM and phase,
        plus amplitude modulation.
        """
        fs = self.cfg.fs
        t = np.arange(n, dtype=np.float32) / fs
        vf = np.zeros(n, dtype=np.float32)
        n_components = 6
        freqs = self.rng.uniform(
            self.cfg.vf_freq_low_hz, self.cfg.vf_freq_high_hz, n_components
        ).astype(np.float32)
        amps = self.rng.uniform(0.1, 0.4, n_components).astype(np.float32)
        phases = self.rng.uniform(0, 2 * np.pi, n_components).astype(np.float32)
        for f, a, ph in zip(freqs, amps, phases):
            vf += a * np.sin(2 * np.pi * f * t + ph)
        # Random amplitude modulation (0.5–1.5 Hz)
        am_freq = self.rng.uniform(0.5, 1.5)
        am = (0.5 + 0.5 * np.sin(2 * np.pi * float(am_freq) * t)).astype(np.float32)
        return (vf * am).astype(np.float32)


# ---------------------------------------------------------------------------
# Perturbation injectors
# ---------------------------------------------------------------------------

class PerturbationInjector:
    """
    Adds physiological and environmental artifacts to a clean ECG.

    All operations are in-place on pre-allocated float32 arrays.
    """

    def __init__(
        self,
        cfg: SyntheticConfig = DEFAULT_CONFIG.synthetic,
        seed: int = 42,
    ) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def baseline_wander(self, signal: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
        """
        Add baseline wander: sum of sinusoids (0.05–0.5 Hz) + random walk.
        """
        n = len(signal)
        t = np.arange(n, dtype=np.float32) / fs
        bw = np.zeros(n, dtype=np.float32)
        for f in self.cfg.bw_freqs_hz:
            phase = float(self.rng.uniform(0, 2 * np.pi))
            bw += np.float32(self.cfg.bw_amplitude_mv / len(self.cfg.bw_freqs_hz)) * np.sin(
                2 * np.pi * np.float32(f) * t + np.float32(phase)
            )
        # Random walk component (integrate white noise)
        rw = np.cumsum(self.rng.normal(0, 0.001, n).astype(np.float32))
        rw -= rw.mean()
        rw = rw / (np.abs(rw).max() + 1e-8) * np.float32(self.cfg.bw_amplitude_mv * 0.3)
        return (signal + bw + rw).astype(np.float32)

    def motion_artifact(self, signal: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
        """Low-frequency motion burst (1–3 s duration) at a random position."""
        n = len(signal)
        t = np.arange(n, dtype=np.float32) / fs
        burst_dur = float(self.rng.uniform(1.0, 3.0))
        burst_start = float(self.rng.uniform(0, max(0.1, (n / fs) - burst_dur)))
        burst_end = burst_start + burst_dur
        mask = ((t >= burst_start) & (t <= burst_end)).astype(np.float32)
        motion = (
            np.float32(self.cfg.motion_amplitude_mv)
            * np.sin(2 * np.pi * np.float32(self.cfg.motion_freq_hz) * t)
            * mask
        )
        out = signal.copy()
        out += motion
        return out.astype(np.float32)

    def emg_burst(
        self,
        signal: np.ndarray,
        fs: int = SAMPLE_RATE,
        snr_db: Optional[float] = None,
    ) -> np.ndarray:
        """
        Band-limited EMG noise (20–450 Hz) at specified SNR.
        Uses a second-order Butterworth band-pass filter.
        """
        from scipy.signal import butter, sosfilt

        if snr_db is None:
            snr_db = self.cfg.emg_snr_db

        n = len(signal)
        raw_noise = self.rng.standard_normal(n).astype(np.float32)
        # Band-pass 20–450 Hz
        nyq = fs / 2.0
        lo = max(0.01, self.cfg.emg_band_low_hz / nyq)
        hi = min(0.99, self.cfg.emg_band_high_hz / nyq)
        sos = butter(2, [lo, hi], btype="band", output="sos")
        emg = sosfilt(sos, raw_noise).astype(np.float32)

        # Scale to desired SNR
        sig_power = float(np.mean(signal ** 2))
        emg_power = float(np.mean(emg ** 2))
        if emg_power > 0 and sig_power > 0:
            target_emg_power = sig_power / (10 ** (snr_db / 10.0))
            emg = emg * np.float32(np.sqrt(target_emg_power / emg_power))

        return (signal + emg).astype(np.float32)

    def mains_interference(
        self, signal: np.ndarray, fs: int = SAMPLE_RATE
    ) -> np.ndarray:
        """Add 50/60 Hz mains interference."""
        n = len(signal)
        t = np.arange(n, dtype=np.float32) / fs
        phase = float(self.rng.uniform(0, 2 * np.pi))
        mains = np.float32(self.cfg.mains_amplitude_mv) * np.sin(
            2 * np.pi * np.float32(self.cfg.mains_freq_hz) * t + np.float32(phase)
        )
        return (signal + mains).astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience: build labelled dataset from synthetic data
# ---------------------------------------------------------------------------

def build_synthetic_dataset(
    duration_per_class_sec: float = 30.0,
    seed: int = 42,
    cfg: SyntheticConfig = DEFAULT_CONFIG.synthetic,
    add_perturbations: bool = False,
    perturbation_snr_db: float = 10.0,
) -> dict:
    """
    Build a labelled synthetic dataset.

    Returns a dict with keys:
      'records': list of dicts {signal, r_peaks, label, mode}
      'source': 'synthetic'
    """
    gen = SyntheticECG(cfg=cfg, seed=seed)
    inj = PerturbationInjector(cfg=cfg, seed=seed + 1)
    records = []

    for mode, label in [("normal", 0), ("vt", 1), ("vf", 2)]:
        ecg, r_peaks = gen.generate(
            duration_sec=duration_per_class_sec,
            mode=mode,  # type: ignore[arg-type]
        )
        if add_perturbations:
            ecg = inj.baseline_wander(ecg)
            ecg = inj.emg_burst(ecg, snr_db=perturbation_snr_db)
        records.append(
            {
                "signal": ecg,
                "r_peaks": r_peaks,
                "label": label,       # 0=Normal, 1=VT, 2=VF
                "mode": mode,
                "fs": cfg.fs,
            }
        )

    return {"records": records, "source": "synthetic"}
