"""
dsp/quality.py — Fuzzy Signal Quality Index (SQI).

Computes a scalar SQI in [0, 1] from multiple complementary sub-indices:
  - kSQI  : kurtosis (QRS sharpness)
  - pSQI  : spectral purity (ratio of QRS-band power to total power)
  - basSQI: baseline noise (high-frequency power fraction)
  - rSQI  : R-peak amplitude consistency (1 - CV of R amplitudes)

The sub-indices are combined via a weighted Sugeno-style averaging rule.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from biosig_fuzzy_pso.config import SAMPLE_RATE


# ---------------------------------------------------------------------------
# Individual SQI sub-indices
# ---------------------------------------------------------------------------

def kurtosis_sqi(window: np.ndarray) -> float:
    """
    kSQI = signal kurtosis (excess).
    High kurtosis implies sharp R-peaks; clean ECG typically has kSQI > 5.
    Normalized to [0, 1] by mapping [0, 20] -> [0, 1].
    """
    mu = float(np.mean(window))
    sigma = float(np.std(window))
    if sigma < 1e-9:
        return 0.0
    kurt = float(np.mean(((window - mu) / sigma) ** 4)) - 3.0  # excess kurtosis
    return float(np.clip(kurt / 20.0, 0.0, 1.0))


def spectral_purity_sqi(window: np.ndarray, fs: int = SAMPLE_RATE) -> float:
    """
    pSQI = fraction of power in QRS-band [5–15 Hz] / total power.
    Clean ECG dominated by QRS energy; noisy signal has spread spectrum.
    """
    nfft = 256
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    psd = np.abs(np.fft.rfft(window[:nfft], n=nfft)) ** 2
    total_power = float(np.sum(psd)) + 1e-12
    qrs_mask = (freqs >= 5.0) & (freqs <= 15.0)
    qrs_power = float(np.sum(psd[qrs_mask]))
    return float(np.clip(qrs_power / total_power, 0.0, 1.0))


def baseline_noise_sqi(window: np.ndarray, fs: int = SAMPLE_RATE) -> float:
    """
    basSQI = 1 - (high-frequency noise fraction).
    High-freq power (> 40 Hz) indicates EMG/noise; low fraction -> high quality.
    """
    nfft = 256
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    psd = np.abs(np.fft.rfft(window[:nfft], n=nfft)) ** 2
    total_power = float(np.sum(psd)) + 1e-12
    hf_mask = freqs > 40.0
    hf_power = float(np.sum(psd[hf_mask]))
    hf_frac = float(np.clip(hf_power / total_power, 0.0, 1.0))
    return 1.0 - hf_frac


def r_amplitude_sqi(r_amplitudes: np.ndarray) -> float:
    """
    rSQI = 1 - CV(R_amplitudes).
    Consistent R-peaks -> low CV -> high quality.
    Returns 0.0 if fewer than 2 peaks available.
    """
    if len(r_amplitudes) < 2:
        return 0.5  # neutral
    mu = float(np.mean(r_amplitudes))
    if mu < 1e-9:
        return 0.0
    cv = float(np.std(r_amplitudes)) / mu
    return float(np.clip(1.0 - cv, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Composite fuzzy SQI
# ---------------------------------------------------------------------------

def compute_sqi(
    window: np.ndarray,
    fs: int = SAMPLE_RATE,
    r_amplitudes: np.ndarray | None = None,
    weights: tuple = (0.35, 0.30, 0.20, 0.15),
) -> float:
    """
    Compute composite Signal Quality Index.

    Parameters
    ----------
    window       : float32 ECG window (n_samples,)
    fs           : sample rate
    r_amplitudes : optional array of R-peak amplitudes in the window
    weights      : (w_kSQI, w_pSQI, w_basSQI, w_rSQI) — must sum to 1

    Returns
    -------
    sqi : float in [0, 1]
    """
    k = kurtosis_sqi(window)
    p = spectral_purity_sqi(window, fs)
    b = baseline_noise_sqi(window, fs)
    r = r_amplitude_sqi(r_amplitudes if r_amplitudes is not None else np.array([]))

    w = weights
    sqi = w[0] * k + w[1] * p + w[2] * b + w[3] * r
    return float(np.clip(sqi, 0.0, 1.0))
