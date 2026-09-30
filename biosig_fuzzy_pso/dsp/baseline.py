"""
dsp/baseline.py — Zero-phase baseline-wander removal (C3 compliant).

Strategy: cascaded median filters (~200 ms then ~600 ms window), which are
zero-phase (no group delay), followed by subtraction.  A FIR high-pass
fallback (fc ≤ 0.5 Hz) is available for reference comparisons.

QRS amplitude and ST level are preserved because the baseline estimate
is subtracted rather than the signal being high-pass filtered.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import firwin, filtfilt

from biosig_fuzzy_pso.config import BaselineConfig, SAMPLE_RATE, DEFAULT_CONFIG


class BaselineRemover:
    """
    Zero-phase baseline-wander removal using cascaded median filters.

    Parameters
    ----------
    cfg : BaselineConfig
    fs  : sample rate in Hz
    """

    def __init__(
        self,
        cfg: BaselineConfig = DEFAULT_CONFIG.baseline,
        fs: int = SAMPLE_RATE,
    ) -> None:
        self.cfg = cfg
        self.fs = fs
        self._w1 = cfg.med_win1_samples(fs)
        self._w2 = cfg.med_win2_samples(fs)

    # ------------------------------------------------------------------
    def remove(self, signal: np.ndarray) -> np.ndarray:
        """
        Remove baseline wander.

        Parameters
        ----------
        signal : float32 array (n_samples,)

        Returns
        -------
        clean : float32 array (n_samples,)
        """
        signal = np.asarray(signal, dtype=np.float32)
        # Stage 1: coarse baseline (200 ms window)
        bl1 = median_filter(signal, size=self._w1, mode="reflect").astype(np.float32)
        # Stage 2: refined baseline (600 ms window)
        bl2 = median_filter(bl1, size=self._w2, mode="reflect").astype(np.float32)
        return (signal - bl2).astype(np.float32)

    # ------------------------------------------------------------------
    def remove_fir(self, signal: np.ndarray) -> np.ndarray:
        """
        FIR high-pass baseline removal (fc <= 0.5 Hz, zero-phase via filtfilt).
        Provided for comparison / fallback only.
        """
        signal = np.asarray(signal, dtype=np.float32)
        nyq = self.fs / 2.0
        cutoff = self.cfg.fir_cutoff_hz / nyq
        # firwin gives a *low-pass* prototype; subtract to get high-pass
        lp_taps = firwin(self.cfg.fir_order + 1, cutoff, window="hamming").astype(np.float32)
        baseline = filtfilt(lp_taps, [1.0], signal).astype(np.float32)
        return (signal - baseline).astype(np.float32)


# ---------------------------------------------------------------------------
# C3 measurement helpers
# ---------------------------------------------------------------------------

def measure_qrs_amplitude_distortion(
    original: np.ndarray,
    processed: np.ndarray,
    r_peaks: np.ndarray,
    fs: int = SAMPLE_RATE,
    window_ms: float = 50.0,
) -> float:
    """
    QRS amplitude distortion = |A_out - A_ref| / A_ref   (mean over beats).
    Limit: <= 5 percent (0.05).
    """
    if len(r_peaks) == 0:
        return 0.0
    half = max(1, int(window_ms * 1e-3 * fs // 2))
    distortions = []
    for r in r_peaks:
        s, e = max(0, r - half), min(len(original), r + half)
        a_ref = float(np.max(np.abs(original[s:e])))
        a_out = float(np.max(np.abs(processed[s:e])))
        if a_ref > 1e-6:
            distortions.append(abs(a_out - a_ref) / a_ref)
    return float(np.mean(distortions)) if distortions else 0.0


def measure_st_error(
    original: np.ndarray,
    processed: np.ndarray,
    r_peaks: np.ndarray,
    fs: int = SAMPLE_RATE,
    st_offset_ms: float = 80.0,
    st_window_ms: float = 40.0,
) -> float:
    """
    ST-level error = |ST_out - ST_ref| in mV   (mean over beats).
    Limit: <= 0.02 mV.
    """
    if len(r_peaks) == 0:
        return 0.0
    j_offset = int(st_offset_ms * 1e-3 * fs)
    win = max(1, int(st_window_ms * 1e-3 * fs))
    errors = []
    for r in r_peaks:
        j = r + j_offset
        if j + win >= len(original):
            continue
        st_ref = float(np.mean(original[j: j + win]))
        st_out = float(np.mean(processed[j: j + win]))
        errors.append(abs(st_out - st_ref))
    return float(np.mean(errors)) if errors else 0.0
