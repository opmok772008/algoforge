"""
features/rpeak.py — Pan-Tompkins style R-peak detector with adaptive thresholds.

Steps:
  1. Band-pass filter (5–15 Hz) to isolate QRS energy
  2. Differentiation (first-difference)
  3. Squaring (non-linear amplification)
  4. Moving-window integration
  5. Adaptive thresholding with running signal and noise estimates
  6. Search-back for missed beats
  7. Refractory period enforcement

All arrays are float32; no growing lists in the streaming path.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from biosig_fuzzy_pso.config import RPeakConfig, SAMPLE_RATE, DEFAULT_CONFIG


class RPeakDetector:
    """
    Pan-Tompkins style R-peak detector with adaptive thresholds.

    Parameters
    ----------
    cfg : RPeakConfig
    fs  : sample rate in Hz
    """

    def __init__(
        self,
        cfg: RPeakConfig = DEFAULT_CONFIG.rpeak,
        fs: int = SAMPLE_RATE,
    ) -> None:
        self.cfg = cfg
        self.fs = fs
        # Band-pass SOS filter (5–15 Hz)
        nyq = fs / 2.0
        lo = max(0.01, cfg.bp_low_hz / nyq)
        hi = min(0.99, cfg.bp_high_hz / nyq)
        self._sos = butter(cfg.bp_order, [lo, hi], btype="band", output="sos")
        # Moving window
        self._mw = cfg.mw_width_samples(fs)
        # Refractory
        self._ref = cfg.refractory_samples(fs)

    # ------------------------------------------------------------------
    def detect(self, signal: np.ndarray) -> np.ndarray:
        """
        Detect R-peak sample indices in the given signal.

        Parameters
        ----------
        signal : float32 (n_samples,)

        Returns
        -------
        r_peaks : int32 array of sample indices (sorted, unique)
        """
        signal = np.asarray(signal, dtype=np.float32)
        n = len(signal)
        if n < self._mw * 2:
            return np.array([], dtype=np.int32)

        # --- Stage 1: Band-pass filter ---
        bp = sosfiltfilt(self._sos, signal).astype(np.float32)

        # --- Stage 2: Differentiate (5-point derivative) ---
        diff = np.zeros(n, dtype=np.float32)
        diff[2:-2] = (2 * bp[4:] + bp[3:-1] - bp[1:-3] - 2 * bp[:-4]) / (8.0 / self.fs)

        # --- Stage 3: Square ---
        squared = (diff ** 2).astype(np.float32)

        # --- Stage 4: Moving-window integration ---
        kernel = np.ones(self._mw, dtype=np.float32) / self._mw
        mwi = np.convolve(squared, kernel, mode="same").astype(np.float32)

        # --- Stage 5: Adaptive thresholding ---
        r_peaks = self._adaptive_threshold(mwi)

        return r_peaks

    # ------------------------------------------------------------------
    def _adaptive_threshold(self, mwi: np.ndarray) -> np.ndarray:
        """
        Adaptive threshold from Pan-Tompkins with search-back.

        Uses running estimates of signal and noise peaks to adapt the
        threshold dynamically.
        """
        n = len(mwi)
        # Initialise estimates from first 2 seconds
        init_end = min(2 * self.fs, n)
        spki = float(np.max(mwi[:init_end]))  # signal peak estimate
        npki = float(np.mean(mwi[:init_end]))  # noise peak estimate

        threshold1 = npki + self.cfg.threshold_factor_signal * (spki - npki)

        peaks = []
        last_peak = -self._ref

        # Search-back window
        sb_window = int(1.5 * self.fs)  # 1.5 s

        i = 1
        while i < n - 1:
            # Local max detection
            if mwi[i] > mwi[i - 1] and mwi[i] >= mwi[i + 1]:
                if mwi[i] >= threshold1 and (i - last_peak) >= self._ref:
                    # Accepted as signal peak
                    spki = 0.125 * mwi[i] + 0.875 * spki
                    threshold1 = npki + self.cfg.threshold_factor_signal * (spki - npki)
                    peaks.append(i)
                    last_peak = i
                else:
                    # Noise peak
                    npki = 0.125 * mwi[i] + 0.875 * npki
                    threshold1 = npki + self.cfg.threshold_factor_signal * (spki - npki)

                    # Search-back: if RR interval > 1.66 * mean, lower threshold
                    if len(peaks) >= 2:
                        mean_rr = np.mean(np.diff(peaks[-5:]))
                        if (i - last_peak) > 1.66 * mean_rr:
                            threshold2 = (
                                self.cfg.searchback_factor * threshold1
                            )
                            sb_start = max(0, last_peak + self._ref)
                            sb_end = i
                            candidate = sb_start + int(
                                np.argmax(mwi[sb_start:sb_end])
                            )
                            if (
                                mwi[candidate] > threshold2
                                and (candidate - last_peak) >= self._ref
                            ):
                                peaks.append(candidate)
                                last_peak = candidate
                                spki = 0.125 * mwi[candidate] + 0.875 * spki
            i += 1

        if not peaks:
            return np.array([], dtype=np.int32)

        return np.array(sorted(set(peaks)), dtype=np.int32)


# ---------------------------------------------------------------------------
# Convenience: R-peak amplitudes for downstream features
# ---------------------------------------------------------------------------

def extract_r_amplitudes(
    signal: np.ndarray, r_peaks: np.ndarray, window: int = 5
) -> np.ndarray:
    """
    Extract the maximum amplitude in a small window around each R-peak.
    """
    if len(r_peaks) == 0:
        return np.array([], dtype=np.float32)
    amps = []
    n = len(signal)
    for r in r_peaks:
        s = max(0, r - window)
        e = min(n, r + window + 1)
        amps.append(float(np.max(np.abs(signal[s:e]))))
    return np.array(amps, dtype=np.float32)
