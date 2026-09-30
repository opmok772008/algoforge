"""
features/fiducials.py — Fiducial-point extraction and feature vector computation.

Extracts:
  - QRS width (onset to offset around R)
  - ST level at J+80 ms
  - RR statistics (mean HR, coefficient of variation)
  - R-amplitude variability
  - Dominant spectral frequency and spectral concentration (VF indicator)

All features are normalised to [0, 1] for the FIS.
Returns a named dataclass for type safety.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from biosig_fuzzy_pso.config import FiducialConfig, FISConfig, SAMPLE_RATE, DEFAULT_CONFIG
from biosig_fuzzy_pso.features.rpeak import extract_r_amplitudes


# ---------------------------------------------------------------------------
# Feature vector dataclass
# ---------------------------------------------------------------------------

@dataclass
class FiducialFeatures:
    """
    Normalised feature vector fed into the Fuzzy Inference System.
    All values are in [0, 1] unless otherwise noted.
    """
    heart_rate: float       # 0  normalised: [20, 300] bpm -> [0, 1]
    rr_cv: float            # 1  RR coefficient of variation (clipped to [0, 1])
    qrs_width: float        # 2  normalised: [40, 200] ms -> [0, 1]
    r_amp_var: float        # 3  CV of R amplitudes -> [0, 1]
    dom_freq: float         # 4  dominant spectral frequency: [0.5, 15] Hz -> [0, 1]
    spectral_conc: float    # 5  fraction of power in [3, 9] Hz -> [0, 1]
    st_deviation: float     # 6  |ST offset| mV: [0, 0.5] mV -> [0, 1]
    sqi: float              # 7  Signal Quality Index [0, 1]

    # Raw (un-normalised) for reporting
    hr_bpm: float = 0.0
    qrs_width_ms: float = 0.0
    st_mv: float = 0.0
    rr_mean_ms: float = 0.0

    def to_array(self) -> np.ndarray:
        """Return the 8-element normalised feature array (float32)."""
        return np.array(
            [
                self.heart_rate,
                self.rr_cv,
                self.qrs_width,
                self.r_amp_var,
                self.dom_freq,
                self.spectral_conc,
                self.st_deviation,
                self.sqi,
            ],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Normalisation bounds (must match FISConfig input names)
# ---------------------------------------------------------------------------
_HR_MIN, _HR_MAX = 20.0, 300.0          # bpm
_QRS_MIN, _QRS_MAX = 40.0, 200.0       # ms
_FREQ_MIN, _FREQ_MAX = 0.5, 15.0       # Hz
_ST_MAX = 0.5                           # mV absolute


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class FiducialExtractor:
    """
    Extracts and normalises fiducial features from an ECG window.

    Parameters
    ----------
    cfg    : FiducialConfig
    fs_cfg : FISConfig  (for normalisation bounds reference)
    fs     : sample rate
    """

    def __init__(
        self,
        cfg: FiducialConfig = DEFAULT_CONFIG.fiducial,
        fs: int = SAMPLE_RATE,
    ) -> None:
        self.cfg = cfg
        self.fs = fs
        self._qrs_half = max(1, int(cfg.qrs_search_ms * 1e-3 * fs))
        self._j_offset = int(cfg.st_offset_ms * 1e-3 * fs)
        self._st_win = max(1, int(cfg.st_window_ms * 1e-3 * fs))

    # ------------------------------------------------------------------
    def extract(
        self,
        window: np.ndarray,
        r_peaks: np.ndarray,
        sqi: float = 1.0,
    ) -> FiducialFeatures:
        """
        Compute fiducial features from one analysis window.

        Parameters
        ----------
        window  : float32 ECG window (n_samples,)
        r_peaks : int32 R-peak indices relative to the window start
        sqi     : pre-computed Signal Quality Index

        Returns
        -------
        FiducialFeatures dataclass
        """
        window = np.asarray(window, dtype=np.float32)
        n = len(window)

        # ---- RR statistics ----
        rr_vals = self._compute_rr(r_peaks)
        hr_bpm, hr_norm, rr_cv = self._rr_stats(rr_vals)

        # ---- QRS width ----
        qrs_ms, qrs_norm = self._qrs_width(window, r_peaks)

        # ---- R-amplitude variability ----
        r_amps = extract_r_amplitudes(window, r_peaks)
        r_amp_var = self._amplitude_cv(r_amps)

        # ---- ST level ----
        st_mv, st_norm = self._st_level(window, r_peaks)

        # ---- Spectral features ----
        dom_freq_norm, spectral_conc = self._spectral(window)

        return FiducialFeatures(
            heart_rate=hr_norm,
            rr_cv=rr_cv,
            qrs_width=qrs_norm,
            r_amp_var=r_amp_var,
            dom_freq=dom_freq_norm,
            spectral_conc=spectral_conc,
            st_deviation=st_norm,
            sqi=sqi,
            hr_bpm=hr_bpm,
            qrs_width_ms=qrs_ms,
            st_mv=st_mv,
            rr_mean_ms=float(np.mean(rr_vals)) if len(rr_vals) > 0 else 0.0,
        )

    # ------------------------------------------------------------------
    def _compute_rr(self, r_peaks: np.ndarray) -> np.ndarray:
        """RR intervals in ms."""
        if len(r_peaks) < 2:
            return np.array([], dtype=np.float32)
        rr = np.diff(r_peaks).astype(np.float32) / self.fs * 1000.0
        # Physiological limits
        mask = (rr >= self.cfg.rr_min_ms) & (rr <= self.cfg.rr_max_ms)
        return rr[mask]

    def _rr_stats(
        self, rr_vals: np.ndarray
    ) -> tuple[float, float, float]:
        """Returns (hr_bpm, hr_normalised, rr_cv)."""
        if len(rr_vals) == 0:
            return 75.0, 0.25, 0.5   # neutral defaults
        rr_mean = float(np.mean(rr_vals))
        hr_bpm = 60_000.0 / rr_mean if rr_mean > 1 else 75.0
        hr_norm = float(np.clip((hr_bpm - _HR_MIN) / (_HR_MAX - _HR_MIN), 0.0, 1.0))
        rr_cv = float(np.std(rr_vals) / (rr_mean + 1e-6))
        rr_cv = float(np.clip(rr_cv, 0.0, 1.0))
        return hr_bpm, hr_norm, rr_cv

    def _qrs_width(
        self, window: np.ndarray, r_peaks: np.ndarray
    ) -> tuple[float, float]:
        """Estimate QRS width by threshold crossing around R-peak."""
        if len(r_peaks) == 0:
            return 80.0, 0.25
        widths_ms = []
        for r in r_peaks:
            s = max(0, r - self._qrs_half)
            e = min(len(window), r + self._qrs_half)
            seg = np.abs(window[s:e])
            if len(seg) == 0:
                continue
            thr = 0.5 * float(np.max(seg))
            above = np.where(seg >= thr)[0]
            if len(above) < 2:
                continue
            width_samples = above[-1] - above[0]
            widths_ms.append(width_samples / self.fs * 1000.0)
        if not widths_ms:
            return 80.0, 0.25
        qrs_ms = float(np.median(widths_ms))
        qrs_norm = float(
            np.clip((qrs_ms - _QRS_MIN) / (_QRS_MAX - _QRS_MIN), 0.0, 1.0)
        )
        return qrs_ms, qrs_norm

    def _amplitude_cv(self, r_amps: np.ndarray) -> float:
        """CV of R-amplitude (clipped to [0, 1])."""
        if len(r_amps) < 2:
            return 0.0
        mu = float(np.mean(r_amps))
        if mu < 1e-9:
            return 0.0
        return float(np.clip(float(np.std(r_amps)) / mu, 0.0, 1.0))

    def _st_level(
        self, window: np.ndarray, r_peaks: np.ndarray
    ) -> tuple[float, float]:
        """Mean ST level (mV) and normalised deviation."""
        if len(r_peaks) == 0:
            return 0.0, 0.0
        st_vals = []
        for r in r_peaks:
            j = r + self._j_offset
            if j + self._st_win >= len(window):
                continue
            st_vals.append(float(np.mean(window[j: j + self._st_win])))
        if not st_vals:
            return 0.0, 0.0
        st_mv = float(np.mean(st_vals))
        st_norm = float(np.clip(abs(st_mv) / _ST_MAX, 0.0, 1.0))
        return st_mv, st_norm

    def _spectral(self, window: np.ndarray) -> tuple[float, float]:
        """Dominant spectral frequency and spectral concentration in VF band."""
        nfft = self.cfg.spectral_nfft
        freqs = np.fft.rfftfreq(nfft, 1.0 / self.fs)
        psd = np.abs(np.fft.rfft(window[:nfft], n=nfft)) ** 2

        # Dominant frequency (peak of PSD)
        dom_idx = int(np.argmax(psd))
        dom_freq = float(freqs[dom_idx]) if dom_idx < len(freqs) else 1.0
        dom_norm = float(
            np.clip((dom_freq - _FREQ_MIN) / (_FREQ_MAX - _FREQ_MIN), 0.0, 1.0)
        )

        # Spectral concentration in VF band
        total = float(np.sum(psd)) + 1e-12
        vf_mask = (freqs >= self.cfg.spectral_fmin_hz) & (freqs <= self.cfg.spectral_fmax_hz)
        vf_power = float(np.sum(psd[vf_mask]))
        spectral_conc = float(np.clip(vf_power / total, 0.0, 1.0))

        return dom_norm, spectral_conc
