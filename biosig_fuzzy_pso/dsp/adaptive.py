"""
dsp/adaptive.py — NLMS adaptive artifact canceller with fuzzy-controlled step size.

The fuzzy step-size controller adjusts mu based on the current Signal Quality
Index (SQI): high-quality signal -> small mu (preserve morphology);
low-quality signal -> larger mu (fast adaptation to artifact).

Ring-buffer based: no growing lists.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from biosig_fuzzy_pso.config import NLMSConfig, SAMPLE_RATE, DEFAULT_CONFIG


class NLMSFilter:
    """
    Normalised Least Mean Squares adaptive filter.

    Uses a reference signal (e.g., motion or EMG channel) to estimate
    and cancel the artifact component from the primary ECG.

    Parameters
    ----------
    cfg : NLMSConfig
    fs  : sample rate
    """

    def __init__(
        self,
        cfg: NLMSConfig = DEFAULT_CONFIG.nlms,
        fs: int = SAMPLE_RATE,
    ) -> None:
        self.cfg = cfg
        self.fs = fs
        # Pre-allocated weight vector and reference buffer
        self._weights = np.zeros(cfg.filter_length, dtype=np.float32)
        self._ref_buf = np.zeros(cfg.filter_length, dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset filter state (e.g., between records)."""
        self._weights[:] = 0.0
        self._ref_buf[:] = 0.0

    # ------------------------------------------------------------------
    def process_sample(
        self,
        primary: float,
        reference: float,
        mu: Optional[float] = None,
    ) -> float:
        """
        Process a single sample.

        Parameters
        ----------
        primary   : primary ECG sample (float)
        reference : reference artifact channel sample (float)
        mu        : step size override; if None, uses cfg.mu0

        Returns
        -------
        error : float  (artifact-cancelled output)
        """
        mu_eff = float(mu) if mu is not None else self.cfg.mu0

        # Shift buffer
        self._ref_buf[1:] = self._ref_buf[:-1]
        self._ref_buf[0] = np.float32(reference)

        # Filter output
        y = float(np.dot(self._weights, self._ref_buf))
        error = np.float32(primary) - np.float32(y)

        # Update weights: w += mu * e * x / (||x||^2 + eps)
        norm = float(np.dot(self._ref_buf, self._ref_buf)) + self.cfg.epsilon
        self._weights += np.float32(mu_eff / norm) * error * self._ref_buf

        return float(error)

    # ------------------------------------------------------------------
    def process_block(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
        mu_array: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Process a block of samples.

        Parameters
        ----------
        primary   : (N,) float32
        reference : (N,) float32
        mu_array  : (N,) float32 step sizes or None (use cfg.mu0)

        Returns
        -------
        output : (N,) float32
        """
        n = len(primary)
        output = np.empty(n, dtype=np.float32)
        for i in range(n):
            mu_i = float(mu_array[i]) if mu_array is not None else None
            output[i] = self.process_sample(
                float(primary[i]), float(reference[i]), mu_i
            )
        return output


# ---------------------------------------------------------------------------
# Fuzzy step-size controller
# ---------------------------------------------------------------------------

class FuzzyStepController:
    """
    Adjusts the NLMS step size (mu) using a simple Mamdani-style rule
    on the Signal Quality Index (SQI) and residual error power.

    Rules (embedded, not using the full FIS to avoid circular imports):
      IF sqi=High  AND err_power=Low  THEN mu = mu_min   (preserve morphology)
      IF sqi=Low   AND err_power=High THEN mu = mu_max   (fast adaptation)
      IF sqi=Med   THEN mu = mu_mid
      Lethal patterns detected -> override to mu_min (do not disturb signal)
    """

    def __init__(self, cfg: NLMSConfig = DEFAULT_CONFIG.nlms) -> None:
        self.cfg = cfg
        self._mu_mid = float((cfg.mu_min + cfg.mu_max) / 2)

    def compute_mu(
        self,
        sqi: float,
        err_power: float,
        lethal_flag: bool = False,
    ) -> float:
        """
        Compute adaptive step size.

        Parameters
        ----------
        sqi         : Signal Quality Index [0, 1]  (1 = perfect)
        err_power   : normalised residual error power [0, 1]
        lethal_flag : if True, force mu to minimum to protect the signal

        Returns
        -------
        mu : float in [mu_min, mu_max]
        """
        if lethal_flag:
            return self.cfg.mu_min

        # Triangular membership for SQI
        mu_high_sqi = max(0.0, min(1.0, (sqi - 0.5) / 0.5))     # high if sqi > 0.5
        mu_low_sqi = max(0.0, min(1.0, (0.5 - sqi) / 0.5))       # low  if sqi < 0.5

        # Triangular membership for error power
        mu_high_err = max(0.0, min(1.0, err_power))
        mu_low_err = 1.0 - mu_high_err

        # Rules (product T-norm)
        w_min = mu_high_sqi * mu_low_err     # -> mu_min
        w_max = mu_low_sqi * mu_high_err     # -> mu_max
        w_mid = max(0.0, 1.0 - w_min - w_max)  # -> mu_mid

        total = w_min + w_max + w_mid
        if total < 1e-9:
            return self._mu_mid

        mu = (
            w_min * self.cfg.mu_min
            + w_max * self.cfg.mu_max
            + w_mid * self._mu_mid
        ) / total

        # Clamp to configured bounds
        return float(np.clip(mu, self.cfg.mu_min, self.cfg.mu_max))


# ---------------------------------------------------------------------------
# SNR measurement helper
# ---------------------------------------------------------------------------

def compute_snr_improvement(
    clean_ref: np.ndarray,
    noisy: np.ndarray,
    processed: np.ndarray,
) -> float:
    """
    dSNR = SNR_out - SNR_in  (in dB).
    SNR = 10 * log10(signal_power / noise_power)
    Noise approximated as (signal - reference).
    """
    def snr_db(ref: np.ndarray, sig: np.ndarray) -> float:
        noise = sig - ref
        p_sig = float(np.mean(ref ** 2))
        p_noise = float(np.mean(noise ** 2))
        if p_noise < 1e-12:
            return 60.0
        return 10.0 * float(np.log10(p_sig / p_noise + 1e-12))

    return snr_db(clean_ref, processed) - snr_db(clean_ref, noisy)
