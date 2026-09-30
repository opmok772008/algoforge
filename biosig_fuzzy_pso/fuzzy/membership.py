"""
fuzzy/membership.py — Membership function implementations (vectorized, float32).

Supported types:
  - Triangular (trimf)
  - Gaussian   (gaussmf)

All functions accept scalar or numpy array inputs and always return float32.
"""

from __future__ import annotations

import numpy as np


def trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Triangular membership function.

    Parameters
    ----------
    x       : input value(s)
    a, b, c : left foot, peak, right foot  (a <= b <= c)

    Returns
    -------
    mu : float32 array in [0, 1]
    """
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    if b > a:
        mask1 = (x >= a) & (x <= b)
        out[mask1] = (x[mask1] - a) / (b - a)
    if c > b:
        mask2 = (x > b) & (x <= c)
        out[mask2] = (c - x[mask2]) / (c - b)
    if b == a and b == c:
        out[x == b] = 1.0
    elif b == a:
        out[x <= b] = 1.0
    elif b == c:
        out[x >= b] = 1.0
    return out.astype(np.float32)


def gaussmf(x: np.ndarray, center: float, sigma: float) -> np.ndarray:
    """
    Gaussian membership function.

    mu(x) = exp(-0.5 * ((x - c) / sigma)^2)

    Parameters
    ----------
    x      : input value(s)
    center : Gaussian centre
    sigma  : Gaussian width  (> 0)

    Returns
    -------
    mu : float32 array in [0, 1]
    """
    x = np.asarray(x, dtype=np.float32)
    sigma = max(float(sigma), 1e-6)
    return np.exp(-0.5 * ((x - np.float32(center)) / np.float32(sigma)) ** 2).astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# MF parameter set (for one input with 3 MFs: Low / Med / High)
# ---------------------------------------------------------------------------

class InputMFs:
    """
    Three Gaussian MFs (Low, Med, High) for a single FIS input.

    Parameters stored as (center, sigma) pairs.
    """

    def __init__(
        self,
        low_c: float,
        low_s: float,
        med_c: float,
        med_s: float,
        high_c: float,
        high_s: float,
    ) -> None:
        self.params = np.array(
            [[low_c, low_s], [med_c, med_s], [high_c, high_s]], dtype=np.float32
        )

    def evaluate(self, x: float) -> np.ndarray:
        """
        Evaluate all three MFs at scalar x.

        Returns
        -------
        mu : float32 array (3,)  — [mu_low, mu_med, mu_high]
        """
        x_arr = np.float32(x)
        mu = np.array(
            [
                float(gaussmf(x_arr, self.params[i, 0], self.params[i, 1]))
                for i in range(3)
            ],
            dtype=np.float32,
        )
        return mu

    def evaluate_batch(self, xs: np.ndarray) -> np.ndarray:
        """
        Evaluate all three MFs on a batch of inputs.

        Parameters
        ----------
        xs : float32 (N,)

        Returns
        -------
        mu : float32 (N, 3)
        """
        xs = np.asarray(xs, dtype=np.float32)
        cols = []
        for i in range(3):
            cols.append(gaussmf(xs, self.params[i, 0], self.params[i, 1]))
        return np.stack(cols, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Default MF centres for 8 inputs (uniform spread across [0, 1])
# ---------------------------------------------------------------------------
DEFAULT_MF_CENTERS = np.array(
    [
        # low_c, low_s, med_c, med_s, high_c, high_s
        [0.15, 0.15, 0.50, 0.15, 0.85, 0.15],  # 0 heart_rate
        [0.15, 0.15, 0.50, 0.15, 0.85, 0.15],  # 1 rr_cv
        [0.15, 0.15, 0.50, 0.15, 0.85, 0.15],  # 2 qrs_width
        [0.15, 0.15, 0.50, 0.15, 0.85, 0.15],  # 3 r_amp_var
        [0.15, 0.15, 0.50, 0.15, 0.85, 0.15],  # 4 dom_freq
        [0.15, 0.15, 0.50, 0.15, 0.85, 0.15],  # 5 spectral_conc
        [0.15, 0.15, 0.50, 0.15, 0.85, 0.15],  # 6 st_deviation
        [0.15, 0.15, 0.50, 0.15, 0.85, 0.15],  # 7 sqi
    ],
    dtype=np.float32,
)


def build_input_mfs(mf_params: np.ndarray) -> list[InputMFs]:
    """
    Build a list of InputMFs from a (8, 6) parameter array.

    Parameters
    ----------
    mf_params : (n_inputs, 6) float32 — [lc, ls, mc, ms, hc, hs] per row

    Returns
    -------
    list of InputMFs of length n_inputs
    """
    return [
        InputMFs(
            float(row[0]), float(row[1]),
            float(row[2]), float(row[3]),
            float(row[4]), float(row[5]),
        )
        for row in mf_params
    ]
