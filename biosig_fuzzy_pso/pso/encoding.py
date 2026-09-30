"""
pso/encoding.py — Particle vector encoding / decoding.

The PSO optimises a single flat float32 vector per particle.
This module defines the layout (offsets and bounds) and provides
encode / decode functions to convert between the flat vector and
the structured parameter objects used by the pipeline.

Particle vector layout (total dimension D):
  [0 : 48]  MF params: 8 inputs × 6 params (lc, ls, mc, ms, hc, hs)
  [48: 66]  Rule weights: 18 values
  [66]      NLMS mu0 (base step size)
  [67]      NLMS filter length (discretised)
  [68]      Baseline med_win1_ms
  [69]      Baseline med_win2_ms
  [70]      R-peak threshold_factor_signal
  [71]      R-peak threshold_factor_noise
  [72]      tau_lethal
  [73]      tau_artifact
Total: D = 74
"""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import numpy as np

from biosig_fuzzy_pso.config import (
    BaselineConfig, FISConfig, NLMSConfig, RPeakConfig, DEFAULT_CONFIG
)
from biosig_fuzzy_pso.fuzzy.rules import RULE_BASE


# ---------------------------------------------------------------------------
# Dimension layout
# ---------------------------------------------------------------------------
N_INPUTS = 8
MF_PARAMS_PER_INPUT = 6            # lc, ls, mc, ms, hc, hs
N_MF_PARAMS = N_INPUTS * MF_PARAMS_PER_INPUT   # 48
N_RULES = len(RULE_BASE)                        # 18

_OFF_MF = 0
_OFF_RW = N_MF_PARAMS               # 48
_OFF_MU0 = _OFF_RW + N_RULES        # 66
_OFF_FLEN = _OFF_MU0 + 1            # 67
_OFF_BW1 = _OFF_FLEN + 1            # 68
_OFF_BW2 = _OFF_BW1 + 1            # 69
_OFF_TF1 = _OFF_BW2 + 1            # 70
_OFF_TF2 = _OFF_TF1 + 1            # 71
_OFF_TAU_L = _OFF_TF2 + 1          # 72
_OFF_TAU_A = _OFF_TAU_L + 1        # 73
PARTICLE_DIM = _OFF_TAU_A + 1      # 74


# ---------------------------------------------------------------------------
# Bounds (lower, upper) per dimension
# ---------------------------------------------------------------------------

def build_bounds() -> Tuple[np.ndarray, np.ndarray]:
    """
    Build lower-bound and upper-bound arrays for all particle dimensions.

    Returns
    -------
    lb, ub : (D,) float32 arrays
    """
    lb = np.empty(PARTICLE_DIM, dtype=np.float32)
    ub = np.empty(PARTICLE_DIM, dtype=np.float32)

    # MF center/sigma bounds (all inputs normalised to [0,1])
    # Centers: [0.0, 1.0];  Sigmas: [0.02, 0.4]
    for i in range(N_INPUTS):
        base = _OFF_MF + i * MF_PARAMS_PER_INPUT
        # lc, mc, hc centers
        lb[base + 0] = 0.0;  ub[base + 0] = 0.45   # low center
        lb[base + 1] = 0.02; ub[base + 1] = 0.40   # low sigma
        lb[base + 2] = 0.20; ub[base + 2] = 0.80   # med center
        lb[base + 3] = 0.02; ub[base + 3] = 0.40   # med sigma
        lb[base + 4] = 0.55; ub[base + 4] = 1.00   # high center
        lb[base + 5] = 0.02; ub[base + 5] = 0.40   # high sigma

    # Rule weights: [0.1, 1.0]
    lb[_OFF_RW: _OFF_RW + N_RULES] = 0.1
    ub[_OFF_RW: _OFF_RW + N_RULES] = 1.0

    # NLMS mu0: [0.001, 0.5]
    lb[_OFF_MU0] = 0.001; ub[_OFF_MU0] = 0.5

    # NLMS filter length: [8, 64] — treated as float, cast to int when used
    lb[_OFF_FLEN] = 8.0; ub[_OFF_FLEN] = 64.0

    # Baseline windows (ms): [100, 400] and [300, 800]
    lb[_OFF_BW1] = 100.0; ub[_OFF_BW1] = 400.0
    lb[_OFF_BW2] = 300.0; ub[_OFF_BW2] = 800.0

    # R-peak threshold factors: [0.1, 0.5]
    lb[_OFF_TF1] = 0.1; ub[_OFF_TF1] = 0.5
    lb[_OFF_TF2] = 0.2; ub[_OFF_TF2] = 0.8

    # Tau lethal: [0.3, 0.8];  tau artifact: [0.4, 0.9]
    lb[_OFF_TAU_L] = 0.30; ub[_OFF_TAU_L] = 0.80
    lb[_OFF_TAU_A] = 0.40; ub[_OFF_TAU_A] = 0.90

    return lb, ub


LOWER_BOUNDS, UPPER_BOUNDS = build_bounds()


# ---------------------------------------------------------------------------
# Decode particle -> named parameters
# ---------------------------------------------------------------------------

def decode_particle(x: np.ndarray) -> dict:
    """
    Decode a flat particle vector into a dict of parameter objects.

    Parameters
    ----------
    x : (D,) float32

    Returns
    -------
    params : dict with keys:
        'mf_params'    : (8, 6) float32
        'rule_weights' : (n_rules,) float32
        'nlms'         : NLMSConfig
        'baseline'     : BaselineConfig
        'rpeak'        : RPeakConfig
        'tau_lethal'   : float
        'tau_artifact' : float
    """
    x = np.asarray(x, dtype=np.float32)

    mf_params = x[_OFF_MF: _OFF_MF + N_MF_PARAMS].reshape(N_INPUTS, MF_PARAMS_PER_INPUT)
    rule_weights = x[_OFF_RW: _OFF_RW + N_RULES]

    # Enforce MF ordering: low_c < med_c < high_c (clamp if violated)
    for i in range(N_INPUTS):
        row = mf_params[i]
        row[0] = np.clip(row[0], 0.0, row[2] - 0.01)   # lc < mc
        row[4] = np.clip(row[4], row[2] + 0.01, 1.0)   # hc > mc
        row[1] = np.clip(row[1], 0.02, 0.40)
        row[3] = np.clip(row[3], 0.02, 0.40)
        row[5] = np.clip(row[5], 0.02, 0.40)

    nlms = replace(
        DEFAULT_CONFIG.nlms,
        mu0=float(x[_OFF_MU0]),
        filter_length=max(4, int(round(float(x[_OFF_FLEN])))),
    )
    baseline = replace(
        DEFAULT_CONFIG.baseline,
        med_win1_ms=float(x[_OFF_BW1]),
        med_win2_ms=float(x[_OFF_BW2]),
    )
    rpeak = replace(
        DEFAULT_CONFIG.rpeak,
        threshold_factor_signal=float(x[_OFF_TF1]),
        threshold_factor_noise=float(x[_OFF_TF2]),
    )

    return {
        "mf_params": mf_params,
        "rule_weights": rule_weights,
        "nlms": nlms,
        "baseline": baseline,
        "rpeak": rpeak,
        "tau_lethal": float(np.clip(x[_OFF_TAU_L], 0.30, 0.80)),
        "tau_artifact": float(np.clip(x[_OFF_TAU_A], 0.40, 0.90)),
    }


# ---------------------------------------------------------------------------
# Default (initial) particle
# ---------------------------------------------------------------------------

def default_particle() -> np.ndarray:
    """
    Construct the default particle from the DEFAULT_CONFIG values.
    Used as the initial gbest seed.
    """
    from biosig_fuzzy_pso.fuzzy.membership import DEFAULT_MF_CENTERS

    x = np.empty(PARTICLE_DIM, dtype=np.float32)
    x[_OFF_MF: _OFF_MF + N_MF_PARAMS] = DEFAULT_MF_CENTERS.flatten()
    x[_OFF_RW: _OFF_RW + N_RULES] = [r.weight for r in RULE_BASE]
    x[_OFF_MU0] = DEFAULT_CONFIG.nlms.mu0
    x[_OFF_FLEN] = float(DEFAULT_CONFIG.nlms.filter_length)
    x[_OFF_BW1] = DEFAULT_CONFIG.baseline.med_win1_ms
    x[_OFF_BW2] = DEFAULT_CONFIG.baseline.med_win2_ms
    x[_OFF_TF1] = DEFAULT_CONFIG.rpeak.threshold_factor_signal
    x[_OFF_TF2] = DEFAULT_CONFIG.rpeak.threshold_factor_noise
    x[_OFF_TAU_L] = DEFAULT_CONFIG.fis.tau_lethal
    x[_OFF_TAU_A] = DEFAULT_CONFIG.fis.tau_artifact
    return x
