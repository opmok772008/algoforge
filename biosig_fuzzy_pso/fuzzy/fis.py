"""
fuzzy/fis.py — Vectorized Sugeno-type Fuzzy Inference System.

Architecture:
  - 8 inputs  (normalised [0,1])
  - 3 MFs per input (Gaussian: Low/Med/High)
  - 18 rules (from rules.py)
  - Rule firing: product T-norm
  - Defuzzification: weighted average (Sugeno zero-order)
  - Outputs: lethal_score, artifact_score, class_logit (→ argmax for class)

All operations are vectorized NumPy float32 for efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from biosig_fuzzy_pso.config import FISConfig, DEFAULT_CONFIG
from biosig_fuzzy_pso.fuzzy.membership import InputMFs, DEFAULT_MF_CENTERS, build_input_mfs
from biosig_fuzzy_pso.fuzzy.rules import RULE_BASE, FuzzyRule


# ---------------------------------------------------------------------------
# FIS output dataclass
# ---------------------------------------------------------------------------

@dataclass
class FISOutput:
    """Output of a single FIS inference pass."""
    lethal_score: float         # [0, 1]
    artifact_score: float       # [0, 1]
    class_index: int            # 0=Normal, 1=VT, 2=VF, 3=Artifact, 4=Other
    class_name: str
    firing_strengths: np.ndarray  # (n_rules,) raw firing strengths

    CLASS_NAMES = ["Normal", "VT", "VF", "Artifact", "Other"]


# ---------------------------------------------------------------------------
# Sugeno FIS engine
# ---------------------------------------------------------------------------

class SugenoFIS:
    """
    Sugeno-type FIS with product T-norm and weighted-average defuzzification.

    Parameters
    ----------
    mf_params     : (n_inputs, 6) float32 — [lc, ls, mc, ms, hc, hs] per input
    rule_weights  : (n_rules,) float32 — rule weights (tuned by PSO)
    rule_base     : list of FuzzyRule (from rules.py)
    cfg           : FISConfig
    """

    def __init__(
        self,
        mf_params: Optional[np.ndarray] = None,
        rule_weights: Optional[np.ndarray] = None,
        rule_base: Optional[List[FuzzyRule]] = None,
        cfg: FISConfig = DEFAULT_CONFIG.fis,
    ) -> None:
        self.cfg = cfg
        self.rule_base = rule_base if rule_base is not None else RULE_BASE

        # MF parameters
        if mf_params is None:
            mf_params = DEFAULT_MF_CENTERS.copy()
        self.mf_params = np.asarray(mf_params, dtype=np.float32)
        self.input_mfs: List[InputMFs] = build_input_mfs(self.mf_params)

        # Rule weights
        if rule_weights is None:
            rule_weights = np.array([r.weight for r in self.rule_base], dtype=np.float32)
        self.rule_weights = np.asarray(rule_weights, dtype=np.float32)

        # Consequent matrix (n_rules, 3)
        self._consequents = np.array(
            [r.consequent for r in self.rule_base], dtype=np.float32
        )

        # Pre-build antecedent index arrays for vectorized evaluation
        self._max_ant = max(len(r.antecedent) for r in self.rule_base)
        n_rules = len(self.rule_base)
        # antecedent_map[r, k] = (input_idx, mf_level) or (-1, -1) for padding
        self._ant_input = np.full((n_rules, self._max_ant), -1, dtype=np.int32)
        self._ant_level = np.full((n_rules, self._max_ant), -1, dtype=np.int32)
        for r_idx, rule in enumerate(self.rule_base):
            for k, (inp, lvl) in enumerate(rule.antecedent):
                self._ant_input[r_idx, k] = inp
                self._ant_level[r_idx, k] = lvl

    # ------------------------------------------------------------------
    def infer(self, features: np.ndarray) -> FISOutput:
        """
        Run one FIS inference pass.

        Parameters
        ----------
        features : (8,) float32 — normalised input vector

        Returns
        -------
        FISOutput
        """
        features = np.asarray(features, dtype=np.float32)
        assert len(features) == self.cfg.n_inputs, (
            f"Expected {self.cfg.n_inputs} inputs, got {len(features)}"
        )

        # Compute all MF memberships: mu_all[i, j] = mu_j(x_i)  shape (8, 3)
        mu_all = np.stack(
            [mf.evaluate(float(features[i])) for i, mf in enumerate(self.input_mfs)],
            axis=0,
        ).astype(np.float32)  # (8, 3)

        # Compute firing strengths via product T-norm
        n_rules = len(self.rule_base)
        firing = np.ones(n_rules, dtype=np.float32)
        for r_idx in range(n_rules):
            for k in range(self._max_ant):
                inp = self._ant_input[r_idx, k]
                lvl = self._ant_level[r_idx, k]
                if inp < 0:
                    break   # padded entry
                firing[r_idx] *= mu_all[inp, lvl]

        # Apply rule weights
        weighted = firing * self.rule_weights  # (n_rules,)
        total_w = float(np.sum(weighted))

        if total_w < 1e-9:
            # All rules fire with zero strength — output neutral defaults
            return FISOutput(
                lethal_score=0.0,
                artifact_score=0.0,
                class_index=0,
                class_name="Normal",
                firing_strengths=firing,
            )

        # Defuzzification: weighted average of consequents
        out = np.dot(weighted, self._consequents) / total_w  # (3,)

        lethal_score = float(np.clip(out[0], 0.0, 1.0))
        artifact_score = float(np.clip(out[1], 0.0, 1.0))
        # Class via weighted consequent (round to nearest integer)
        class_idx = int(np.clip(round(float(out[2])), 0, 4))

        return FISOutput(
            lethal_score=lethal_score,
            artifact_score=artifact_score,
            class_index=class_idx,
            class_name=FISOutput.CLASS_NAMES[class_idx],
            firing_strengths=firing,
        )

    # ------------------------------------------------------------------
    def update_params(
        self,
        mf_params: np.ndarray,
        rule_weights: np.ndarray,
    ) -> None:
        """
        Update MF parameters and rule weights (called by PSO).
        Re-builds InputMFs from the new parameter matrix.
        """
        self.mf_params = np.asarray(mf_params, dtype=np.float32)
        self.input_mfs = build_input_mfs(self.mf_params)
        self.rule_weights = np.asarray(rule_weights, dtype=np.float32)
