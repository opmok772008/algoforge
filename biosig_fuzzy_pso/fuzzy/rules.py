"""
fuzzy/rules.py — Clinically motivated Sugeno rule base.

Rule format: (antecedent_indices, consequent_values, description)
  antecedent_indices : list of (input_idx, mf_level)
      input_idx : 0–7 (matches FiducialFeatures order)
      mf_level  : 0=Low, 1=Med, 2=High
  consequent_values : [z_lethal, z_artifact, z_class]
      z_lethal  : crisp output for lethal_score (Sugeno zero-order)
      z_artifact: crisp output for artifact_score
      z_class   : crisp class index (0=Normal,1=VT,2=VF,3=Artifact,4=Other)

Rule firing strength: w_r = product_i mu_ri(x_i)   (product T-norm)
Defuzzification:      y = sum_r(w_r * z_r) / sum_r(w_r)  (weighted average)

INPUT INDICES:
  0: heart_rate (HR)         — Low:<50 bpm norm, Med:50–150, High:>150
  1: rr_cv                   — Low:regular, Med:moderate, High:irregular
  2: qrs_width               — Low:narrow, Med:moderate, High:wide
  3: r_amp_var               — Low:stable, Med:moderate, High:variable
  4: dom_freq                — Low:<3Hz, Med:3–8Hz, High:>8Hz
  5: spectral_conc (VF band) — Low:<20%, Med:20–60%, High:>60%
  6: st_deviation            — Low:<0.1mV norm, Med:0.1–0.3, High:>0.3
  7: sqi                     — Low:poor quality, Med:moderate, High:good
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# MF level aliases for readability
L, M, H = 0, 1, 2

# Class indices
NORMAL = 0
VT = 1
VF = 2
ARTIFACT = 3
OTHER = 4


@dataclass
class FuzzyRule:
    """One Sugeno zero-order fuzzy rule."""
    antecedent: List[Tuple[int, int]]   # [(input_idx, mf_level), ...]
    consequent: List[float]             # [z_lethal, z_artifact, z_class]
    weight: float                       # rule weight (tuned by PSO)
    description: str                    # clinical rationale


# ---------------------------------------------------------------------------
# Rule base (≥15 clinically motivated rules)
# ---------------------------------------------------------------------------

RULE_BASE: List[FuzzyRule] = [

    # ── R1: Normal sinus rhythm ──────────────────────────────────────────
    FuzzyRule(
        antecedent=[(0, L), (1, L), (2, L), (7, H)],    # Low HR, Low RRCV, Narrow QRS, Good SQI
        consequent=[0.0, 0.0, NORMAL],
        weight=1.0,
        description="Low HR + regular RR + narrow QRS + good SQI => Normal",
    ),

    # ── R2: Ventricular Tachycardia (VT) — classic ───────────────────────
    FuzzyRule(
        antecedent=[(0, H), (2, H), (1, L)],             # High HR + Wide QRS + Regular
        consequent=[0.95, 0.0, VT],
        weight=1.0,
        description="High HR + wide QRS + regular RR => VT (classic signature)",
    ),

    # ── R3: VT with moderate HR variability ──────────────────────────────
    FuzzyRule(
        antecedent=[(0, H), (2, H), (1, M)],
        consequent=[0.80, 0.0, VT],
        weight=0.9,
        description="High HR + wide QRS + moderate RR variability => likely VT",
    ),

    # ── R4: Ventricular Fibrillation — spectral signature ────────────────
    FuzzyRule(
        antecedent=[(5, H), (1, H), (3, H)],             # High spectral_conc + Irregular RR + High R-var
        consequent=[0.98, 0.0, VF],
        weight=1.0,
        description="High VF-band concentration + irregular RR + high R-amp variability => VF",
    ),

    # ── R5: VF — no distinct QRS + dominant freq in VF band ──────────────
    FuzzyRule(
        antecedent=[(5, H), (4, M), (2, L)],             # High spectral_conc + mid dom_freq + narrow/absent QRS
        consequent=[0.95, 0.0, VF],
        weight=1.0,
        description="High VF-band + VF-range dominant freq + no distinct QRS => VF",
    ),

    # ── R6: VF with low SQI (but lethal override still fires) ────────────
    FuzzyRule(
        antecedent=[(5, H), (1, H), (7, L)],
        consequent=[0.90, 0.3, VF],
        weight=0.85,
        description="VF spectral + irregular RR + poor signal quality => likely VF (low SQI hedge)",
    ),

    # ── R7: EMG artifact — high-freq noise, low SQI ──────────────────────
    FuzzyRule(
        antecedent=[(7, L), (4, H), (5, L)],             # Poor SQI + high dom_freq + no VF spectral conc
        consequent=[0.0, 0.95, ARTIFACT],
        weight=1.0,
        description="Poor SQI + high dominant freq (EMG) + no VF band => Artifact",
    ),

    # ── R8: Artifact with motion-like baseline ────────────────────────────
    FuzzyRule(
        antecedent=[(7, L), (4, L), (5, L), (1, M)],     # Poor SQI + low freq noise + no VF + moderate RR var
        consequent=[0.0, 0.85, ARTIFACT],
        weight=0.9,
        description="Poor SQI + low dom_freq (motion) + no VF spectral => Motion artifact",
    ),

    # ── R9: ST elevation — possible STEMI or LV overload ─────────────────
    FuzzyRule(
        antecedent=[(6, H), (0, L), (2, L), (7, H)],     # High ST + normal HR + narrow QRS + good SQI
        consequent=[0.4, 0.0, OTHER],
        weight=0.8,
        description="High ST deviation + normal HR + narrow QRS => STEMI/ischemia (Other)",
    ),

    # ── R10: Sinus tachycardia (non-lethal) ──────────────────────────────
    FuzzyRule(
        antecedent=[(0, H), (2, L), (1, L)],             # High HR + Narrow QRS + Regular RR
        consequent=[0.1, 0.0, NORMAL],
        weight=0.85,
        description="High HR + narrow QRS + regular RR => Sinus tachycardia (non-lethal)",
    ),

    # ── R11: Moderate quality, medium features — Other ───────────────────
    FuzzyRule(
        antecedent=[(7, M), (0, M), (2, M)],
        consequent=[0.2, 0.2, OTHER],
        weight=0.6,
        description="Moderate quality + medium HR + medium QRS => uncertain (Other)",
    ),

    # ── R12: VT with amplitude collapse (R-amp decreasing) ───────────────
    FuzzyRule(
        antecedent=[(0, H), (2, H), (3, H)],             # Fast + wide + variable amplitude
        consequent=[0.92, 0.0, VT],
        weight=0.95,
        description="High HR + wide QRS + high R-amp variability => VT with deterioration",
    ),

    # ── R13: Artifact mimicking fast rhythm ──────────────────────────────
    FuzzyRule(
        antecedent=[(7, L), (0, H), (5, L)],             # Poor SQI + apparent high HR + no VF spec
        consequent=[0.05, 0.80, ARTIFACT],
        weight=0.9,
        description="Poor SQI + apparent high HR + no VF spectrum => likely artifact (not lethal)",
    ),

    # ── R14: VF transitioning — medium spectral concentration ────────────
    FuzzyRule(
        antecedent=[(5, M), (1, H), (3, H)],             # Med spectral + irregular + variable
        consequent=[0.70, 0.1, VF],
        weight=0.75,
        description="Medium VF-band + irregular RR + variable amplitude => early/transitional VF",
    ),

    # ── R15: PVCs / ectopic beats ────────────────────────────────────────
    FuzzyRule(
        antecedent=[(2, H), (1, M), (0, L)],             # Wide QRS + moderate RR variability + low HR
        consequent=[0.25, 0.0, OTHER],
        weight=0.7,
        description="Wide QRS + moderate irregularity + low HR => PVCs/ectopic (Other)",
    ),

    # ── R16: Normal rhythm with artefact present ─────────────────────────
    FuzzyRule(
        antecedent=[(0, L), (2, L), (7, M)],             # Normal HR + narrow QRS + medium SQI
        consequent=[0.0, 0.4, NORMAL],
        weight=0.7,
        description="Low HR + narrow QRS + medium SQI => Normal with mild artifact",
    ),

    # ── R17: VT with good signal quality (high confidence) ───────────────
    FuzzyRule(
        antecedent=[(0, H), (2, H), (7, H)],
        consequent=[0.97, 0.0, VT],
        weight=1.0,
        description="High HR + wide QRS + good SQI (high confidence) => VT",
    ),

    # ── R18: Clean signal, regular, normal HR ────────────────────────────
    FuzzyRule(
        antecedent=[(7, H), (0, L), (1, L), (5, L)],
        consequent=[0.0, 0.0, NORMAL],
        weight=1.0,
        description="Good SQI + low HR + regular RR + no VF band => clearly Normal",
    ),
]


def get_rule_weights() -> np.ndarray:
    """Return rule weights as a float32 array (shape: n_rules)."""
    return np.array([r.weight for r in RULE_BASE], dtype=np.float32)


def get_rule_consequents() -> np.ndarray:
    """Return consequent matrix (n_rules, 3) float32: [z_lethal, z_artifact, z_class]."""
    return np.array([r.consequent for r in RULE_BASE], dtype=np.float32)
