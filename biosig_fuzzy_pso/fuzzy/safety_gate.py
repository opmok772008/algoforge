"""
fuzzy/safety_gate.py — Asymmetric alarm-suppression logic (C2 compliance).

Rules (hard-wired, NOT tunable by PSO):
  1. IF lethal_score >= tau_lethal  => ALWAYS alert (C2: never suppress lethal)
  2. IF artifact_score >= tau_artifact AND lethal_score < tau_lethal
     => suppress alarm (noise suppression only when safe to do so)
  3. ELSE => follow lethal_score >= 0.5 as alarm trigger

The gate is deliberately asymmetric: false suppression of a lethal event is
treated as infinitely costly, so the tau_lethal threshold takes unconditional
precedence.

Constraint C2 is guaranteed by the structure of the code, not by a learned
parameter: the `if lethal_score >= tau_lethal` branch always fires first.
"""

from __future__ import annotations

from dataclasses import dataclass

from biosig_fuzzy_pso.config import FISConfig, DEFAULT_CONFIG


@dataclass
class GateDecision:
    """Output of the safety gate."""
    alert: bool
    suppressed: bool        # True if an alarm was suppressed (not lethal)
    lethal_score: float
    artifact_score: float
    reason: str             # human-readable explanation


class SafetyGate:
    """
    Asymmetric alarm-suppression safety gate.

    Parameters
    ----------
    tau_lethal   : lethal alert threshold (tuned by PSO, but gate logic is fixed)
    tau_artifact : artifact suppression threshold
    """

    def __init__(
        self,
        tau_lethal: float = DEFAULT_CONFIG.fis.tau_lethal,
        tau_artifact: float = DEFAULT_CONFIG.fis.tau_artifact,
    ) -> None:
        self.tau_lethal = tau_lethal
        self.tau_artifact = tau_artifact

    # ------------------------------------------------------------------
    def evaluate(
        self,
        lethal_score: float,
        artifact_score: float,
    ) -> GateDecision:
        """
        Evaluate alarm decision.

        Parameters
        ----------
        lethal_score   : FIS lethal output [0, 1]
        artifact_score : FIS artifact output [0, 1]

        Returns
        -------
        GateDecision
        """
        # C2: Lethal override — ALWAYS alert if lethal_score >= tau_lethal
        if lethal_score >= self.tau_lethal:
            return GateDecision(
                alert=True,
                suppressed=False,
                lethal_score=lethal_score,
                artifact_score=artifact_score,
                reason=(
                    f"LETHAL ALERT: lethal_score={lethal_score:.3f} >= "
                    f"tau_lethal={self.tau_lethal:.3f}"
                ),
            )

        # Artifact suppression (only when lethal_score < tau_lethal)
        if artifact_score >= self.tau_artifact:
            return GateDecision(
                alert=False,
                suppressed=True,
                lethal_score=lethal_score,
                artifact_score=artifact_score,
                reason=(
                    f"SUPPRESSED: artifact_score={artifact_score:.3f} >= "
                    f"tau_artifact={self.tau_artifact:.3f} (lethal={lethal_score:.3f} < tau)"
                ),
            )

        # Default: alert if lethal_score crosses 0.5
        alert = lethal_score >= 0.5
        return GateDecision(
            alert=alert,
            suppressed=False,
            lethal_score=lethal_score,
            artifact_score=artifact_score,
            reason=(
                f"DEFAULT: lethal_score={lethal_score:.3f} "
                f"({'alert' if alert else 'no alert'})"
            ),
        )

    def update_thresholds(
        self, tau_lethal: float, tau_artifact: float
    ) -> None:
        """Update thresholds (called by PSO after each iteration)."""
        self.tau_lethal = float(tau_lethal)
        self.tau_artifact = float(tau_artifact)
