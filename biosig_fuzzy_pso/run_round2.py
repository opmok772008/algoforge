#!/usr/bin/env python3
"""
run_round2.py — Execute Round 2 EMG-shift adaptation benchmark.

Loads the best_params.json from Round 1 (or runs Round 1 first if missing),
then applies severe EMG perturbation and warm-start PSO re-optimisation.

Usage:
    python run_round2.py [--iter N] [--swarm S] [--seed K]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from biosig_fuzzy_pso.config import DEFAULT_CONFIG
from biosig_fuzzy_pso.eval.benchmark import run_round1, run_round2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 2 EMG shift + warm-start PSO")
    parser.add_argument("--iter", type=int, default=8)
    parser.add_argument("--swarm", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=20.0)
    args = parser.parse_args()

    pso_cfg = replace(
        DEFAULT_CONFIG.pso,
        max_iter=args.iter,
        swarm_size=args.swarm,
        seed=args.seed,
    )

    # Load or generate Round 1 best
    best_path = "results/best_params.json"
    if os.path.exists(best_path):
        print(f"[ROUND 2] Loading Round 1 best from {best_path}")
        with open(best_path) as f:
            data = json.load(f)
        round1_best = np.array(data["particle"], dtype=np.float32)
    else:
        print("[ROUND 2] Round 1 results not found. Running Round 1 first...")
        round1_best, _ = run_round1(pso_cfg=pso_cfg, seed=args.seed,
                                     duration_per_class_sec=args.duration)

    unadapted, adapted = run_round2(
        round1_best=round1_best,
        pso_cfg=pso_cfg,
        seed=args.seed,
        duration_per_class_sec=args.duration,
    )

    print("\n" + "=" * 60)
    print("ROUND 2 COMPARISON — Un-adapted vs Adapted on Perturbed Test")
    print("=" * 60)
    print(f"{'Metric':<30} {'Unadapted':>12} {'Adapted':>12}")
    print("-" * 54)
    print(f"{'Se':<30} {unadapted.Se:>12.4f} {adapted.Se:>12.4f}")
    print(f"{'Sp':<30} {unadapted.Sp:>12.4f} {adapted.Sp:>12.4f}")
    print(f"{'lethal_Se':<30} {unadapted.lethal_Se:>12.4f} {adapted.lethal_Se:>12.4f}")
    print(f"{'FAFI':<30} {unadapted.FAFI:>12.4f} {adapted.FAFI:>12.4f}")
    print(f"{'R-jitter (ms)':<30} {unadapted.r_jitter_ms:>12.2f} {adapted.r_jitter_ms:>12.2f}")
    print(f"{'dSNR (dB)':<30} {unadapted.dSNR_db:>12.2f} {adapted.dSNR_db:>12.2f}")
    print(f"{'ms/window':<30} {unadapted.ms_per_window:>12.2f} {adapted.ms_per_window:>12.2f}")
    print(f"{'Peak memory (KB)':<30} {unadapted.peak_memory_kb:>12.1f} {adapted.peak_memory_kb:>12.1f}")
    print("=" * 60)

    # Append to ATTEMPT_LOG
    _append_attempt_log(unadapted, adapted)


def _append_attempt_log(unadapted, adapted) -> None:
    import time
    path = "ATTEMPT_LOG.md"
    with open(path, "a", encoding="utf-8") as f:
        ts = time.strftime("%Y-%m-%dT%H:%M")
        f.write(
            f"| {ts} | R2 (unadapted) "
            f"| (see results/convergence_round2.csv) "
            f"| {unadapted.Se:.3f} | {unadapted.Sp:.3f} "
            f"| {unadapted.FAFI:.3f} | {unadapted.r_jitter_ms:.1f} "
            f"| {unadapted.dSNR_db:.1f} | {unadapted.ms_per_window:.1f} "
            f"| What changed and why: EMG perturbation applied; Round 1 params used without adaptation |\n"
        )
        f.write(
            f"| {ts} | R2 (adapted) "
            f"| (see results/convergence_round2.csv) "
            f"| {adapted.Se:.3f} | {adapted.Sp:.3f} "
            f"| {adapted.FAFI:.3f} | {adapted.r_jitter_ms:.1f} "
            f"| {adapted.dSNR_db:.1f} | {adapted.ms_per_window:.1f} "
            f"| What changed and why: warm-start PSO with 30% reinit for robustness to EMG shift |\n"
        )


if __name__ == "__main__":
    main()
