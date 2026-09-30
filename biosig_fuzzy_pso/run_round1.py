#!/usr/bin/env python3
"""
run_round1.py — Execute Round 1 PSO optimisation and benchmarking.

Usage:
    python run_round1.py [--iter N] [--swarm S] [--seed K]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace

# Ensure project root on path
sys.path.insert(0, os.path.dirname(__file__))

from biosig_fuzzy_pso.config import DEFAULT_CONFIG
from biosig_fuzzy_pso.eval.benchmark import run_round1, run_stability_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 1 PSO + benchmark")
    parser.add_argument("--iter", type=int, default=8, help="PSO iterations (default 8 for speed)")
    parser.add_argument("--swarm", type=int, default=8, help="Swarm size (default 8)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stability", action="store_true", help="Run stability test after Round 1")
    parser.add_argument("--duration", type=float, default=20.0, help="Sec per class in synthetic dataset")
    args = parser.parse_args()

    pso_cfg = replace(
        DEFAULT_CONFIG.pso,
        max_iter=args.iter,
        swarm_size=args.swarm,
        seed=args.seed,
    )

    best_particle, test_metrics = run_round1(
        pso_cfg=pso_cfg,
        seed=args.seed,
        duration_per_class_sec=args.duration,
    )

    print("\n[ROUND 1 COMPLETE]")
    print(f"  Best particle saved to results/best_params.json")
    print(f"  Convergence log:   results/convergence_round1.csv")
    print(f"  Convergence plot:  results/convergence_round1.png")

    if args.stability:
        from dataclasses import replace as rp
        stab_cfg = rp(DEFAULT_CONFIG.pso, max_iter=10, swarm_size=10)
        run_stability_test(n_seeds=5, pso_cfg=stab_cfg, duration_per_class_sec=10.0)

    # Append to ATTEMPT_LOG.md
    _append_attempt_log(1, test_metrics)


def _append_attempt_log(round_n: int, metrics) -> None:
    os.makedirs("results", exist_ok=True)
    path = "ATTEMPT_LOG.md"
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write("# ATTEMPT LOG\n\n")
            f.write("| Attempt | Round | Fitness | Se   | Sp   | FAFI | Jitter(ms) | dSNR(dB) | ms/win | Note |\n")
            f.write("|---------|-------|---------|------|------|------|------------|----------|--------|------|\n")
        import time
        ts = time.strftime("%Y-%m-%dT%H:%M")
        f.write(
            f"| {ts} | R{round_n} "
            f"| (see results/convergence_round{round_n}.csv) "
            f"| {metrics.Se:.3f} "
            f"| {metrics.Sp:.3f} "
            f"| {metrics.FAFI:.3f} "
            f"| {metrics.r_jitter_ms:.1f} "
            f"| {metrics.dSNR_db:.1f} "
            f"| {metrics.ms_per_window:.1f} "
            f"| Initial attempt — baseline defaults |\n"
        )


if __name__ == "__main__":
    main()
