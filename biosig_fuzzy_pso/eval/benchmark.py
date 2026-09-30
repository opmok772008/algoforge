"""
eval/benchmark.py — Round 1 and Round 2 benchmark runners.

Round 1:
  - Load data (PhysioNet or synthetic fallback)
  - Run PSO on train set
  - Evaluate best params on held-out test set
  - Save results/best_params.json and results/convergence_round1.csv

Round 2:
  - Perturb test set with severe EMG burst
  - Warm-start PSO from Round 1 gbest (reinit 30% particles)
  - Re-optimise and compare against Round 1 un-adapted on same shifted test set
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import numpy as np

from biosig_fuzzy_pso.config import DEFAULT_CONFIG, PSOConfig
from biosig_fuzzy_pso.data.loaders import load_dataset
from biosig_fuzzy_pso.data.synthetic import PerturbationInjector
from biosig_fuzzy_pso.eval.metrics import evaluate_on_records, BenchmarkMetrics
from biosig_fuzzy_pso.pso.encoding import decode_particle, default_particle
from biosig_fuzzy_pso.pso.fitness import FitnessEvaluator
from biosig_fuzzy_pso.pso.swarm import PSOSwarm

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"


def _save_best_params(best_particle: np.ndarray, round_n: int, metrics: dict) -> None:
    """Save best particle and metrics to results/best_params.json."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    params = decode_particle(best_particle)
    out = {
        "round": round_n,
        "particle": best_particle.tolist(),
        "tau_lethal": params["tau_lethal"],
        "tau_artifact": params["tau_artifact"],
        "nlms_mu0": params["nlms"].mu0,
        "nlms_filter_length": params["nlms"].filter_length,
        "baseline_win1_ms": params["baseline"].med_win1_ms,
        "baseline_win2_ms": params["baseline"].med_win2_ms,
        "rpeak_tf_signal": params["rpeak"].threshold_factor_signal,
        "rpeak_tf_noise": params["rpeak"].threshold_factor_noise,
        "metrics": metrics,
    }
    path = os.path.join(RESULTS_DIR, "best_params.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Best params saved to %s", path)


def run_round1(
    pso_cfg: PSOConfig = DEFAULT_CONFIG.pso,
    seed: int = DEFAULT_CONFIG.seed,
    duration_per_class_sec: float = 30.0,
) -> tuple[np.ndarray, BenchmarkMetrics]:
    """
    Round 1: PSO optimisation on train set + evaluation on held-out test.

    Returns
    -------
    best_particle : (D,) float32
    test_metrics  : BenchmarkMetrics
    """
    print("\n" + "=" * 60)
    print("ROUND 1 — PSO Optimisation")
    print("=" * 60)

    # --- Load data ---
    dataset = load_dataset(seed=seed, duration_per_class_sec=duration_per_class_sec)
    print(f"[DATA] source={dataset['source']}  "
          f"train={len(dataset['train'])}  test={len(dataset['test'])}")

    # --- Fitness evaluator on train set ---
    evaluator = FitnessEvaluator(records=dataset["train"])

    # --- PSO ---
    swarm = PSOSwarm(
        fitness_fn=evaluator.evaluate,
        cfg=pso_cfg,
        seed=seed,
    )
    # Seed particle 0 with default params
    swarm.initialise(seed_position=default_particle())

    log_path = os.path.join(RESULTS_DIR, "convergence_round1.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    result = swarm.run(log_path=log_path)

    print(f"\n[ROUND 1] PSO complete in {result.total_time_s:.1f}s")
    print(f"[ROUND 1] gbest fitness = {result.gbest_fitness:.4f}")

    # --- Evaluate on held-out test set ---
    print("\n[ROUND 1] Evaluating on held-out test set...")
    test_metrics = evaluate_on_records(dataset["test"], result.gbest_position)
    test_metrics.print_table()

    # --- Save ---
    _save_best_params(result.gbest_position, 1, test_metrics.to_dict())

    # --- Convergence plot ---
    _save_convergence_plot(result.convergence, "results/convergence_round1.png", "Round 1")

    return result.gbest_position, test_metrics


def run_round2(
    round1_best: np.ndarray,
    pso_cfg: PSOConfig = DEFAULT_CONFIG.pso,
    seed: int = DEFAULT_CONFIG.seed,
    duration_per_class_sec: float = 30.0,
) -> tuple[BenchmarkMetrics, BenchmarkMetrics]:
    """
    Round 2: EMG-perturbed shift test with warm-start PSO.

    Parameters
    ----------
    round1_best : best particle from Round 1

    Returns
    -------
    (unadapted_metrics, adapted_metrics)
    """
    print("\n" + "=" * 60)
    print("ROUND 2 — Perturbation Shift + Warm-Start PSO")
    print("=" * 60)

    # --- Load data ---
    dataset = load_dataset(seed=seed, duration_per_class_sec=duration_per_class_sec)

    # --- Apply severe EMG perturbation to test set ---
    inj = PerturbationInjector(cfg=DEFAULT_CONFIG.synthetic, seed=seed + 99)
    perturbed_test = []
    for rec in dataset["test"]:
        perturbed_sig = inj.emg_burst(
            rec["signal"].copy(),
            fs=rec.get("fs", 360),
            snr_db=-5.0,   # severe EMG
        )
        perturbed_rec = dict(rec)
        perturbed_rec["signal"] = perturbed_sig
        perturbed_test.append(perturbed_rec)

    print(f"[ROUND 2] Perturbed {len(perturbed_test)} test records with severe EMG (SNR=-5 dB)")

    # --- Un-adapted Round 1 solution on perturbed test ---
    print("\n[ROUND 2] Evaluating Round 1 (unadapted) on perturbed test...")
    unadapted = evaluate_on_records(perturbed_test, round1_best)
    print("[ROUND 2] Un-adapted metrics:")
    unadapted.print_table()

    # --- Warm-start PSO on perturbed train set ---
    perturbed_train = []
    for rec in dataset["train"]:
        ps = inj.emg_burst(rec["signal"].copy(), snr_db=-5.0)
        pr = dict(rec)
        pr["signal"] = ps
        perturbed_train.append(pr)

    evaluator2 = FitnessEvaluator(records=perturbed_train)
    swarm2 = PSOSwarm(
        fitness_fn=evaluator2.evaluate,
        cfg=pso_cfg,
        seed=seed + 100,
    )
    # Initialise from Round 1 gbest, reinit 30% randomly
    swarm2.initialise(seed_position=round1_best)
    swarm2.reinit_fraction(seed_position=round1_best)

    log_path = os.path.join(RESULTS_DIR, "convergence_round2.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    result2 = swarm2.run(log_path=log_path)

    print(f"\n[ROUND 2] PSO complete in {result2.total_time_s:.1f}s")
    print(f"[ROUND 2] gbest fitness = {result2.gbest_fitness:.4f}")

    # --- Evaluate adapted solution ---
    print("\n[ROUND 2] Evaluating adapted solution on perturbed test...")
    adapted = evaluate_on_records(perturbed_test, result2.gbest_position)
    print("[ROUND 2] Adapted metrics:")
    adapted.print_table()

    # Save Round 2 best
    _save_best_params(result2.gbest_position, 2, adapted.to_dict())
    _save_convergence_plot(result2.convergence, "results/convergence_round2.png", "Round 2")

    return unadapted, adapted


def run_stability_test(
    n_seeds: int = 5,
    pso_cfg: Optional[PSOConfig] = None,
    duration_per_class_sec: float = 20.0,
) -> dict:
    """
    Stability test: repeat Round 1 PSO with n_seeds different seeds.
    Report mean ± std of final fitness.
    """
    print("\n" + "=" * 60)
    print("STABILITY TEST")
    print("=" * 60)

    if pso_cfg is None:
        from dataclasses import replace
        pso_cfg = replace(DEFAULT_CONFIG.pso, max_iter=3, swarm_size=5)

    dataset = load_dataset(seed=DEFAULT_CONFIG.seed, duration_per_class_sec=duration_per_class_sec)
    ev = FitnessEvaluator(records=dataset["train"], max_windows_per_record=10)

    fitnesses = []
    for s in range(n_seeds):
        seed_s = 100 + s * 7
        swarm = PSOSwarm(fitness_fn=ev.evaluate, cfg=pso_cfg, seed=seed_s)
        swarm.initialise(seed_position=default_particle())
        res = swarm.run()
        fitnesses.append(res.gbest_fitness)
        print(f"  seed={seed_s}  gbest_fitness={res.gbest_fitness:.4f}")

    arr = np.array(fitnesses)
    print(f"\nStability: mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")
    return {"mean": float(arr.mean()), "std": float(arr.std()), "fitnesses": fitnesses}


# ---------------------------------------------------------------------------
# Convergence plot helper
# ---------------------------------------------------------------------------

def _save_convergence_plot(convergence, path: str, title: str) -> None:
    """Save convergence curve as PNG using matplotlib (offline)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        iters = [e.iteration for e in convergence]
        gbest = [e.gbest_fitness for e in convergence]
        mean_f = [e.mean_fitness for e in convergence]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(iters, gbest, "b-", linewidth=2, label="gbest")
        ax.plot(iters, mean_f, "r--", linewidth=1, alpha=0.6, label="mean")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness")
        ax.set_title(f"PSO Convergence — {title}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Convergence plot saved: {path}")
    except Exception as e:
        print(f"[PLOT] Could not save plot ({e})")
