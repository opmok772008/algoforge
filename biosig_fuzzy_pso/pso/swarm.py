"""
pso/swarm.py — PSO engine with inertia-weight scheduling, velocity clamping,
               reflective boundary handling, and per-iteration logging.

Update equations:
  v_i(t+1) = w(t)*v_i(t) + c1*r1*(pbest_i - x_i(t)) + c2*r2*(gbest - x_i(t))
  x_i(t+1) = x_i(t) + v_i(t+1)
  w(t) = w_max - (w_max - w_min) * t / T     (linear decay)

Velocity clamping: |v| <= v_max = vel_clamp_frac * (ub - lb)
Boundary: reflective (bounce off bounds)
"""

from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from biosig_fuzzy_pso.config import PSOConfig, DEFAULT_CONFIG
from biosig_fuzzy_pso.pso.encoding import LOWER_BOUNDS, UPPER_BOUNDS, PARTICLE_DIM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Iteration log entry
# ---------------------------------------------------------------------------

@dataclass
class IterationLog:
    iteration: int
    gbest_fitness: float
    mean_fitness: float
    swarm_diversity: float
    feasible_count: int
    elapsed_s: float


# ---------------------------------------------------------------------------
# PSO result
# ---------------------------------------------------------------------------

@dataclass
class PSOResult:
    gbest_position: np.ndarray
    gbest_fitness: float
    convergence: List[IterationLog]
    total_iterations: int
    total_time_s: float


# ---------------------------------------------------------------------------
# PSO swarm
# ---------------------------------------------------------------------------

class PSOSwarm:
    """
    Particle Swarm Optimiser.

    Parameters
    ----------
    fitness_fn : callable (x: np.ndarray) -> (float, dict)
    cfg        : PSOConfig
    lb, ub     : (D,) float32 lower/upper bounds
    seed       : RNG seed
    """

    def __init__(
        self,
        fitness_fn: Callable[[np.ndarray], Tuple[float, dict]],
        cfg: PSOConfig = DEFAULT_CONFIG.pso,
        lb: np.ndarray = LOWER_BOUNDS,
        ub: np.ndarray = UPPER_BOUNDS,
        seed: Optional[int] = None,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.cfg = cfg
        self.lb = lb.astype(np.float32)
        self.ub = ub.astype(np.float32)
        self.dim = PARTICLE_DIM
        self.rng = np.random.default_rng(seed if seed is not None else cfg.seed)

        # Velocity clamp
        self.v_max = self.cfg.vel_clamp_frac * (self.ub - self.lb)

        # Swarm state (pre-allocated)
        n = cfg.swarm_size
        self.positions = np.empty((n, self.dim), dtype=np.float32)
        self.velocities = np.empty((n, self.dim), dtype=np.float32)
        self.pbest_pos = np.empty((n, self.dim), dtype=np.float32)
        self.pbest_fit = np.full(n, np.inf, dtype=np.float64)
        self.gbest_pos = np.empty(self.dim, dtype=np.float32)
        self.gbest_fit = np.inf
        self._metrics: List[dict] = [{}] * n

        self._initialised = False

    # ------------------------------------------------------------------
    def initialise(self, seed_position: Optional[np.ndarray] = None) -> None:
        """
        Random initialisation. If seed_position is given, particle 0 is
        placed there (warm-start for Round 2).
        """
        n = self.cfg.swarm_size
        # Uniform random positions
        uniform = self.rng.uniform(0, 1, (n, self.dim)).astype(np.float32)
        self.positions[:] = self.lb + uniform * (self.ub - self.lb)

        if seed_position is not None:
            self.positions[0] = np.clip(seed_position, self.lb, self.ub)

        # Uniform random velocities in [-v_max, v_max]
        vel_init = self.rng.uniform(-1, 1, (n, self.dim)).astype(np.float32)
        self.velocities[:] = vel_init * self.v_max

        self.pbest_pos[:] = self.positions
        self.pbest_fit[:] = np.inf
        self.gbest_fit = np.inf
        self._initialised = True

    def reinit_fraction(self, seed_position: np.ndarray) -> None:
        """
        Round-2 warm start: keep particle 0 at seed_position, randomly
        re-initialise `reinit_fraction` percent of the rest.
        """
        n = self.cfg.swarm_size
        n_reinit = max(1, int(n * self.cfg.reinit_fraction))
        # Particle 0 = gbest from Round 1
        self.positions[0] = np.clip(seed_position, self.lb, self.ub)
        self.pbest_pos[0] = self.positions[0]
        self.pbest_fit[0] = np.inf   # will be re-evaluated

        # Re-init a random subset
        reinit_idx = self.rng.choice(n - 1, size=n_reinit, replace=False) + 1
        uniform = self.rng.uniform(0, 1, (n_reinit, self.dim)).astype(np.float32)
        self.positions[reinit_idx] = self.lb + uniform * (self.ub - self.lb)
        self.velocities[reinit_idx] = (
            self.rng.uniform(-1, 1, (n_reinit, self.dim)).astype(np.float32)
            * self.v_max
        )
        self.pbest_fit[reinit_idx] = np.inf

    # ------------------------------------------------------------------
    def run(
        self,
        max_iter: Optional[int] = None,
        log_path: Optional[str] = None,
    ) -> PSOResult:
        """
        Execute PSO optimisation.

        Parameters
        ----------
        max_iter : override cfg.max_iter
        log_path : CSV file path for iteration log (None = no file)

        Returns
        -------
        PSOResult
        """
        if not self._initialised:
            self.initialise()

        T = max_iter if max_iter is not None else self.cfg.max_iter
        convergence: List[IterationLog] = []
        t_total_start = time.perf_counter()

        # Prepare CSV log
        csv_file = None
        csv_writer = None
        if log_path:
            os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
            csv_file = open(log_path, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "iteration", "gbest_fitness", "mean_fitness",
                "swarm_diversity", "feasible_count", "elapsed_s"
            ])

        try:
            for t in range(T):
                t_iter_start = time.perf_counter()
                w_t = self.cfg.w_max - (self.cfg.w_max - self.cfg.w_min) * t / max(T - 1, 1)

                # Evaluate all particles
                fitnesses = np.empty(self.cfg.swarm_size, dtype=np.float64)
                all_metrics = []
                for i in range(self.cfg.swarm_size):
                    f, m = self.fitness_fn(self.positions[i])
                    fitnesses[i] = f
                    all_metrics.append(m)

                    # Update personal best
                    if f < self.pbest_fit[i]:
                        self.pbest_fit[i] = f
                        self.pbest_pos[i] = self.positions[i].copy()

                    # Update global best
                    if f < self.gbest_fit:
                        self.gbest_fit = f
                        self.gbest_pos = self.positions[i].copy()
                        self._metrics[i] = m

                # Compute diversity = mean std across dimensions
                diversity = float(np.mean(np.std(self.positions, axis=0)))
                feasible = int(np.sum([m.get("lethal_Se", 0.0) == 1.0 for m in all_metrics]))

                elapsed = time.perf_counter() - t_total_start
                log_entry = IterationLog(
                    iteration=t,
                    gbest_fitness=self.gbest_fit,
                    mean_fitness=float(np.mean(fitnesses)),
                    swarm_diversity=diversity,
                    feasible_count=feasible,
                    elapsed_s=elapsed,
                )
                convergence.append(log_entry)

                # Console log
                logger.info(
                    "Iter %3d | gbest=%.4f | mean=%.4f | diversity=%.4f | feasible=%d",
                    t, self.gbest_fit, float(np.mean(fitnesses)), diversity, feasible,
                )
                print(
                    f"[PSO] iter={t:3d}  gbest={self.gbest_fit:.4f}  "
                    f"mean={float(np.mean(fitnesses)):.4f}  "
                    f"div={diversity:.4f}  feasible={feasible}/{self.cfg.swarm_size}  "
                    f"t={elapsed:.1f}s",
                    flush=True,
                )

                # CSV log
                if csv_writer:
                    csv_writer.writerow([
                        t, f"{self.gbest_fit:.6f}", f"{float(np.mean(fitnesses)):.6f}",
                        f"{diversity:.6f}", feasible, f"{elapsed:.2f}",
                    ])
                    csv_file.flush()  # type: ignore[union-attr]

                # Update velocities and positions
                for i in range(self.cfg.swarm_size):
                    r1 = self.rng.random(self.dim).astype(np.float32)
                    r2 = self.rng.random(self.dim).astype(np.float32)

                    cognitive = self.cfg.c1 * r1 * (self.pbest_pos[i] - self.positions[i])
                    social = self.cfg.c2 * r2 * (self.gbest_pos - self.positions[i])
                    self.velocities[i] = (
                        np.float32(w_t) * self.velocities[i] + cognitive + social
                    )

                    # Velocity clamping
                    np.clip(self.velocities[i], -self.v_max, self.v_max, out=self.velocities[i])

                    # Position update
                    self.positions[i] += self.velocities[i]

                    # Reflective boundary handling
                    self._reflect(i)

        finally:
            if csv_file:
                csv_file.close()

        total_time = time.perf_counter() - t_total_start
        return PSOResult(
            gbest_position=self.gbest_pos.copy(),
            gbest_fitness=float(self.gbest_fit),
            convergence=convergence,
            total_iterations=T,
            total_time_s=total_time,
        )

    # ------------------------------------------------------------------
    def _reflect(self, i: int) -> None:
        """Reflective boundary: bounce off lb/ub."""
        pos = self.positions[i]
        vel = self.velocities[i]
        # Lower bound
        below = pos < self.lb
        pos[below] = 2 * self.lb[below] - pos[below]
        vel[below] = -vel[below]
        # Upper bound
        above = pos > self.ub
        pos[above] = 2 * self.ub[above] - pos[above]
        vel[above] = -vel[above]
        # Final clamp (handles double-bounce edge cases)
        np.clip(pos, self.lb, self.ub, out=pos)
