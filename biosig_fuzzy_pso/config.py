"""
config.py — Frozen dataclasses for ALL pipeline parameters and PSO bounds.

Every numeric constant used anywhere in the project must be sourced from
this file (no magic numbers elsewhere).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# General / reproducibility
# ---------------------------------------------------------------------------
GLOBAL_SEED: int = 42
SAMPLE_RATE: int = 360          # Hz — MIT-BIH native sample rate
DTYPE = "float32"


# ---------------------------------------------------------------------------
# Ring-buffer / streaming
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StreamConfig:
    window_sec: float = 2.0     # analysis window length in seconds
    hop_sec: float = 0.5        # hop between successive windows
    max_latency_sec: float = 3.0  # C1 hard constraint
    buffer_seconds: float = 10.0  # ring-buffer capacity

    @property
    def window_samples(self) -> int:
        return int(self.window_sec * SAMPLE_RATE)

    @property
    def hop_samples(self) -> int:
        return int(self.hop_sec * SAMPLE_RATE)

    @property
    def buffer_samples(self) -> int:
        return int(self.buffer_seconds * SAMPLE_RATE)


# ---------------------------------------------------------------------------
# DSP — baseline wander removal
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BaselineConfig:
    # Cascaded median-filter approach (C3 compliant — zero-phase)
    med_win1_ms: float = 200.0   # first median filter window  [ms]
    med_win2_ms: float = 600.0   # second median filter window [ms]
    # FIR high-pass fallback
    fir_cutoff_hz: float = 0.5
    fir_order: int = 256

    def med_win1_samples(self, fs: int = SAMPLE_RATE) -> int:
        n = int(self.med_win1_ms * 1e-3 * fs)
        return n if n % 2 == 1 else n + 1   # must be odd for scipy median_filter

    def med_win2_samples(self, fs: int = SAMPLE_RATE) -> int:
        n = int(self.med_win2_ms * 1e-3 * fs)
        return n if n % 2 == 1 else n + 1


# ---------------------------------------------------------------------------
# DSP — NLMS adaptive canceller
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NLMSConfig:
    filter_length: int = 32     # taps
    mu0: float = 0.05           # base step size (tuned by PSO)
    epsilon: float = 1e-6       # regularisation
    mu_min: float = 0.001
    mu_max: float = 0.5


# ---------------------------------------------------------------------------
# R-peak / Pan–Tompkins detector
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RPeakConfig:
    # Band-pass filter for QRS energy
    bp_low_hz: float = 5.0
    bp_high_hz: float = 15.0
    bp_order: int = 4
    # Integration window
    mw_width_ms: float = 150.0          # moving-window integrator
    # Adaptive threshold factors (tuned by PSO)
    threshold_factor_signal: float = 0.25   # fraction of running signal estimate
    threshold_factor_noise: float = 0.5     # fraction of running noise estimate
    # Refractory period
    refractory_ms: float = 200.0
    # Search-back factor
    searchback_factor: float = 0.5

    def mw_width_samples(self, fs: int = SAMPLE_RATE) -> int:
        return max(1, int(self.mw_width_ms * 1e-3 * fs))

    def refractory_samples(self, fs: int = SAMPLE_RATE) -> int:
        return int(self.refractory_ms * 1e-3 * fs)


# ---------------------------------------------------------------------------
# Fiducial-feature extraction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FiducialConfig:
    qrs_search_ms: float = 80.0     # half-window around R for QRS onset/offset
    st_offset_ms: float = 80.0      # J-point offset from R
    st_window_ms: float = 40.0      # ST measurement window
    rr_min_ms: float = 300.0        # physiological RR limits
    rr_max_ms: float = 2000.0
    spectral_nfft: int = 512
    spectral_fmin_hz: float = 3.0   # VF spectral range
    spectral_fmax_hz: float = 9.0


# ---------------------------------------------------------------------------
# Fuzzy Inference System
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FISConfig:
    n_inputs: int = 8
    n_mfs_per_input: int = 3        # Low / Med / High
    # Input names (for documentation)
    input_names: Tuple[str, ...] = (
        "heart_rate",           # 0  normalised [0,1] → [20,300] bpm
        "rr_cv",                # 1  coefficient of variation of RR
        "qrs_width",            # 2  normalised [0,1] → [40,200] ms
        "r_amp_var",            # 3  coefficient of variation of R-amplitude
        "dom_freq",             # 4  dominant spectral frequency [0,1] → [0.5,15] Hz
        "spectral_conc",        # 5  fraction of power in VF band [3–9 Hz]
        "st_deviation",         # 6  normalised |ST| [0,1] → [0,0.5] mV
        "sqi",                  # 7  signal quality index [0,1]
    )
    n_outputs: int = 3              # lethal_score, artifact_score, class_logit
    tau_lethal: float = 0.5         # safety-gate threshold (tuned by PSO)
    tau_artifact: float = 0.6


# ---------------------------------------------------------------------------
# PSO
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PSOConfig:
    swarm_size: int = 30
    max_iter: int = 80
    w_max: float = 0.9
    w_min: float = 0.4
    c1: float = 1.8             # cognitive coefficient
    c2: float = 1.8             # social coefficient
    vel_clamp_frac: float = 0.2 # v_max = frac * (ub - lb)
    # Warm-start / diversity
    reinit_fraction: float = 0.3    # Round 2: 30 % random re-init
    seed: int = GLOBAL_SEED


# ---------------------------------------------------------------------------
# Fitness weights
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FitnessWeights:
    # F = a1*(1-Se) + a2*(1-Sp) + a3*FAFI + a4*(J/J_max)
    #   + a5*(1-dSNR/dSNR_max) + a6*(t_run/t_budget) + P
    a1: float = 3.0     # sensitivity (lethal)
    a2: float = 2.0     # specificity
    a3: float = 2.0     # false-alarm fatigue
    a4: float = 1.0     # R-peak jitter
    a5: float = 1.0     # SNR improvement
    a6: float = 0.5     # compute cost
    # Normalisation references
    J_max_ms: float = 50.0      # worst-acceptable R-peak jitter
    dSNR_max_db: float = 20.0   # best-expected SNR improvement
    t_budget_ms: float = 100.0  # per-window budget
    FA_ref_per_hr: float = 10.0 # reference false-alarm rate for FAFI
    # Constraint penalty coefficient
    penalty_coeff: float = 1e3


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SyntheticConfig:
    fs: int = SAMPLE_RATE
    # Sum-of-Gaussian morphology amplitudes and timing (relative to R=0 ms)
    # (name, amplitude_mV, t_offset_ms, sigma_ms)
    p_wave:  Tuple = (0.15, -160.0, 30.0)
    q_wave:  Tuple = (-0.1,  -30.0,  8.0)
    r_wave:  Tuple = (1.0,     0.0, 10.0)
    s_wave:  Tuple = (-0.2,   30.0, 10.0)
    t_wave:  Tuple = (0.3,   150.0, 40.0)
    # Noise floor
    noise_std: float = 0.01   # mV
    # Baseline wander injector
    bw_freqs_hz: Tuple[float, ...] = (0.05, 0.15, 0.4)
    bw_amplitude_mv: float = 0.3
    # Motion artifact
    motion_freq_hz: float = 1.0
    motion_amplitude_mv: float = 0.5
    # EMG
    emg_band_low_hz: float = 20.0
    emg_band_high_hz: float = 450.0
    emg_snr_db: float = -5.0   # severe case (Round 2 stress)
    # Mains
    mains_freq_hz: float = 50.0
    mains_amplitude_mv: float = 0.1
    # Rhythm modes HR
    hr_normal_bpm: float = 75.0
    hr_vt_bpm: float = 185.0   # VT: 150–220 bpm
    hr_vf_bpm: float = 0.0     # VF: irregular, not HR-based
    vf_freq_low_hz: float = 3.0
    vf_freq_high_hz: float = 7.0
    # QRS width
    qrs_normal_ms: float = 80.0
    qrs_vt_ms: float = 140.0   # wide QRS in VT


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    results_dir: str = "results"
    best_params_file: str = "results/best_params.json"


# ---------------------------------------------------------------------------
# Evaluation / benchmarking
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalConfig:
    n_seeds: int = 5            # stability test seeds
    test_record_fraction: float = 0.3   # 30 % held-out
    max_memory_kb: float = 2048.0       # C4: 2 MB


# ---------------------------------------------------------------------------
# Composite config object (convenience)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineConfig:
    stream: StreamConfig = field(default_factory=StreamConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    nlms: NLMSConfig = field(default_factory=NLMSConfig)
    rpeak: RPeakConfig = field(default_factory=RPeakConfig)
    fiducial: FiducialConfig = field(default_factory=FiducialConfig)
    fis: FISConfig = field(default_factory=FISConfig)
    pso: PSOConfig = field(default_factory=PSOConfig)
    weights: FitnessWeights = field(default_factory=FitnessWeights)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    api: APIConfig = field(default_factory=APIConfig)
    evl: EvalConfig = field(default_factory=EvalConfig)
    seed: int = GLOBAL_SEED


# Singleton default config
DEFAULT_CONFIG = PipelineConfig()
