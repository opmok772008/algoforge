"""
api/schemas.py — Pydantic schemas for the Frontend Adapter Contract.
Specifies the exact JSON schema matching frontend/src/data/dataset.ts field for field.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Meta & Budgets
# ---------------------------------------------------------------------------

class Budgets(BaseModel):
    latency_s: float = 3.0
    memory_kb: float = 2048.0
    ms_per_window: float = 20.0


class Meta(BaseModel):
    seed: int
    data_source: Literal["physionet", "synthetic"]
    fs: int
    algorithm: Literal["PSO"] = "PSO"
    population_size: int
    generations: int
    budgets: Budgets


# ---------------------------------------------------------------------------
# Metrics & Record Evaluation
# ---------------------------------------------------------------------------

class Metrics(BaseModel):
    fitness: float
    se: float
    sp: float
    ppv: float
    lethal_se: float
    fafi: float
    jitter_ms: float
    dsnr_db: float
    ms_per_window: float
    peak_kb: float


class PerRecord(BaseModel):
    record: str
    se: float
    sp: float


# ---------------------------------------------------------------------------
# Ledger (Hard Constraints C1-C4)
# ---------------------------------------------------------------------------

class LedgerItem(BaseModel):
    id: Literal["C1", "C2", "C3", "C4"]
    label: str
    measured: float
    threshold: float
    unit: str
    comparator: str
    pass_: bool = Field(..., alias="pass")
    detail: str
    measured_secondary: Optional[float] = None
    threshold_secondary: Optional[float] = None
    unit_secondary: Optional[str] = None
    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# SignalPayload & Fiducials
# ---------------------------------------------------------------------------

class RPeakItem(BaseModel):
    idx: int
    t_s: float
    ref_t_s: float
    error_ms: float


class QRSItem(BaseModel):
    onset_idx: int
    offset_idx: int
    width_ms: float


class STItem(BaseModel):
    onset_idx: int
    offset_idx: int
    level_mv: float
    ref_level_mv: float


class ZonesItem(BaseModel):
    wander: List[List[float]] = []
    emg: List[List[float]] = []
    mains: List[List[float]] = []


class RhythmEpisodeItem(BaseModel):
    label: Literal["NSR", "VT", "VF"]
    onset_s: float
    offset_s: float


class LiveMetrics(BaseModel):
    jitter_ms: float
    dsnr_db: float
    qrs_amp_distortion_pct: float
    st_error_mv: float


class SignalPayload(BaseModel):
    fs: int
    t0_s: float
    raw: List[float]
    filtered: List[float]
    reference: Optional[List[float]] = None
    rpeaks: List[RPeakItem]
    qrs: List[QRSItem]
    st: List[STItem]
    zones: ZonesItem
    rhythm_episodes: List[RhythmEpisodeItem]
    live_metrics: LiveMetrics


# ---------------------------------------------------------------------------
# Stress Grid
# ---------------------------------------------------------------------------

class StressGridItem(BaseModel):
    wander_amp_mv: float
    emg_snr_db: float
    signal: SignalPayload


# ---------------------------------------------------------------------------
# Alarms
# ---------------------------------------------------------------------------

class AlarmEvent(BaseModel):
    id: str
    type: Literal["VF", "VT", "other"]
    lethal: bool
    onset_s: float
    alert_s: Optional[float] = None
    latency_s: Optional[float] = None
    suppressed: bool
    cause_if_suppressed: Optional[str] = None


class AlarmRatePerHour(BaseModel):
    before_filter: float
    after_filter: float


class Alarms(BaseModel):
    budget_s: float = 3.0
    events: List[AlarmEvent]
    rate_per_hour: AlarmRatePerHour
    true_events_total: int
    true_events_preserved: int
    fafi: float


# ---------------------------------------------------------------------------
# Classification & ROC
# ---------------------------------------------------------------------------

class OperatingPoint(BaseModel):
    fpr: float
    tpr: float
    threshold: float


class ROC(BaseModel):
    fpr: List[float]
    tpr: List[float]
    auc: float
    thresholds: List[float]
    operating_point: OperatingPoint


class Classification(BaseModel):
    labels: List[str] = ["Normal", "VT", "VF", "Artifact", "Other"]
    confusion: List[List[int]]
    roc: ROC
    roc_target: Literal["lethal_vs_non_lethal"] = "lethal_vs_non_lethal"


# ---------------------------------------------------------------------------
# Optimization, Pareto, Snapshots & Stability
# ---------------------------------------------------------------------------

class ConvergenceItem(BaseModel):
    generation: int
    best: float
    mean: float
    worst: float
    diversity: float
    feasible_count: int


class ParetoItem(BaseModel):
    id: str
    se: float
    sp: float
    fafi: float
    jitter_ms: float
    dsnr_db: float
    ms_per_window: float
    fitness: float
    feasible: bool
    selected: bool


class MFItem(BaseModel):
    label: Literal["low", "med", "high"]
    center: float
    sigma: float


class SnapshotItem(BaseModel):
    generation: int
    mf: Dict[str, List[MFItem]]
    rule_weights: List[float]


class Stability(BaseModel):
    seeds: List[int]
    fitness_mean: float
    fitness_std: float


class Optimization(BaseModel):
    convergence: List[ConvergenceItem]
    pareto: List[ParetoItem]
    selected_id: str
    snapshots: List[SnapshotItem]
    stability: Optional[Stability] = None


# ---------------------------------------------------------------------------
# Fuzzy Rules
# ---------------------------------------------------------------------------

class FuzzyRuleFiringItem(BaseModel):
    id: str
    if_text: str
    then_text: str
    weight: float
    firing_strength: float
    clinical_note: str


class FuzzyExampleItem(BaseModel):
    record: str
    t_s: float
    rules: List[FuzzyRuleFiringItem]


class FuzzyRulesGenItem(BaseModel):
    generation: int
    examples: List[FuzzyExampleItem]


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

class ResourceStageItem(BaseModel):
    name: Literal["baseline", "nlms", "rpeak", "features", "fis", "gate"]
    kb: float
    ms: float


class TraceKBItem(BaseModel):
    t_s: float
    kb: float


class Resources(BaseModel):
    peak_kb: float
    budget_kb: float = 2048.0
    ms_per_window: float
    budget_ms: float = 50.0
    stages: List[ResourceStageItem]
    trace_kb_over_time: List[TraceKBItem]


# ---------------------------------------------------------------------------
# Attempt & Dataset
# ---------------------------------------------------------------------------

class Attempt(BaseModel):
    id: Literal["round1", "round2", "final"]
    step: Literal[1, 2, 3]
    title: str
    scenario: Literal["clean_rhythm", "severe_emg_burst", "jury_defence"]
    what_changed_and_why: Optional[str] = None
    metrics: Metrics
    per_record: List[PerRecord]
    unadapted: Optional[Metrics] = None
    ledger: List[LedgerItem]
    signal: SignalPayload
    stress_grid: List[StressGridItem]
    alarms: Alarms
    classification: Classification
    optimization: Optimization
    fuzzy_rules: List[FuzzyRulesGenItem]
    resources: Resources


class Dataset(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    meta: Meta
    attempts: List[Attempt]


# ---------------------------------------------------------------------------
# Live Stress Test API Request
# ---------------------------------------------------------------------------

class SignalStressRequest(BaseModel):
    wander_amp_mv: float = 0.0
    emg_snr_db: float = 20.0
    mains: bool = False
    seed: int = 42
    attempt: str = "final"
