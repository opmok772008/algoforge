import raw from "./dataset.json";

export const dataset = raw as Dataset;

export interface Budgets {
  latency_s: number;
  memory_kb: number;
  ms_per_window: number;
}

export interface Meta {
  seed: number;
  data_source: "physionet" | "synthetic";
  fs: number;
  algorithm: "PSO";
  population_size: number;
  generations: number;
  budgets: Budgets;
}

export interface Metrics {
  fitness: number;
  se: number;
  sp: number;
  ppv: number;
  lethal_se: number;
  fafi: number;
  jitter_ms: number;
  dsnr_db: number;
  ms_per_window: number;
  peak_kb: number;
}

export interface PerRecord {
  record: string;
  se: number;
  sp: number;
}

export interface LedgerItem {
  id: "C1" | "C2" | "C3" | "C4";
  label: string;
  measured: number;
  threshold: number;
  unit: string;
  comparator: string;
  pass: boolean;
  detail: string;
  measured_secondary: number | null;
  threshold_secondary: number | null;
  unit_secondary: string | null;
}

export interface RPeakItem {
  idx: number;
  t_s: number;
  ref_t_s: number;
  error_ms: number;
}

export interface QRSItem {
  onset_idx: number;
  offset_idx: number;
  width_ms: number;
}

export interface STItem {
  onset_idx: number;
  offset_idx: number;
  level_mv: number;
  ref_level_mv: number;
}

export interface ZonesItem {
  wander: number[][];
  emg: number[][];
  mains: number[][];
}

export interface RhythmEpisodeItem {
  label: "NSR" | "VT" | "VF";
  onset_s: number;
  offset_s: number;
}

export interface LiveMetrics {
  jitter_ms: number;
  dsnr_db: number;
  qrs_amp_distortion_pct: number;
  st_error_mv: number;
}

export interface SignalPayload {
  fs: number;
  t0_s: number;
  raw: number[];
  filtered: number[];
  reference: number[] | null;
  rpeaks: RPeakItem[];
  qrs: QRSItem[];
  st: STItem[];
  zones: ZonesItem;
  rhythm_episodes: RhythmEpisodeItem[];
  live_metrics: LiveMetrics;
}

export interface StressGridItem {
  wander_amp_mv: number;
  emg_snr_db: number;
  signal: SignalPayload;
}

export interface AlarmEvent {
  id: string;
  type: "VF" | "VT" | "other";
  lethal: boolean;
  onset_s: number;
  alert_s: number | null;
  latency_s: number | null;
  suppressed: boolean;
  cause_if_suppressed: string | null;
}

export interface AlarmRatePerHour {
  before_filter: number;
  after_filter: number;
}

export interface Alarms {
  budget_s: number;
  events: AlarmEvent[];
  rate_per_hour: AlarmRatePerHour;
  true_events_total: number;
  true_events_preserved: number;
  fafi: number;
}

export interface OperatingPoint {
  fpr: number;
  tpr: number;
  threshold: number;
}

export interface ROC {
  fpr: number[];
  tpr: number[];
  auc: number;
  thresholds: number[];
  operating_point: OperatingPoint;
}

export interface Classification {
  labels: string[];
  confusion: number[][];
  roc: ROC;
  roc_target: "lethal_vs_non_lethal";
}

export interface ConvergenceItem {
  generation: number;
  best: number;
  mean: number;
  worst: number;
  diversity: number;
  feasible_count: number;
}

export interface ParetoItem {
  id: string;
  se: number;
  sp: number;
  fafi: number;
  jitter_ms: number;
  dsnr_db: number;
  ms_per_window: number;
  fitness: number;
  feasible: boolean;
  selected: boolean;
}

export interface MFItem {
  label: "low" | "med" | "high";
  center: number;
  sigma: number;
}

export interface SnapshotItem {
  generation: number;
  mf: Record<string, MFItem[]>;
  rule_weights: number[];
}

export interface Stability {
  seeds: number[];
  fitness_mean: number;
  fitness_std: number;
}

export interface Optimization {
  convergence: ConvergenceItem[];
  pareto: ParetoItem[];
  selected_id: string;
  snapshots: SnapshotItem[];
  stability: Stability | null;
}

export interface FuzzyRuleFiringItem {
  id: string;
  if_text: string;
  then_text: string;
  weight: number;
  firing_strength: number;
  clinical_note: string;
}

export interface FuzzyExampleItem {
  record: string;
  t_s: number;
  rules: FuzzyRuleFiringItem[];
}

export interface FuzzyRulesGenItem {
  generation: number;
  examples: FuzzyExampleItem[];
}

export interface ResourceStageItem {
  name: "baseline" | "nlms" | "rpeak" | "features" | "fis" | "gate";
  kb: number;
  ms: number;
}

export interface TraceKBItem {
  t_s: number;
  kb: number;
}

export interface Resources {
  peak_kb: number;
  budget_kb: number;
  ms_per_window: number;
  budget_ms: number;
  stages: ResourceStageItem[];
  trace_kb_over_time: TraceKBItem[];
}

export interface Attempt {
  id: "round1" | "round2" | "final";
  step: 1 | 2 | 3;
  title: string;
  scenario: "clean_rhythm" | "severe_emg_burst" | "jury_defence";
  what_changed_and_why: string | null;
  metrics: Metrics;
  per_record: PerRecord[];
  unadapted: Metrics | null;
  ledger: LedgerItem[];
  signal: SignalPayload;
  stress_grid: StressGridItem[];
  alarms: Alarms;
  classification: Classification;
  optimization: Optimization;
  fuzzy_rules: FuzzyRulesGenItem[];
  resources: Resources;
}

export interface Dataset {
  schema_version: "1.0";
  meta: Meta;
  attempts: [Attempt, Attempt, Attempt] | Attempt[];
}
