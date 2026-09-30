# EXPLANATION.md — BioSig Fuzzy-PSO ICU Pipeline

## (a) Representation — Particle Encoding

Each PSO particle is a flat **float32 vector of dimension D = 74**, encoding the entire
parameterisation of the pipeline:

| Offset | Count | Description |
|--------|-------|-------------|
| 0–47   | 48    | MF parameters: 8 inputs × 6 params (low\_c, low\_σ, med\_c, med\_σ, high\_c, high\_σ) |
| 48–65  | 18    | Rule weights (one per Sugeno rule) |
| 66     | 1     | NLMS base step size μ₀ |
| 67     | 1     | NLMS filter length (discretised to nearest int) |
| 68–69  | 2     | Baseline median-filter windows (ms) |
| 70–71  | 2     | R-peak adaptive threshold factors |
| 72     | 1     | τ\_lethal (safety-gate threshold) |
| 73     | 1     | τ\_artifact (suppression threshold) |

This joint encoding allows PSO to co-optimise the DSP front-end, the fuzzy classification
layer, and the alarm-decision policy simultaneously, treating the entire pipeline as a
single black-box objective.

**Bounds enforcement**: reflective boundary handling prevents particles from drifting
outside the physiologically motivated bounds defined in `pso/encoding.py`.  
MF ordering constraints (low\_c < med\_c < high\_c) are enforced at decode time.

---

## (b) Operators and Rules

### PSO Update Equations

```
v_i(t+1) = w(t)·v_i(t) + c1·r1·(pbest_i − x_i(t)) + c2·r2·(gbest − x_i(t))
x_i(t+1) = x_i(t) + v_i(t+1)
w(t)      = w_max − (w_max − w_min) · t / T          (linear inertia decay)
```

Parameters: w\_max=0.9, w\_min=0.4, c1=c2=1.8, T=80 (configurable).  
Velocity clamping: |v| ≤ v\_max = 0.2 × (ub − lb) per dimension.

### FIS Rules (18 clinically motivated)

The Sugeno rule base (see `fuzzy/rules.py`) covers:

| Rule Family | Condition | Consequent |
|-------------|-----------|------------|
| R1, R10, R16, R18 | Normal sinus patterns | lethal=0, class=Normal |
| R2, R3, R12, R17 | High HR + wide QRS (VT) | lethal≥0.80, class=VT |
| R4, R5, R6, R14 | VF-band spectral concentration + irregular RR | lethal≥0.70, class=VF |
| R7, R8, R13 | Poor SQI + spectral signature of noise | artifact≥0.80, class=Artifact |
| R9, R11, R15 | ST elevation, PVCs, uncertain | lethal<0.4, class=Other |

**Firing strength**: product T-norm (AND aggregation).  
**Defuzzification**: weighted-average Sugeno (crisp output per rule, no shape integration needed).

### Safety Gate Logic (C2 guarantee)

```python
if lethal_score >= tau_lethal:      # C2: ALWAYS alert — hard-wired first
    return alert=True
elif artifact_score >= tau_artifact: # suppress only when NON-lethal
    return alert=False, suppressed=True
else:
    return alert = (lethal_score >= 0.5)
```

The `if lethal_score >= tau_lethal` branch is **structurally unconditional** — it cannot
be disabled by any parameter setting, including the learned τ\_lethal. C2 is guaranteed
by *code structure*, not by training.

---

## (c) CI-Technique Rationale

### Why Fuzzy Logic for Uncertain Clinical Boundaries?

Cardiac rhythms exist on a continuum. The boundary between *sinus tachycardia* (HR=130 bpm,
non-lethal) and *VT* (HR=160 bpm, lethal) is not a crisp threshold — it depends on QRS width,
regularity, haemodynamic context, and patient history. Fuzzy logic:

- Models **graded membership** (e.g., HR=150 bpm is 60% "high" rather than binary high/low)
- Encodes **clinical expert knowledge** as transparent IF-THEN rules that clinicians can
  audit and extend
- Handles **sensor uncertainty** gracefully via the SQI input hedging rule confidence
- Provides **interpretable outputs**: alarm decisions can be traced back to specific rule firings

### Why PSO for a Non-Differentiable Multi-Objective Problem?

The fitness landscape is **non-differentiable** because:
1. The pipeline includes discrete operations (R-peak detection, argmax for class)
2. Constraint penalties introduce discontinuities (e.g., lethal\_Se suddenly jumps)
3. The fitness aggregates multiple competing objectives (Se vs. FAFI vs. compute time)

PSO advantages over gradient-based methods:
- **Derivative-free**: handles non-smooth, non-convex landscapes
- **Population-based**: explores multiple local optima simultaneously (diversity)
- **Warm-start capable**: Round 2 re-uses Round 1 gbest, accelerating adaptation
- **Configurable trade-off**: swarm diversity (reinit\_fraction) vs. exploitation (inertia decay)
- **Simple constraint handling**: hard penalties fold C1–C4 into the scalar fitness

---

## (d) Jury Defence — Limitations and Failure Modes

### Limitations

1. **Synthetic data fallback**: If PhysioNet is unavailable, all metrics are computed on
   synthetic signals that may not capture real arrhythmia morphology diversity (inter-patient
   variability, myocardial infarction morphology, etc.).

2. **Single-lead ECG**: The pipeline uses one ECG channel. Multi-lead discrimination
   (e.g., distinguishing LBBB from VT) requires orthogonal leads.

3. **PSO scalability**: With D=74, the search space is large. Swarm size 30 with 80 iterations
   may not converge to the global optimum; the result is a **good local optimum** within the
   time budget.

4. **NLMS reference channel**: In a wearable single-lead system, the reference artifact channel
   is estimated as the high-pass residual rather than a true orthogonal sensor, limiting
   adaptive cancellation effectiveness.

### Failure Modes

| Scenario | Risk | Mitigation |
|----------|------|------------|
| Severe EMG mimicking VF (high spectral energy in 3–7 Hz) | VF false positive | SQI hedge rule (R6) lowers confidence; C3 distortion penalty trains the filter to separate bands |
| Long-duration VF with amplitude decay | Reduced lethal\_score near end | Multi-rule coverage (R4, R5, R14) + τ\_lethal at 0.5 |
| Sinus tachycardia with RBBB (wide QRS) | VT false positive | RR-CV = Low (regular) combined with narrow P-wave expected; rules distinguish via rr\_cv |
| Electrode disconnection (SQI→0, random noise) | May trigger lethal | Artifact rules (R7, R8, R13) produce high artifact\_score; gate suppresses **only** if lethal\_score < τ |
| Paediatric ECG (different HR/QRS norms) | MF centers mis-calibrated | Normalisation bounds in config.py can be adjusted per population |

### How C1–C4 are Guaranteed

| Constraint | Guarantee Mechanism |
|------------|---------------------|
| **C1** (latency ≤ 3 s) | Window=2 s by design; compute budget=1 s; PSO penalises (×1e3) any violation; measured per-window in benchmark |
| **C2** (lethal Se=1.0) | `if lethal_score >= tau_lethal: alert=True` is the *first* branch in SafetyGate, structurally unconditional; PSO penalises (×1e3) any lethal miss |
| **C3** (QRS ≤5%, ST ≤0.02 mV) | Zero-phase median filters (no group delay) with subtraction-based approach; measured and penalised in fitness |
| **C4** (memory ≤2 MB streaming) | Pre-allocated ring buffers; `tracemalloc` measured per particle; streaming path uses no growing lists |
