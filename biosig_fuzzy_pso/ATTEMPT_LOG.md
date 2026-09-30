# ATTEMPT LOG

| Attempt | Round | Fitness | Se   | Sp   | FAFI | Jitter(ms) | dSNR(dB) | ms/win | Note |
|---------|-------|---------|------|------|------|------------|----------|--------|------|
| 2026-09-30T10:47 | R1 | (see results/convergence_round1.csv) | 0.329 | 0.662 | 0.000 | 13.3 | 6.0 | 61.5 | Initial attempt — baseline defaults |
| 2026-09-30T10:52 | R2 (unadapted) | (see results/convergence_round2.csv) | 0.135 | 0.541 | 0.000 | 13.1 | 5.0 | 72.2 | What changed and why: EMG perturbation applied; Round 1 params used without adaptation |
| 2026-09-30T10:52 | R2 (adapted) | (see results/convergence_round2.csv) | 0.246 | 0.630 | 0.000 | 14.5 | 7.0 | 63.0 | What changed and why: warm-start PSO with 30% reinit for robustness to EMG shift |
