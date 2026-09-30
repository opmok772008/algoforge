"""
api/sim_engine.py — Simulation, signal analysis, fuzzy diagnostics, and verification test engine.

Provides 100% interoperability with the frontend in FRONTEND.html:
- Sim: Deterministic physiological ECG synthesizer (Normal, VT, VF + artifacts)
- clean(), detect(), quality(), fuzzy(): Real-time DSP & Sugeno zero-order FIS
- Monitor: Fixed 6 KB preallocated ring-buffer streaming monitor
- analyze_ecg_record(): RR-feature analyzer for uploaded ECG / repo pipeline
- run_automated_tests(): Complete 9-check verification suite
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

FS = 250
N = 1500
WIN = 1000
DEFAULT_PARAMS = {"bw": 200, "sm": 5, "th": 0.3, "mw": 25}
BOUNDS = {
    "bw": (50, 500),
    "sm": (1, 15),
    "th": (0.05, 0.9),
    "mw": (8, 50),
}


# ---------------------------------------------------------------------------
# Deterministic PRNG matching front-end Mulberry32 / SplitMix
# ---------------------------------------------------------------------------

def create_rng(seed: int):
    s = int(seed) & 0xFFFFFFFF

    def next_val() -> float:
        nonlocal s
        s = (s + 0x6D2B79F5) & 0xFFFFFFFF
        t = (s ^ (s >> 15)) * (1 | s) & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * 61)) & 0xFFFFFFFF
        return float(((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0)

    return next_val


def _gauss(x: float, m: float, s: float, a: float) -> float:
    return float(a * math.exp(-((x - m) ** 2) / (2.0 * s * s)))


# ---------------------------------------------------------------------------
# Sim — Deterministic synthetic signal generator
# ---------------------------------------------------------------------------

class Sim:
    def __init__(self, seed: int = 7) -> None:
        self.R = create_rng(seed)
        self.ph: float = 0.0
        self.rr: float = 0.85
        self.t: float = 0.0
        self.vp: float = 0.0
        self.i: int = 0
        self.rs: List[int] = []

    def next(self, o: Dict[str, Any]) -> float:
        R = self.R
        self.t += 1.0 / FS
        t = self.t
        idx = self.i
        self.i += 1
        TWO_PI = 6.283185307179586

        scn = o.get("scn", "normal")
        if scn == "vf":
            f = 5.0 + 1.2 * math.sin(TWO_PI * 0.3 * t) + 0.6 * math.sin(TWO_PI * 0.83 * t)
            self.vp += TWO_PI * f / FS
            x = (0.7 + 0.3 * math.sin(TWO_PI * 0.9 * t)) * math.sin(self.vp) + 0.3 * math.sin(1.7 * self.vp + 1.0)
        else:
            is_vt = (scn == "vt")
            pv = self.ph
            pk = 0.35 if is_vt else 0.385
            self.ph += 1.0 / (FS * self.rr)
            if self.ph >= 1.0:
                self.ph -= 1.0
                self.rr = (0.33 + R() * 0.02) if is_vt else (0.85 + R() * 0.06)
            p = self.ph
            if pv < pk and p >= pk:
                self.rs.append(idx)
            if is_vt:
                x = _gauss(p, 0.35, 0.05, 1.0) + _gauss(p, 0.7, 0.07, -0.45)
            else:
                x = (
                    _gauss(p, 0.16, 0.025, 0.12)
                    + _gauss(p, 0.36, 0.008, -0.15)
                    + _gauss(p, 0.385, 0.011, 1.0)
                    + _gauss(p, 0.41, 0.009, -0.25)
                    + _gauss(p, 0.62, 0.045, 0.3)
                )

        if o.get("wander", False):
            x += 0.8 * math.sin(TWO_PI * 0.25 * t) + 0.4 * math.sin(TWO_PI * 0.11 * t + 1.0)
        if o.get("motion", False):
            e = math.sin(TWO_PI * 0.2 * t)
            x += 1.2 * e * e * math.sin(TWO_PI * 2.2 * t)

        noise_amp = 0.9 if o.get("emg", False) else 0.03
        return float(x + (R() - 0.5) * noise_amp)


# ---------------------------------------------------------------------------
# DSP utilities
# ---------------------------------------------------------------------------

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    cumsum = np.pad(np.cumsum(x, dtype=np.float64), (1, 0))
    h = max(1, int(w)) >> 1
    o = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = max(0, i - h)
        b = min(n, i + h + 1)
        o[i] = (cumsum[b] - cumsum[a]) / (b - a)
    return o


def clean_signal(x: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    base = moving_average(x, int(p.get("bw", 200)))
    return (x - base).astype(np.float32)


def percentile_val(a: np.ndarray, q: float) -> float:
    s = np.sort(np.asarray(a, dtype=np.float32))
    idx = min(len(s) - 1, int(len(s) * q))
    return float(s[idx])


def detect_peaks(y: np.ndarray, p: Dict[str, Any], fs: int = FS) -> Dict[str, List[int]]:
    n = len(y)
    sm = moving_average(y, int(p.get("sm", 5)))
    e = np.zeros(n, dtype=np.float32)
    e[2:] = (sm[2:] - sm[:-2]) ** 2
    e = moving_average(e, int(p.get("mw", 25)))

    thr = max(float(p.get("th", 0.3)) * percentile_val(e, 0.97), 2.5 * percentile_val(e, 0.5))
    pk: List[int] = []
    last = -99999
    min_refractory = int(0.2 * fs)

    for i in range(1, n - 1):
        if e[i] > thr and e[i] >= e[i - 1] and e[i] > e[i + 1]:
            a = max(0, i - 15)
            b = min(n - 1, i + 15)
            bi = a
            for j in range(a, b + 1):
                if sm[j] > sm[bi]:
                    bi = j
            if bi - last > min_refractory:
                pk.append(int(bi))
                last = bi

    return {"pk": pk}


def compute_signal_quality(y: np.ndarray) -> float:
    if len(y) < 5:
        return 1.0
    d = np.abs(y[2:] - 2.0 * y[1:-1] + y[:-2])
    a = np.abs(y)
    sg = percentile_val(d, 0.5) / 1.652 + 1e-6
    r = percentile_val(a, 0.98) / sg
    return float(max(0.0, min(1.0, (math.log10(max(1e-9, r)) - 0.4) / 0.9)))


def trapezoid_mf(x: float, a: float, b: float, c: float, d: float) -> float:
    term1 = 1.0 if (b <= a) else (x - a) / (b - a)
    term2 = 1.0 if (d <= c) else (d - x) / (d - c)
    return float(max(0.0, min(term1, 1.0, term2)))


def fuzzy_diagnostics(hr: Optional[float], cv: Optional[float], q: float) -> Dict[str, Any]:
    m = {
        "hrHigh": 0.0,
        "hrVery": 0.0,
        "regular": 0.0,
        "irregular": 0.0,
        "good": trapezoid_mf(q, 0.35, 0.6, 1.0, 1.01),
        "poor": trapezoid_mf(q, -1.0, -0.5, 0.25, 0.5),
    }
    if hr is not None and cv is not None:
        m["hrHigh"] = trapezoid_mf(hr, 110.0, 150.0, 400.0, 401.0)
        m["hrVery"] = trapezoid_mf(hr, 180.0, 230.0, 400.0, 401.0)
        m["regular"] = trapezoid_mf(cv, -1.0, -0.5, 0.12, 0.3)
        m["irregular"] = trapezoid_mf(cv, 0.15, 0.35, 5.0, 6.0)

    vt = min(m["hrHigh"], m["regular"])
    vf = min(m["hrVery"], m["good"])
    noise = min(m["poor"], m["irregular"])
    conf = max(vt, vf)
    alarm = (conf >= 0.6 and noise < 0.5)
    hold = (conf < 0.6 and noise >= 0.5) or (False if hr is None else (m["poor"] > 0.5 and conf < 0.6))

    return {
        "hr": hr,
        "cv": cv,
        "q": q,
        "m": m,
        "vt": vt,
        "vf": vf,
        "noise": noise,
        "conf": conf,
        "alarm": alarm,
        "hold": hold,
    }


def analyze_window(x: np.ndarray, p: Dict[str, Any], fs: int = FS) -> Dict[str, Any]:
    y = clean_signal(x, p)
    d = detect_peaks(y, p, fs=fs)
    q = compute_signal_quality(y)
    n = len(d["pk"])
    hr: Optional[float] = None
    cv: Optional[float] = None

    if n >= 6:
        rr: List[float] = []
        for i in range(n - 5, n):
            rr.append(float((d["pk"][i] - d["pk"][i - 1]) / fs))
        mu = sum(rr) / 5.0
        v = sum((r - mu) ** 2 for r in rr)
        cv = math.sqrt(v / 5.0) / (mu if mu > 0 else 1.0)
        rr_sorted = sorted(rr)
        hr = float(60.0 / rr_sorted[2]) if rr_sorted[2] > 0 else 0.0

    st = fuzzy_diagnostics(hr, cv, q)
    return {"y": y, "pk": d["pk"], "st": st, "q": q, "hr": hr, "cv": cv}


# ---------------------------------------------------------------------------
# Streaming Monitor
# ---------------------------------------------------------------------------

class StreamingMonitor:
    def __init__(self, seed: int = 7, p: Optional[Dict[str, Any]] = None) -> None:
        self.sim = Sim(seed)
        self.p = dict(p) if p else dict(DEFAULT_PARAMS)
        self.buf = np.zeros(N, dtype=np.float32)
        self.o: Dict[str, Any] = {"scn": "normal", "wander": False, "emg": False, "motion": False}
        self.acc: int = 0
        self.onset: Optional[float] = None
        self.alarm_at: Optional[float] = None
        self.y = np.zeros(WIN, dtype=np.float32)
        self.pk: List[int] = []
        self.st = fuzzy_diagnostics(None, None, 1.0)

    def set(self, k: str, v: Any) -> None:
        if k == "scn" and v != self.o.get("scn"):
            self.onset = None if v == "normal" else self.sim.t
            self.alarm_at = None
        self.o[k] = v

    def step(self, k: int) -> None:
        b = self.buf
        b[:-k] = b[k:]
        for i in range(N - k, N):
            b[i] = self.sim.next(self.o)
        self.acc += k
        if self.acc >= 25:
            self.acc = 0
            r = analyze_window(b[-WIN:], self.p)
            self.y = r["y"]
            self.pk = r["pk"]
            self.st = r["st"]
            if r["st"]["alarm"] and self.onset is not None and self.alarm_at is None:
                self.alarm_at = self.sim.t

    def run(self, sec: float) -> None:
        total_steps = int(sec * FS / 25)
        for _ in range(total_steps):
            self.step(25)

    def latency(self) -> Optional[float]:
        return None if self.alarm_at is None else float(self.alarm_at - self.onset)


# ---------------------------------------------------------------------------
# Repo Pipeline & Upload ECG Analyzer
# ---------------------------------------------------------------------------

def analyze_ecg_record(x: np.ndarray, fs: int = 300) -> Dict[str, Any]:
    fs = max(50, min(2000, int(fs)))
    n = len(x)
    base = moving_average(x, fs)
    y = x - base
    h = moving_average(y, 5)

    mx = float(np.max(h)) if len(h) > 0 else 1.0
    th = 0.45 * mx
    p: List[int] = []
    last = -1000000000
    refractory = int(0.25 * fs)

    for i in range(1, len(h) - 1):
        if h[i] > h[i - 1] and h[i] > h[i + 1] and h[i] > th and (i - last) > refractory:
            p.append(int(i))
            last = i

    rr: List[float] = []
    for i in range(1, len(p)):
        rr.append(float((p[i] - p[i - 1]) / fs))

    if len(rr) > 0:
        m = float(np.mean(rr))
        sd = float(np.std(rr))
        d = [rr[i] - rr[i - 1] for i in range(1, len(rr))]
        rms = float(np.sqrt(np.mean(np.square(d)))) if len(d) > 0 else 0.0
        nn50 = int(sum(1 for v in d if abs(v) > 0.05))
        hr = float(60.0 / m) if m > 0 else 0.0
        cv = float(sd / m) if m > 0 else 0.0
    else:
        m = sd = rms = hr = cv = 0.0
        nn50 = 0

    cond = "Bradycardia" if hr < 60 else ("Tachycardia" if hr > 100 else "Normal heart rate")

    if len(rr) < 3:
        rhy = "Insufficient beats"
        prog = "Could not detect enough beats: check the sampling rate, column and signal quality"
    elif cv > 0.22:
        rhy = "Atrial fibrillation suspected"
        prog = "Consult doctor ASAP" if hr < 60 else "Doctor review needed"
    elif cv > 0.09:
        rhy = "Other arrhythmia suspected"
        prog = "Consult doctor ASAP" if hr < 60 else "Doctor review needed"
    else:
        rhy = "Normal sinus rhythm"
        prog = "No problem detected" if 60 <= hr <= 100 else "Heart rate outside normal range: doctor review suggested"

    items = [
        ["Duration", f"{len(x) / fs:.1f} s"],
        ["Rhythm", rhy],
        ["Condition", cond],
        ["Heart rate", f"{round(hr)} bpm"],
        ["R peaks", str(len(p))],
        ["Avg RR", f"{m:.2f} s"],
        ["HRV (SDNN)", f"{round(sd * 1000)} ms"],
        ["RMSSD", f"{round(rms * 1000)} ms"],
        ["NN50", str(nn50)],
        ["RR variation", f"{round(cv * 100)}%"],
    ]

    return {
        "p": p,
        "y": h.tolist(),
        "rr": rr,
        "hr": hr,
        "m": m,
        "sd": sd,
        "rms": rms,
        "nn50": nn50,
        "cv": cv,
        "condition": cond,
        "rhythm": rhy,
        "prog": prog,
        "items": items,
        "advice": f"Advice: {prog} (research demo, not a medical device)",
    }


# ---------------------------------------------------------------------------
# Test Suite Evaluator (9 checks)
# ---------------------------------------------------------------------------

def _score_record(rec: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, float]:
    y = clean_signal(rec["x"], p)
    d = detect_peaks(y, p)
    n = len(y)
    tp = 0
    js = 0.0
    used: Dict[int, int] = {}
    pk = [q for q in d["pk"] if 40 < q < n - 40]
    tr = [q for q in rec["truth"] if 40 < q < n - 40]

    for q in pk:
        b = -1
        bd = 11.0
        for k, t in enumerate(tr):
            dd = abs(t - q)
            if dd < bd and k not in used:
                bd = dd
                b = k
        if b >= 0:
            used[b] = 1
            tp += 1
            js += bd

    fp = len(pk) - tp
    fn = len(tr) - tp
    se = tp / (tp + fn) if (tp + fn) else 1.0
    ppv = tp / (tp + fp) if (tp + fp) else 1.0
    jit = (js / tp * 1000.0 / FS) if tp else 0.0
    return {"se": se, "ppv": ppv, "jit": jit}


def _record_sim(seed: int, secs: float, o: Dict[str, Any]) -> Dict[str, Any]:
    s = Sim(seed)
    n = int(secs * FS)
    x = np.array([s.next(o) for _ in range(n)], dtype=np.float32)
    return {"x": x, "truth": list(s.rs)}


def run_automated_tests() -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    p = DEFAULT_PARAMS

    # 1. Clean rhythm Se and PPV >= 95%
    c = _score_record(_record_sim(21, 30.0, {"scn": "normal"}), p)
    c_ok = (c["se"] >= 0.95 and c["ppv"] >= 0.95)
    results.append({
        "name": "Clean rhythm: sensitivity and precision at least 95%",
        "ok": bool(c_ok),
        "val": f"Se {c['se'] * 100.0:.1f}%, PPV {c['ppv'] * 100.0:.1f}%",
    })

    # 2. Timing jitter under wander <= 20 ms
    w = _score_record(_record_sim(22, 30.0, {"scn": "normal", "wander": True}), p)
    w_ok = (w["jit"] <= 20.0)
    results.append({
        "name": "R-peak timing jitter under baseline wander is at most 20 ms",
        "ok": bool(w_ok),
        "val": f"{w['jit']:.1f} ms",
    })

    # 3. QRS amplitude preserved >= 85% of true 1.0 mV
    r = _record_sim(23, 20.0, {"scn": "normal", "wander": True})
    y = clean_signal(r["x"], p)
    a = 0.0
    k = 0
    for q in r["truth"]:
        if 60 < q < len(y) - 60:
            a += float(y[q])
            k += 1
    amp_ratio = (a / k) if k > 0 else 0.0
    amp_ok = (amp_ratio >= 0.85)
    results.append({
        "name": "QRS amplitude preserved after filtering (at least 85% of true 1.0 mV)",
        "ok": bool(amp_ok),
        "val": f"{amp_ratio:.2f} mV",
    })

    # 4 & 5. VT and VF alerts within 3 seconds
    for s_name in ["vt", "vf"]:
        m = StreamingMonitor(31, p)
        m.set("wander", True)
        m.run(10.0)
        m.set("scn", s_name)
        m.run(6.0)
        lat = m.latency()
        name_str = ("Ventricular tachycardia" if s_name == "vt" else "Ventricular fibrillation") + " alert within 3 seconds"
        lat_ok = (lat is not None and lat <= 3.0)
        results.append({
            "name": name_str,
            "ok": bool(lat_ok),
            "val": "no alert" if lat is None else f"{lat:.1f} s",
        })

    # 6. EMG burst raises no lethal alarm
    m_emg = StreamingMonitor(32, p)
    m_emg.set("emg", True)
    m_emg.run(8.0)
    bad_count = 0
    total_steps = int(20 * FS / 25)
    for _ in range(total_steps):
        m_emg.step(25)
        if m_emg.st["alarm"]:
            bad_count += 1
    results.append({
        "name": "Normal rhythm with severe EMG burst raises no lethal alarm",
        "ok": bool(bad_count == 0),
        "val": f"{bad_count} false alarm windows in 20 s",
    })

    # 7. Fuzzy alarm confidence monotonic with HR
    hr_list = [100.0, 130.0, 150.0, 180.0, 220.0]
    confs = [fuzzy_diagnostics(h, 0.05, 0.9)["conf"] for h in hr_list]
    is_monotonic = all(i == 0 or confs[i] >= confs[i - 1] for i in range(len(confs)))
    results.append({
        "name": "Fuzzy alarm confidence never decreases as heart rate rises",
        "ok": bool(is_monotonic),
        "val": " ".join(f"{v:.2f}" for v in confs),
    })

    # 8. Fixed streaming buffer (6 KB) over 60 s
    q_mon = StreamingMonitor(33, p)
    q_mon.run(60.0)
    buf_ok = (len(q_mon.buf) == N)
    results.append({
        "name": f"Streaming buffer stays fixed at {N} samples (6 KB) over 60 s",
        "ok": bool(buf_ok),
        "val": f"{q_mon.buf.nbytes} bytes",
    })

    # 9. Evolution elitism (never loses fitness) over 12 generations
    train_records = [
        _record_sim(11, 10.0, {"scn": "normal", "wander": True}),
        _record_sim(12, 10.0, {"scn": "normal", "emg": True}),
        _record_sim(13, 10.0, {"scn": "normal", "wander": True, "motion": True, "emg": True}),
    ]

    def fit_cand(cand: Dict[str, Any]) -> float:
        s_val = 0.0
        j_val = 0.0
        for rec in train_records:
            scr = _score_record(rec, cand)
            denom = scr["se"] + scr["ppv"]
            s_val += 2.0 * scr["se"] * scr["ppv"] / (denom if denom > 0 else 1.0)
            j_val += scr["jit"]
        return float(s_val / 3.0 - 0.005 * j_val / 3.0)

    class TestGA:
        def __init__(self, s_seed: int) -> None:
            self.R = create_rng(s_seed)
            self.g = 0
            self.pop = [self._rand() for _ in range(14)]
            self._eval()

        def _rand(self) -> Dict[str, Any]:
            res: Dict[str, Any] = {}
            for k, (lo, hi) in BOUNDS.items():
                res[k] = lo + self.R() * (hi - lo)
            res["bw"] = round(res["bw"])
            res["sm"] = round(res["sm"])
            res["mw"] = round(res["mw"])
            return res

        def _eval(self) -> None:
            for cand in self.pop:
                cand["f"] = fit_cand(cand)
            self.pop.sort(key=lambda x: x["f"], reverse=True)
            self.best = self.pop[0]
            self.mean = sum(x["f"] for x in self.pop) / len(self.pop)

        def step(self) -> None:
            R = self.R
            nx = [dict(self.pop[0])]

            def pick():
                a = self.pop[int(R() * 14)]
                b = self.pop[int(R() * 14)]
                return a if a["f"] > b["f"] else b

            while len(nx) < 14:
                a = pick()
                b = pick()
                c: Dict[str, Any] = {}
                for k, (lo, hi) in BOUNDS.items():
                    w = R()
                    v = w * a[k] + (1.0 - w) * b[k]
                    if R() < 0.3:
                        v += (R() - 0.5) * 0.3 * (hi - lo)
                    c[k] = min(hi, max(lo, v))
                c["bw"] = round(c["bw"])
                c["sm"] = round(c["sm"])
                c["mw"] = round(c["mw"])
                nx.append(c)
            self.pop = nx
            self.g += 1
            self._eval()

    ga_inst = TestGA(5)
    f0 = ga_inst.best["f"]
    for _ in range(12):
        ga_inst.step()
    f12 = ga_inst.best["f"]
    ga_ok = (f12 >= f0)
    results.append({
        "name": "Evolution never loses fitness (elitism) over 12 generations",
        "ok": bool(ga_ok),
        "val": f"{f0:.3f} to {f12:.3f}",
    })

    passed_count = sum(1 for r in results if r["ok"])
    return {
        "summary": f"{passed_count} / {len(results)} passed",
        "passed": passed_count,
        "total": len(results),
        "tests": results,
    }
