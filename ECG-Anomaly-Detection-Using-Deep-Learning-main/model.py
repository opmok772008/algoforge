"""
Arrhythmia & ECG Anomaly Detection Model (Pure NumPy Implementation)
Runs on pure NumPy/SciPy without external C-extension/DLL dependencies.
Implements:
1. Stage 1: Noise Detector (Naive Bayes classifier matching nbclassifier.m)
2. Stage 2: Arrhythmia Ensemble Classifier (Normal Sinus Rhythm, Atrial Fibrillation, Other Arrhythmia)
3. Stage 3: Clinical Recommendation & Prognosis Engine (matching RealTimeGUI.m)
"""

import numpy as np


class PureNumPyNaiveBayes:
    """Gaussian Naive Bayes classifier implemented in pure NumPy"""
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = np.mean(X_c, axis=0)
            self.var[idx, :] = np.var(X_c, axis=0) + 1e-4  # smoothing variance
            self.priors[idx] = X_c.shape[0] / float(X.shape[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_probs = np.zeros((n_samples, n_classes), dtype=np.float64)

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx] + 1e-12)
            # Gaussian log-likelihood: -0.5 * log(2*pi*var) - ((x - mean)^2 / (2*var))
            var = self.var[idx, :]
            mean = self.mean[idx, :]
            log_lik = -0.5 * np.sum(np.log(2.0 * np.pi * var)) - 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            log_probs[:, idx] = prior + log_lik

        # Numerical stability via log-sum-exp trick
        max_log = np.max(log_probs, axis=1, keepdims=True)
        exp_probs = np.exp(log_probs - max_log)
        probs = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
        return probs


class ECGAnomalyClassifier:
    """
    Ensemble ECG Anomaly Detection and Arrhythmia Classifier:
    - Normal Sinus Rhythm (N / 0)
    - Atrial Fibrillation (A / 1)
    - Other Arrhythmia (O / 2)
    - Noisy / Artifact (~ / 3)
    """
    def __init__(self):
        self.classes = [
            "Normal Sinus Rhythm",
            "Atrial Fibrillation",
            "Other Arrhythmia",
            "Noisy Signal"
        ]
        self.class_codes = ["N", "A", "O", "~"]
        self.nb = PureNumPyNaiveBayes()
        self._train_base_model()

    def _train_base_model(self):
        np.random.seed(42)
        # Feature vector dimensions (17 features):
        # [std(D1..D5, A5), mean_abs(D1..D3), HR, SDNN, RMSSD, AVRR, NN50, pNN50, spec_entropy, dom_freq]
        X = []
        y = []
        n_per_class = 200

        for _ in range(n_per_class):
            # 0: Normal Sinus Rhythm
            hr = np.random.normal(72, 7)
            sdnn = np.random.normal(45, 8)
            rmssd = np.random.normal(32, 6)
            pnn50 = np.random.normal(12, 4)
            spec_ent = np.random.normal(3.8, 0.25)
            dom_f = np.random.normal(1.2, 0.15)
            snr = np.random.normal(18, 3)
            f_norm = [
                np.random.normal(0.08, 0.02), np.random.normal(0.12, 0.02), np.random.normal(0.20, 0.03),
                np.random.normal(0.28, 0.04), np.random.normal(0.35, 0.04), np.random.normal(0.85, 0.08),
                0.05, 0.08, 0.14,
                hr, sdnn, rmssd, 60.0/hr, int(pnn50*0.3), pnn50, spec_ent, dom_f, snr
            ]
            X.append(f_norm)
            y.append(0)

            # 1: Atrial Fibrillation (Marked irregular RR, elevated RMSSD/SDNN, high pNN50, high entropy)
            hr = np.random.normal(96, 18)
            sdnn = np.random.normal(110, 20)
            rmssd = np.random.normal(95, 18)
            pnn50 = np.random.normal(38, 8)
            spec_ent = np.random.normal(5.2, 0.35)
            dom_f = np.random.normal(5.8, 0.8)
            snr = np.random.normal(16, 3)
            f_af = [
                np.random.normal(0.15, 0.03), np.random.normal(0.18, 0.03), np.random.normal(0.25, 0.04),
                np.random.normal(0.32, 0.04), np.random.normal(0.40, 0.05), np.random.normal(0.70, 0.08),
                0.10, 0.13, 0.18,
                hr, sdnn, rmssd, 60.0/hr, int(pnn50*0.4), pnn50, spec_ent, dom_f, snr
            ]
            X.append(f_af)
            y.append(1)

            # 2: Other Arrhythmia (Bradycardia, Tachycardia, premature contractions)
            is_brady = np.random.rand() > 0.5
            hr = np.random.normal(48, 5) if is_brady else np.random.normal(126, 10)
            sdnn = np.random.normal(65, 12)
            rmssd = np.random.normal(42, 10)
            pnn50 = np.random.normal(18, 5)
            spec_ent = np.random.normal(4.2, 0.3)
            dom_f = np.random.normal(2.1, 0.4)
            snr = np.random.normal(17, 3)
            f_other = [
                np.random.normal(0.10, 0.02), np.random.normal(0.14, 0.02), np.random.normal(0.22, 0.03),
                np.random.normal(0.30, 0.04), np.random.normal(0.38, 0.04), np.random.normal(0.80, 0.08),
                0.07, 0.10, 0.15,
                hr, sdnn, rmssd, 60.0/hr, int(pnn50*0.3), pnn50, spec_ent, dom_f, snr
            ]
            X.append(f_other)
            y.append(2)

            # 3: Noisy Signal (Extreme detail wavelet energy, flat/erratic spectrum, low SNR)
            hr = np.random.normal(75, 12)
            snr = np.random.normal(-1.0, 1.5)
            f_noise = [
                np.random.normal(0.65, 0.12), np.random.normal(0.55, 0.10), np.random.normal(0.45, 0.08),
                np.random.normal(0.50, 0.08), np.random.normal(0.60, 0.08), np.random.normal(0.95, 0.10),
                0.40, 0.35, 0.30,
                hr, 120, 110, 60.0/hr, 25, 45, 6.5, 50.0, snr
            ]
            X.append(f_noise)
            y.append(3)

        self.nb.fit(np.array(X), np.array(y))

    def predict(self, feature_vector: np.ndarray, metrics: dict):
        """
        Predicts ECG class, calculates confidence, and determines clinical prognosis.
        """
        feat = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        hr = float(metrics.get("heart_rate", 70.0))
        sdnn = float(metrics.get("hrv_sdnn", 40.0))
        rmssd = float(metrics.get("rmssd", 30.0))
        pnn50 = float(metrics.get("pnn50", 10.0))
        rr_sec = metrics.get("rr_intervals_sec", np.array([]))

        # Base Naive Bayes probabilities
        probs = self.nb.predict_proba(feat)[0]

        # Clinical rule-based refinement (PhysioNet standard criteria)
        # Stage 1: Noise Screening (SNR < 3.5 dB indicates heavily degraded / noisy signal)
        snr_db = float(feature_vector[17]) if len(feature_vector) > 17 else 15.0
        if snr_db < 3.5:
            probs[3] += 3.0
            probs[0] = max(0.01, probs[0] - 0.8)
            probs[1] = max(0.01, probs[1] - 0.8)
            probs[2] = max(0.01, probs[2] - 0.8)
        else:
            probs[3] = max(0.01, probs[3] - 0.5)

        # Atrial Fibrillation detection: Highly irregular RR intervals
        if len(rr_sec) >= 4 and snr_db >= 3.5:
            cv_rr = np.std(rr_sec) / (np.mean(rr_sec) + 1e-6)
            if cv_rr > 0.20 and rmssd > 50.0:
                probs[1] += 0.8
                probs[0] = max(0.01, probs[0] - 0.5)

        # Bradycardia / Tachycardia (Other Arrhythmia)
        if (hr < 55.0 or hr > 110.0) and hr > 0:
            if probs[1] < probs[2]:  # If not dominant AFib
                probs[2] += 0.65
                probs[0] = max(0.01, probs[0] - 0.4)

        # Normal Sinus Rhythm criteria
        high_freq_noise = float(feature_vector[0] + feature_vector[1])
        if 60.0 <= hr <= 100.0 and len(rr_sec) >= 4:
            cv_rr = np.std(rr_sec) / (np.mean(rr_sec) + 1e-6)
            if cv_rr < 0.12 and rmssd < 45.0 and high_freq_noise < 0.4:
                probs[0] += 0.75
                probs[1] = max(0.01, probs[1] - 0.4)
                probs[2] = max(0.01, probs[2] - 0.4)

        # Normalize probabilities
        probs = np.maximum(probs, 1e-4)
        probs = probs / np.sum(probs)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx] * 100.0)

        label = self.classes[pred_idx]
        code = self.class_codes[pred_idx]

        # Prognosis matching original MATLAB GUI logic:
        # lines 288-301 of RealTimeGUI.m
        if pred_idx == 0 and 60.0 <= hr <= 100.0:
            prognosis = "No problem detected - Normal Sinus Rhythm"
            urgency = "Normal"
        elif pred_idx == 1 and hr < 60.0:
            prognosis = "Consult doctor ASAP (Atrial Fibrillation with Bradycardia)"
            urgency = "Urgent"
        elif pred_idx == 2 and hr < 60.0:
            prognosis = "Consult doctor ASAP (Arrhythmia with Bradycardia)"
            urgency = "Urgent"
        elif pred_idx == 1 and hr >= 60.0:
            prognosis = "Doctor review needed (Atrial Fibrillation Detected)"
            urgency = "Attention"
        elif pred_idx == 2 and hr >= 60.0:
            prognosis = "Doctor review needed (Arrhythmia Detected)"
            urgency = "Attention"
        elif pred_idx == 3:
            prognosis = "Excessive noise in signal. Clean electrode contacts and repeat ECG."
            urgency = "Warning"
        else:
            prognosis = "Routine monitoring recommended"
            urgency = "Normal"

        return {
            "prediction": label,
            "code": code,
            "pred_idx": pred_idx,
            "confidence": confidence,
            "probabilities": {self.classes[i]: float(probs[i] * 100.0) for i in range(4)},
            "prognosis": prognosis,
            "urgency": urgency
        }
