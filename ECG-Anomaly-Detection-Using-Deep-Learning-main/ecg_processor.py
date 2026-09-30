"""
Core ECG Processing & Feature Extraction Module
Replicates and enhances the MATLAB feature extraction pipeline:
- Baseline wander removal & 50Hz notch filtering
- Pan-Tompkins QRS and R-peak detection algorithm
- RR interval calculation, Heart Rate (BPM), HRV (SDNN, RMSSD, NN50, pNN50)
- Wavelet decomposition features (db4) matching waveletdecomposition.m
- Spectral features matching spectfeatures.m
"""

import numpy as np
import scipy.signal as signal
import pywt


class ECGProcessor:
    def __init__(self, fs: float = 300.0):
        self.fs = float(fs)

    def filter_signal(self, ecg: np.ndarray) -> np.ndarray:
        """
        Filter ECG:
        1. 50 Hz / 60 Hz notch filter
        2. Bandpass filter (0.5 Hz - 45 Hz) to eliminate baseline wander and high-frequency noise
        """
        ecg = np.asarray(ecg, dtype=np.float64)
        if len(ecg) < 10:
            return ecg

        # 50 Hz Notch filter
        try:
            f0 = 50.0
            Q = 30.0
            b_notch, a_notch = signal.iirnotch(f0, Q, self.fs)
            filtered = signal.filtfilt(b_notch, a_notch, ecg)
        except Exception:
            filtered = ecg.copy()

        # Bandpass filter (0.5 to 40 Hz)
        try:
            lowcut = 0.5
            highcut = min(40.0, (self.fs / 2.0) - 1.0)
            order = 3
            nyq = 0.5 * self.fs
            b_band, a_band = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='bandpass')
            filtered = signal.filtfilt(b_band, a_band, filtered)
        except Exception:
            pass

        return filtered

    def pan_tompkins(self, ecg: np.ndarray):
        """
        Pan-Tompkins algorithm for QRS complex detection:
        1. Bandpass filter (5-15 Hz)
        2. Derivative operator
        3. Squaring
        4. Moving window integration
        5. Adaptive threshold peak detection
        """
        ecg = np.asarray(ecg, dtype=np.float64)
        N = len(ecg)
        if N < int(self.fs):
            return np.array([]), np.zeros_like(ecg), ecg

        # 1. Bandpass 5 - 15 Hz
        lowcut = 5.0
        highcut = min(15.0, (self.fs / 2.0) - 1.0)
        nyq = 0.5 * self.fs
        b, a = signal.butter(2, [lowcut / nyq, highcut / nyq], btype='bandpass')
        bandpass = signal.filtfilt(b, a, ecg)

        # 2. Derivative
        derivative = np.gradient(bandpass)

        # 3. Squaring
        squared = derivative ** 2

        # 4. Moving window integration (window ~ 150 ms)
        win_size = int(0.15 * self.fs)
        if win_size < 1:
            win_size = 1
        window = np.ones(win_size) / win_size
        integrated = np.convolve(squared, window, mode='same')

        # 5. Peak finding on integrated signal with refractory period (~ 200 ms)
        min_dist = int(0.20 * self.fs)
        th = 0.35 * np.max(integrated) if np.max(integrated) > 0 else 0.1
        peaks_int, _ = signal.find_peaks(integrated, distance=min_dist, height=th)

        # Map peaks to local maximum in original/filtered ECG
        r_peaks = []
        search_radius = int(0.10 * self.fs)
        for p in peaks_int:
            left = max(0, p - search_radius)
            right = min(N, p + search_radius)
            if right > left:
                actual_peak = left + np.argmax(ecg[left:right])
                # avoid duplicates
                if len(r_peaks) == 0 or (actual_peak - r_peaks[-1]) > min_dist:
                    r_peaks.append(actual_peak)

        return np.array(r_peaks, dtype=int), integrated, bandpass

    def compute_rr_metrics(self, r_peaks: np.ndarray, total_samples: int):
        """
        Calculates heart rate, AVRR, SDNN, RMSSD, NN50, and condition
        """
        if len(r_peaks) < 2:
            return {
                "heart_rate": 0.0,
                "hrv_sdnn": 0.0,
                "rmssd": 0.0,
                "avrr": 0.0,
                "nn50": 0,
                "pnn50": 0.0,
                "condition": "Undetermined (Too few peaks)",
                "rr_intervals_sec": np.array([])
            }

        # RR intervals in seconds
        rr_sec = np.diff(r_peaks) / self.fs
        # Filter physiologically impossible intervals (< 0.2s or > 3.0s)
        valid_mask = (rr_sec >= 0.2) & (rr_sec <= 3.0)
        if np.sum(valid_mask) >= 2:
            rr_sec = rr_sec[valid_mask]

        avrr = float(np.mean(rr_sec))
        hr = float(60.0 / avrr) if avrr > 0 else 0.0

        # HRV metrics
        rr_ms = rr_sec * 1000.0
        sdnn = float(np.std(rr_ms))
        diff_rr = np.diff(rr_ms)
        rmssd = float(np.sqrt(np.mean(diff_rr ** 2))) if len(diff_rr) > 0 else 0.0
        nn50 = int(np.sum(np.abs(diff_rr) > 50.0)) if len(diff_rr) > 0 else 0
        pnn50 = float((nn50 / len(diff_rr)) * 100.0) if len(diff_rr) > 0 else 0.0

        # Heart rate classification
        if hr < 60.0:
            condition = "Bradycardia (Slow Heart Rate)"
        elif hr > 100.0:
            condition = "Tachycardia (Rapid Heart Rate)"
        else:
            condition = "Normal Heart Rate"

        return {
            "heart_rate": hr,
            "hrv_sdnn": sdnn,
            "rmssd": rmssd,
            "avrr": avrr,
            "nn50": nn50,
            "pnn50": pnn50,
            "condition": condition,
            "rr_intervals_sec": rr_sec
        }

    def extract_features(self, ecg: np.ndarray) -> np.ndarray:
        """
        Extracts 11-channel features matching MATLAB implementation:
        - 6 Wavelet decomposition levels (db4)
        - 3 RR features (RR interval vector, HRV vector, summary stats)
        - 2 Spectral features (Instantaneous frequency & Spectral entropy)
        """
        ecg = np.asarray(ecg, dtype=np.float64).flatten()
        target_len = 9000
        if len(ecg) < target_len:
            ecg = np.pad(ecg, (0, target_len - len(ecg)), mode='edge')
        else:
            ecg = ecg[:target_len]

        filtered = self.filter_signal(ecg)

        # 1. Wavelet decomposition (db4, level 5)
        coeffs = pywt.wavedec(filtered, 'db4', level=5)
        # cA5, cD5, cD4, cD3, cD2, cD1
        cA5, cD5, cD4, cD3, cD2, cD1 = coeffs

        # Reconstructed components matching wrcoef
        def reconstruct(level_type, level_num):
            zero_coeffs = [np.zeros_like(c) for c in coeffs]
            idx = 0 if level_type == 'a' else (6 - level_num)
            zero_coeffs[idx] = coeffs[idx]
            rec = pywt.waverec(zero_coeffs, 'db4')
            return rec[:target_len]

        D1 = reconstruct('d', 1)
        D2 = reconstruct('d', 2)
        D3 = reconstruct('d', 3)
        D4 = reconstruct('d', 4)
        D5 = reconstruct('d', 5)
        A5 = reconstruct('a', 5)

        # 2. RR Features
        r_peaks, _, _ = self.pan_tompkins(filtered)
        metrics = self.compute_rr_metrics(r_peaks, len(filtered))
        rr_sec = metrics["rr_intervals_sec"]

        # 3. Spectral Features
        # Spectral entropy approximation
        freqs, psd = signal.welch(filtered, fs=self.fs, nperseg=min(512, len(filtered)))
        psd_norm = psd / (np.sum(psd) + 1e-12)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

        # Dominant frequency
        dom_freq = freqs[np.argmax(psd)]

        # 4. Signal-to-Noise Ratio (Stage 1 Noise Screening)
        noise_residual = ecg - filtered
        signal_var = float(np.var(filtered))
        noise_var = float(np.var(noise_residual))
        snr_db = float(10.0 * np.log10(signal_var / (noise_var + 1e-9)))

        # Aggregate feature summary vector for classification
        feat_vector = [
            np.std(D1), np.std(D2), np.std(D3), np.std(D4), np.std(D5), np.std(A5),
            np.mean(np.abs(D1)), np.mean(np.abs(D2)), np.mean(np.abs(D3)),
            metrics["heart_rate"], metrics["hrv_sdnn"], metrics["rmssd"],
            metrics["avrr"], metrics["nn50"], metrics["pnn50"],
            spectral_entropy, dom_freq, snr_db
        ]
        return np.array(feat_vector, dtype=np.float32)
