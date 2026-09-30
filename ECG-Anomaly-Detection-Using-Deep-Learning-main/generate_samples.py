"""
Sample ECG Generator & File Loader
Generates realistic PhysioNet-style test signals (.mat and .csv)
and provides file loading routines.
"""

import os
import numpy as np
import scipy.io as sio


def generate_ecg_beat(t, p_amp=0.15, q_amp=-0.2, r_amp=1.2, s_amp=-0.35, t_amp=0.25):
    """Generates a single synthetic P-QRS-T cycle over normalized beat phase (-0.5 to 0.5)"""
    # Gaussian waveforms for P, Q, R, S, T waves
    p_wave = p_amp * np.exp(-((t + 0.20) ** 2) / (2 * (0.025 ** 2)))
    q_wave = q_amp * np.exp(-((t + 0.05) ** 2) / (2 * (0.010 ** 2)))
    r_wave = r_amp * np.exp(-(t ** 2) / (2 * (0.015 ** 2)))
    s_wave = s_amp * np.exp(-((t - 0.05) ** 2) / (2 * (0.012 ** 2)))
    t_wave = t_amp * np.exp(-((t - 0.25) ** 2) / (2 * (0.045 ** 2)))
    return p_wave + q_wave + r_wave + s_wave + t_wave


def generate_ecg_signal(condition="normal", duration_sec=30.0, fs=300.0, noise_level=0.03):
    """
    Generates realistic 30-second single-lead ECG signal sampled at 300 Hz.
    Conditions:
    - 'normal': Regular ~72 bpm, normal HRV
    - 'afib': Atrial Fibrillation (irregular RR, no P wave, fibrillatory waves)
    - 'bradycardia': Slow heart rate ~46 bpm
    - 'tachycardia': Rapid heart rate ~128 bpm
    - 'noisy': High muscle / 50Hz / baseline drift noise
    """
    total_samples = int(duration_sec * fs)
    t_total = np.linspace(0, duration_sec, total_samples, endpoint=False)
    ecg = np.zeros(total_samples)

    np.random.seed(42)

    if condition == "normal":
        base_bpm = 72.0
        rr_mean = 60.0 / base_bpm
        current_t = 0.4
        while current_t < duration_sec - 0.4:
            # Normal respiratory sinus arrhythmia variation (~35ms)
            rr_interval = rr_mean + np.random.normal(0, 0.035)
            # Add beat
            beat_samples = int(0.6 * fs)
            t_beat = np.linspace(-0.3, 0.3, beat_samples)
            beat = generate_ecg_beat(t_beat)
            start_idx = int((current_t - 0.3) * fs)
            end_idx = start_idx + beat_samples
            if 0 <= start_idx and end_idx <= total_samples:
                ecg[start_idx:end_idx] += beat
            current_t += rr_interval

        # Gentle baseline wander and small white noise
        baseline = 0.05 * np.sin(2 * np.pi * 0.15 * t_total)
        noise = np.random.normal(0, noise_level, total_samples)
        ecg = ecg + baseline + noise

    elif condition == "afib":
        # Atrial Fibrillation: irregular RR, no distinct P wave, fibrillatory f-waves
        base_bpm = 95.0
        current_t = 0.3
        while current_t < duration_sec - 0.4:
            # Highly variable RR intervals (gamma / exponential distribution)
            rr_interval = np.random.uniform(0.35, 1.05)
            beat_samples = int(0.6 * fs)
            t_beat = np.linspace(-0.3, 0.3, beat_samples)
            # No P wave in AFib
            beat = generate_ecg_beat(t_beat, p_amp=0.0)
            start_idx = int((current_t - 0.3) * fs)
            end_idx = start_idx + beat_samples
            if 0 <= start_idx and end_idx <= total_samples:
                ecg[start_idx:end_idx] += beat
            current_t += rr_interval

        # Fibrillatory waves (4-8 Hz irregular oscillations)
        f_waves = 0.08 * np.sin(2 * np.pi * 5.8 * t_total + np.random.uniform(0, 2*np.pi, total_samples))
        baseline = 0.08 * np.sin(2 * np.pi * 0.2 * t_total)
        noise = np.random.normal(0, noise_level * 1.5, total_samples)
        ecg = ecg + f_waves + baseline + noise

    elif condition == "bradycardia":
        # Slow sinus rhythm (~46 bpm)
        base_bpm = 46.0
        rr_mean = 60.0 / base_bpm
        current_t = 0.5
        while current_t < duration_sec - 0.5:
            rr_interval = rr_mean + np.random.normal(0, 0.04)
            beat_samples = int(0.6 * fs)
            t_beat = np.linspace(-0.3, 0.3, beat_samples)
            beat = generate_ecg_beat(t_beat)
            start_idx = int((current_t - 0.3) * fs)
            end_idx = start_idx + beat_samples
            if 0 <= start_idx and end_idx <= total_samples:
                ecg[start_idx:end_idx] += beat
            current_t += rr_interval

        baseline = 0.04 * np.sin(2 * np.pi * 0.1 * t_total)
        noise = np.random.normal(0, noise_level, total_samples)
        ecg = ecg + baseline + noise

    elif condition == "tachycardia":
        # Rapid sinus rhythm (~128 bpm)
        base_bpm = 128.0
        rr_mean = 60.0 / base_bpm
        current_t = 0.2
        while current_t < duration_sec - 0.3:
            rr_interval = rr_mean + np.random.normal(0, 0.02)
            beat_samples = int(0.5 * fs)
            t_beat = np.linspace(-0.25, 0.25, beat_samples)
            beat = generate_ecg_beat(t_beat, r_amp=1.1, t_amp=0.2)
            start_idx = int((current_t - 0.25) * fs)
            end_idx = start_idx + beat_samples
            if 0 <= start_idx and end_idx <= total_samples:
                ecg[start_idx:end_idx] += beat
            current_t += rr_interval

        baseline = 0.05 * np.sin(2 * np.pi * 0.2 * t_total)
        noise = np.random.normal(0, noise_level, total_samples)
        ecg = ecg + baseline + noise

    elif condition == "noisy":
        # Heavily contaminated with 50 Hz powerline hum, muscle EMG, motion artifact
        base_bpm = 75.0
        rr_mean = 60.0 / base_bpm
        current_t = 0.4
        while current_t < duration_sec - 0.4:
            rr_interval = rr_mean + np.random.normal(0, 0.05)
            beat_samples = int(0.6 * fs)
            t_beat = np.linspace(-0.3, 0.3, beat_samples)
            beat = generate_ecg_beat(t_beat)
            start_idx = int((current_t - 0.3) * fs)
            end_idx = start_idx + beat_samples
            if 0 <= start_idx and end_idx <= total_samples:
                ecg[start_idx:end_idx] += beat
            current_t += rr_interval

        powerline = 0.35 * np.sin(2 * np.pi * 50.0 * t_total)
        drift = 0.7 * np.sin(2 * np.pi * 0.4 * t_total) + 0.4 * np.cos(2 * np.pi * 0.08 * t_total)
        emg_noise = np.random.normal(0, 0.25, total_samples)
        ecg = ecg + powerline + drift + emg_noise

    # Scale to millivolts
    return ecg.astype(np.float64)


def create_sample_files(output_dir="sample_data"):
    """Creates sample test signals in .mat and .csv formats"""
    os.makedirs(output_dir, exist_ok=True)
    samples = [
        ("sample_normal_sinus", "normal"),
        ("sample_atrial_fibrillation", "afib"),
        ("sample_bradycardia", "bradycardia"),
        ("sample_tachycardia", "tachycardia"),
        ("sample_noisy_signal", "noisy")
    ]

    saved_files = []
    for name, cond in samples:
        sig = generate_ecg_signal(condition=cond, duration_sec=30.0, fs=300.0)
        # 1. Save as MATLAB .mat format (val vector matching PhysioNet challenge)
        mat_path = os.path.join(output_dir, f"{name}.mat")
        sio.savemat(mat_path, {"val": sig.reshape(1, -1)})

        # 2. Save as CSV format
        csv_path = os.path.join(output_dir, f"{name}.csv")
        np.savetxt(csv_path, sig, delimiter=",", header="ecg_mv", comments="")

        saved_files.append((name, mat_path, csv_path))

    return saved_files


def load_ecg_file(filepath: str):
    """
    Loads ECG data from .mat, .csv, .npy, or .txt file.
    Returns 1D numpy array of signal values.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".mat":
        data = sio.loadmat(filepath)
        # Look for standard keys ('val', 'signal', 'ECG', etc.)
        for key in ['val', 'signal', 'ecg', 'data', 'Signal']:
            if key in data:
                arr = np.asarray(data[key], dtype=np.float64)
                return arr.flatten()
        # Fallback to the first non-metadata key
        for k, v in data.items():
            if not k.startswith("__") and isinstance(v, np.ndarray):
                return v.astype(np.float64).flatten()
        raise ValueError(f"Could not find ECG signal variable in {filepath}")

    elif ext == ".csv":
        try:
            arr = np.loadtxt(filepath, delimiter=",", skiprows=1)
        except Exception:
            arr = np.loadtxt(filepath, delimiter=",")
        if arr.ndim > 1:
            arr = arr[:, 0]
        return arr.astype(np.float64).flatten()

    elif ext == ".npy":
        arr = np.load(filepath)
        return arr.astype(np.float64).flatten()

    elif ext in [".txt", ".dat"]:
        arr = np.loadtxt(filepath)
        if arr.ndim > 1:
            arr = arr[:, 0]
        return arr.astype(np.float64).flatten()

    else:
        raise ValueError(f"Unsupported file format: {ext}")


if __name__ == "__main__":
    files = create_sample_files()
    print(f"Generated {len(files)} sample signals in sample_data directory.")
