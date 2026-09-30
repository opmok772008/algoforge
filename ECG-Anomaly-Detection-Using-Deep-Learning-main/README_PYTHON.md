# ECG Anomaly Detection & Cardiac Monitor (Python)

This is a complete, self-contained Python implementation of the **ECG Anomaly Detection Using Deep Learning** project. It reproduces and enhances all features from the original MATLAB project without requiring MATLAB licenses or toolboxes.

---

## Quick Start (How to Run)

### Method 1: Double-Click
Double-click `run.bat` in this folder.

### Method 2: Terminal / PowerShell
Open PowerShell or Command Prompt in this folder and run:
```bash
python main_gui.py
```

---

## Features

1. **Cardiac Waveform Display**:
   - Filtered ECG Lead II waveform with marked **R-Peaks** (Pan-Tompkins QRS algorithm).
   - Moving-window integrated energy display with refractory period markers.
2. **Clinical Metrics Dashboard**:
   - **Heart Rate**: Instantaneous and average Beats Per Minute (BPM).
   - **Rhythm Classification**: Normal Heart Rate, Bradycardia (< 60 BPM), or Tachycardia (> 100 BPM).
   - **Heart Rate Variability (HRV)**: SDNN and RMSSD in milliseconds.
   - **NN50 & pNN50**: Successive RR intervals differing by > 50 ms.
   - **Average RR Interval**: In milliseconds.
3. **Anomaly & Arrhythmia Classification**:
   - **Normal Sinus Rhythm (N)**
   - **Atrial Fibrillation (A)**
   - **Other Arrhythmia (O)**
   - **Noisy / Degraded Signal (~)**
4. **Clinical Suggestion / Prognosis**:
   - Matching the original clinical decision rules (e.g., *"No problem detected"*, *"Doctor review needed"*, *"Consult doctor ASAP"*).
5. **Real-Time Simulation Monitor**:
   - Click **▶ Real-Time Monitor** to stream and scroll the ECG dynamically like an ICU cardiac monitor.
6. **File Support**:
   - Load custom `.mat` files (PhysioNet format containing `val`), `.csv`, `.npy`, or `.txt` records via the **📂 Load ECG File** button.
   - Built-in test samples accessible via the **Quick Test Sample** dropdown.

---

## File Structure

- `main_gui.py`: The interactive graphical monitor application.
- `ecg_processor.py`: Core signal processing, 50Hz notch filter, baseline wander removal, Pan-Tompkins QRS detector, and wavelet/spectral feature extraction.
- `model.py`: Arrhythmia classifier and clinical prognosis engine.
- `generate_samples.py`: Creates realistic PhysioNet-style test signals in `.mat` and `.csv` formats.
- `sample_data/`: Contains ready-to-test `.mat` and `.csv` records.
- `run.bat`: One-click runner script for Windows.
