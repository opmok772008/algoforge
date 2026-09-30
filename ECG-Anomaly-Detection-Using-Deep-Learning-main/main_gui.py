"""
ECG Anomaly Detection & Arrhythmia Analysis GUI
Medical Cardiac Monitor Interface (Python Port of MATLAB RealTimeGUI)
Supports:
- Loading PhysioNet .mat files, CSV, NPY, TXT files
- Built-in clinical test signals (Normal, AFib, Bradycardia, Tachycardia, Noisy)
- Dynamic ECG playback / real-time monitoring simulation
- Pan-Tompkins QRS & R-peak detection
- Clinical diagnostic metrics (Heart Rate, HRV, AVRR, NN50, Condition)
- Multi-class arrhythmia & anomaly classification with urgency indicator
"""

import os
import sys
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ecg_processor import ECGProcessor
from model import ECGAnomalyClassifier
from generate_samples import load_ecg_file, generate_ecg_signal


class ECGMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Anomaly Detection & Arrhythmia Monitor - Deep Learning")
        self.root.geometry("1280x820")
        self.root.minsize(1050, 700)

        # Style configuration
        self.bg_color = "#121820"
        self.panel_bg = "#1B222C"
        self.card_bg = "#232D3B"
        self.accent_color = "#00D26A"  # Medical monitor green
        self.text_color = "#F0F4F8"
        self.muted_text = "#8A99AD"

        self.root.configure(bg=self.bg_color)

        # Core logic
        self.fs = 300.0
        self.processor = ECGProcessor(fs=self.fs)
        self.classifier = ECGAnomalyClassifier()

        # State
        self.raw_signal = None
        self.filtered_signal = None
        self.r_peaks = np.array([], dtype=int)
        self.integrated_qrs = None
        self.metrics = {}
        self.analysis_result = None

        # Playback animation state
        self.is_playing = False
        self.play_index = 0
        self.window_samples = int(4.0 * self.fs)  # 4-second scrolling window

        self._build_ui()
        self._load_default_signal()

    def _build_ui(self):
        # 1. Top Header Bar
        header_frame = tk.Frame(self.root, bg=self.panel_bg, height=65)
        header_frame.pack(fill=tk.X, side=tk.TOP, padx=10, pady=(10, 5))

        title_lbl = tk.Label(
            header_frame,
            text="❤ ECG ANOMALY DETECTION & CARDIAC MONITOR",
            font=("Segoe UI", 15, "bold"),
            fg=self.accent_color,
            bg=self.panel_bg
        )
        title_lbl.pack(side=tk.LEFT, padx=15, pady=12)

        # Patient Info in Top Bar
        info_frame = tk.Frame(header_frame, bg=self.panel_bg)
        info_frame.pack(side=tk.RIGHT, padx=15, pady=8)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.lbl_patient = tk.Label(
            info_frame,
            text=f"Patient ID: XXX-001  |  Case: 0001  |  Date: {now_str}",
            font=("Segoe UI", 10),
            fg=self.muted_text,
            bg=self.panel_bg
        )
        self.lbl_patient.pack(anchor="e")

        # 2. Controls Toolbar
        toolbar = tk.Frame(self.root, bg=self.card_bg, height=45)
        toolbar.pack(fill=tk.X, side=tk.TOP, padx=10, pady=5)

        btn_load = tk.Button(
            toolbar, text="📂 Load ECG File (.mat / .csv)",
            font=("Segoe UI", 9, "bold"),
            bg="#2A3B50", fg="white", activebackground="#384F6C", activeforeground="white",
            relief=tk.FLAT, padx=12, pady=5, cursor="hand2",
            command=self.on_load_file
        )
        btn_load.pack(side=tk.LEFT, padx=10, pady=6)

        lbl_sample = tk.Label(toolbar, text="Quick Test Sample:", font=("Segoe UI", 9), fg=self.text_color, bg=self.card_bg)
        lbl_sample.pack(side=tk.LEFT, padx=(15, 5))

        self.sample_var = tk.StringVar(value="Normal Sinus Rhythm")
        sample_options = [
            "Normal Sinus Rhythm",
            "Atrial Fibrillation",
            "Bradycardia (Slow)",
            "Tachycardia (Fast)",
            "Noisy Signal"
        ]
        self.combo_sample = ttk.Combobox(
            toolbar, textvariable=self.sample_var, values=sample_options,
            state="readonly", width=22, font=("Segoe UI", 9)
        )
        self.combo_sample.pack(side=tk.LEFT, padx=5)
        self.combo_sample.bind("<<ComboboxSelected>>", self.on_sample_selected)

        btn_analyze = tk.Button(
            toolbar, text="⚡ Run Analysis",
            font=("Segoe UI", 9, "bold"),
            bg="#007ACC", fg="white", activebackground="#0098FF", activeforeground="white",
            relief=tk.FLAT, padx=14, pady=5, cursor="hand2",
            command=self.run_full_analysis
        )
        btn_analyze.pack(side=tk.LEFT, padx=15)

        self.btn_play = tk.Button(
            toolbar, text="▶ Real-Time Monitor",
            font=("Segoe UI", 9, "bold"),
            bg="#00965E", fg="white", activebackground="#00B873", activeforeground="white",
            relief=tk.FLAT, padx=12, pady=5, cursor="hand2",
            command=self.toggle_playback
        )
        self.btn_play.pack(side=tk.LEFT, padx=5)

        self.lbl_status = tk.Label(
            toolbar, text="Status: Ready", font=("Segoe UI", 9, "italic"),
            fg="#61AFEF", bg=self.card_bg
        )
        self.lbl_status.pack(side=tk.RIGHT, padx=15)

        # 3. Main Center Area (Split: Left Plots, Right Metrics Panel)
        main_content = tk.Frame(self.root, bg=self.bg_color)
        main_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left: Matplotlib Canvas (2 plots)
        plot_frame = tk.Frame(main_content, bg=self.panel_bg)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.fig = Figure(figsize=(7.5, 6), facecolor="#1B222C")
        self.ax_ecg = self.fig.add_subplot(211)
        self.ax_peaks = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)

        self._style_axis(self.ax_ecg, "ECG Lead II Waveform (with R-Peak Markers)", "mV")
        self._style_axis(self.ax_peaks, "Pan-Tompkins QRS Energy & Refractory Detection", "Normalized Energy")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right: Clinical Diagnostics & Metrics Panel
        metrics_panel = tk.Frame(main_content, bg=self.panel_bg, width=370)
        metrics_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        metrics_panel.pack_propagate(False)

        self._build_metrics_panel(metrics_panel)

    def _style_axis(self, ax, title, ylabel):
        ax.set_facecolor("#121820")
        ax.set_title(title, color="#E6EDF3", fontsize=10, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, color="#8A99AD", fontsize=9)
        ax.set_xlabel("Time (seconds)", color="#8A99AD", fontsize=9)
        ax.tick_params(colors="#8A99AD", labelsize=8)
        ax.grid(True, color="#253041", linestyle="--", linewidth=0.6, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color("#2E3C50")

    def _build_metrics_panel(self, parent):
        pad = tk.Frame(parent, bg=self.panel_bg)
        pad.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # Title
        hdr = tk.Label(pad, text="CLINICAL DIAGNOSTICS", font=("Segoe UI", 11, "bold"), fg=self.text_color, bg=self.panel_bg)
        hdr.pack(anchor="w", pady=(0, 10))

        # Cards Frame
        cards_frame = tk.Frame(pad, bg=self.panel_bg)
        cards_frame.pack(fill=tk.X)

        # Row 1: Heart Rate & Condition
        self.card_hr = self._create_card(cards_frame, "HEART RATE", "-- BPM", "#00D26A")
        self.card_condition = self._create_card(cards_frame, "RHYTHM RATE TYPE", "--", "#61AFEF")

        # Row 2: HRV (SDNN) & NN50
        self.card_hrv = self._create_card(cards_frame, "HRV (SDNN)", "-- ms", "#E5C07B")
        self.card_nn50 = self._create_card(cards_frame, "NN50 / pNN50", "--", "#98C379")

        # Row 3: Average RR & Peaks Found
        self.card_avrr = self._create_card(cards_frame, "AVG RR INTERVAL", "-- ms", "#C678DD")
        self.card_peaks = self._create_card(cards_frame, "R-PEAKS DETECTED", "--", "#56B6C2")

        # Divider
        tk.Frame(pad, bg="#2E3C50", height=1).pack(fill=tk.X, pady=12)

        # Diagnosis Result Card (Big)
        diag_box = tk.Frame(pad, bg=self.card_bg, padx=12, pady=10, relief=tk.FLAT)
        diag_box.pack(fill=tk.X, pady=(0, 10))

        tk.Label(diag_box, text="CLASSIFICATION RESULT", font=("Segoe UI", 9, "bold"), fg=self.muted_text, bg=self.card_bg).pack(anchor="w")
        self.lbl_diagnosis = tk.Label(diag_box, text="Awaiting Analysis", font=("Segoe UI", 12, "bold"), fg="#61AFEF", bg=self.card_bg)
        self.lbl_diagnosis.pack(anchor="w", pady=(4, 2))

        self.lbl_confidence = tk.Label(diag_box, text="Confidence: --%", font=("Segoe UI", 9), fg=self.muted_text, bg=self.card_bg)
        self.lbl_confidence.pack(anchor="w")

        # Clinical Recommendation / Prognosis
        prog_box = tk.Frame(pad, bg=self.card_bg, padx=12, pady=10)
        prog_box.pack(fill=tk.X, pady=(0, 10))

        tk.Label(prog_box, text="CLINICAL SUGGESTION / PROGNOSIS", font=("Segoe UI", 9, "bold"), fg=self.muted_text, bg=self.card_bg).pack(anchor="w")
        self.lbl_prognosis = tk.Label(
            prog_box, text="No signal analyzed", font=("Segoe UI", 9),
            fg="#F0F4F8", bg=self.card_bg, wraplength=320, justify="left"
        )
        self.lbl_prognosis.pack(anchor="w", pady=(4, 2))

        # Latency & Model info
        self.lbl_latency = tk.Label(
            pad, text="Inference Latency: -- ms  |  Ensemble Model",
            font=("Segoe UI", 8), fg=self.muted_text, bg=self.panel_bg
        )
        self.lbl_latency.pack(anchor="w", pady=(8, 0))

    def _create_card(self, parent, label_text, default_val, val_color):
        card = tk.Frame(parent, bg=self.card_bg, padx=10, pady=8)
        card.pack(fill=tk.X, pady=4)
        tk.Label(card, text=label_text, font=("Segoe UI", 8, "bold"), fg=self.muted_text, bg=self.card_bg).pack(anchor="w")
        val_lbl = tk.Label(card, text=default_val, font=("Segoe UI", 11, "bold"), fg=val_color, bg=self.card_bg)
        val_lbl.pack(anchor="w")
        return val_lbl

    def _load_default_signal(self):
        """Loads sample normal sinus rhythm upon launch"""
        self.on_sample_selected(None)

    def on_sample_selected(self, event):
        choice = self.sample_var.get()
        cond_map = {
            "Normal Sinus Rhythm": "normal",
            "Atrial Fibrillation": "afib",
            "Bradycardia (Slow)": "bradycardia",
            "Tachycardia (Fast)": "tachycardia",
            "Noisy Signal": "noisy"
        }
        cond = cond_map.get(choice, "normal")

        # Stop playback if running
        if self.is_playing:
            self.toggle_playback()

        self.lbl_status.config(text=f"Status: Generating {choice}...", fg="#E5C07B")
        self.root.update_idletasks()

        sig = generate_ecg_signal(condition=cond, duration_sec=30.0, fs=self.fs)
        self.set_signal(sig, f"Sample: {choice}")
        self.run_full_analysis()

    def on_load_file(self):
        """Opens file dialog for .mat, .csv, .npy, .txt files"""
        if self.is_playing:
            self.toggle_playback()

        filepath = filedialog.askopenfilename(
            title="Select ECG Record File",
            filetypes=[
                ("All Supported Files", "*.mat;*.csv;*.npy;*.txt;*.dat"),
                ("MATLAB MAT files (*.mat)", "*.mat"),
                ("CSV files (*.csv)", "*.csv"),
                ("NumPy files (*.npy)", "*.npy"),
                ("Text files (*.txt, *.dat)", "*.txt;*.dat"),
                ("All files (*.*)", "*.*")
            ],
            initialdir=os.path.join(os.getcwd(), "sample_data") if os.path.exists("sample_data") else os.getcwd()
        )
        if not filepath:
            return

        try:
            sig = load_ecg_file(filepath)
            filename = os.path.basename(filepath)
            self.set_signal(sig, f"File: {filename}")
            self.run_full_analysis()
        except Exception as e:
            messagebox.showerror("File Load Error", f"Failed to load ECG file:\n{str(e)}")

    def set_signal(self, ecg: np.ndarray, source_name: str):
        self.raw_signal = np.asarray(ecg, dtype=np.float64).flatten()
        self.lbl_status.config(text=f"Loaded {source_name} ({len(self.raw_signal)} samples, {len(self.raw_signal)/self.fs:.1f}s)", fg="#98C379")
        self.play_index = 0

    def run_full_analysis(self):
        """Executes full Pan-Tompkins QRS detection, HRV metrics, and classification"""
        if self.raw_signal is None or len(self.raw_signal) == 0:
            messagebox.showwarning("No Signal", "Please load an ECG signal first.")
            return

        t_start = time.time()
        self.lbl_status.config(text="Status: Analyzing ECG...", fg="#E5C07B")
        self.root.update_idletasks()

        # 1. Filter signal
        self.filtered_signal = self.processor.filter_signal(self.raw_signal)

        # 2. Pan-Tompkins peak detection
        self.r_peaks, self.integrated_qrs, bandpass = self.processor.pan_tompkins(self.filtered_signal)

        # 3. Compute RR metrics
        self.metrics = self.processor.compute_rr_metrics(self.r_peaks, len(self.filtered_signal))

        # 4. Feature extraction
        feats = self.processor.extract_features(self.raw_signal)

        # 5. Model prediction
        self.analysis_result = self.classifier.predict(feats, self.metrics)
        latency_ms = (time.time() - t_start) * 1000.0

        # Update GUI displays
        self._update_metrics_display(latency_ms)
        self._plot_static_views()

        self.lbl_status.config(text="Status: Analysis Complete", fg="#00D26A")

    def _update_metrics_display(self, latency_ms: float):
        hr = self.metrics.get("heart_rate", 0.0)
        sdnn = self.metrics.get("hrv_sdnn", 0.0)
        nn50 = self.metrics.get("nn50", 0)
        pnn50 = self.metrics.get("pnn50", 0.0)
        avrr = self.metrics.get("avrr", 0.0)
        cond = self.metrics.get("condition", "--")

        self.card_hr.config(text=f"{hr:.1f} BPM")
        self.card_condition.config(text=cond)
        self.card_hrv.config(text=f"{sdnn:.1f} ms")
        self.card_nn50.config(text=f"{nn50} ({pnn50:.1f}%)")
        self.card_avrr.config(text=f"{avrr * 1000.0:.1f} ms")
        self.card_peaks.config(text=f"{len(self.r_peaks)} peaks")

        if self.analysis_result:
            pred = self.analysis_result["prediction"]
            conf = self.analysis_result["confidence"]
            urgency = self.analysis_result["urgency"]
            prognosis = self.analysis_result["prognosis"]

            # Color coding
            color_map = {
                "Normal": "#00D26A",
                "Attention": "#E5C07B",
                "Urgent": "#E06C75",
                "Warning": "#E5C07B"
            }
            diag_color = color_map.get(urgency, "#61AFEF")

            self.lbl_diagnosis.config(text=pred, fg=diag_color)
            self.lbl_confidence.config(text=f"Confidence: {conf:.1f}%  |  Status: {urgency}")
            self.lbl_prognosis.config(text=prognosis, fg="#F0F4F8")

        self.lbl_latency.config(text=f"Latency: {latency_ms:.1f} ms  |  Stage 1 NB + Stage 2 DL Ensemble")

    def _plot_static_views(self):
        """Renders static overview plot of the full signal and peak detection"""
        if self.filtered_signal is None:
            return

        t = np.arange(len(self.filtered_signal)) / self.fs

        # Display first 8-10 seconds for clear waveform visibility
        disp_len = min(len(self.filtered_signal), int(10.0 * self.fs))
        t_disp = t[:disp_len]
        sig_disp = self.filtered_signal[:disp_len]

        # Top Plot: ECG + Marked R-peaks
        self.ax_ecg.clear()
        self._style_axis(self.ax_ecg, "ECG Lead II Waveform with Detected R-Peaks", "Amplitude (mV)")
        self.ax_ecg.plot(t_disp, sig_disp, color="#00D26A", linewidth=1.2, label="Filtered ECG")

        # Overlay R-peaks within display window
        visible_peaks = self.r_peaks[self.r_peaks < disp_len]
        if len(visible_peaks) > 0:
            self.ax_ecg.scatter(
                t[visible_peaks], self.filtered_signal[visible_peaks],
                color="#FF4081", s=45, zorder=5, label="R-Peak"
            )
        self.ax_ecg.legend(loc="upper right", facecolor="#1B222C", edgecolor="#2E3C50", labelcolor="#E6EDF3", fontsize=8)

        # Bottom Plot: Pan-Tompkins Integrated Energy
        self.ax_peaks.clear()
        self._style_axis(self.ax_peaks, "Pan-Tompkins QRS Moving-Window Integration Waveform", "Integrated Energy")
        if self.integrated_qrs is not None and len(self.integrated_qrs) >= disp_len:
            int_disp = self.integrated_qrs[:disp_len]
            self.ax_peaks.plot(t_disp, int_disp, color="#61AFEF", linewidth=1.0, label="QRS Energy")
            if len(visible_peaks) > 0:
                self.ax_peaks.scatter(
                    t[visible_peaks], int_disp[visible_peaks],
                    color="#FF4081", s=35, zorder=5
                )
            self.ax_peaks.legend(loc="upper right", facecolor="#1B222C", edgecolor="#2E3C50", labelcolor="#E6EDF3", fontsize=8)

        self.fig.tight_layout(pad=2.2)
        self.canvas.draw_idle()

    def toggle_playback(self):
        """Starts or stops real-time dynamic monitor playback simulation"""
        if self.filtered_signal is None:
            messagebox.showinfo("No Signal", "Please load an ECG signal first.")
            return

        if self.is_playing:
            self.is_playing = False
            self.btn_play.config(text="▶ Real-Time Monitor", bg="#00965E")
            self.lbl_status.config(text="Status: Monitor Paused", fg="#61AFEF")
            self._plot_static_views()
        else:
            self.is_playing = True
            self.btn_play.config(text="⏹ Stop Monitor", bg="#E06C75")
            self.lbl_status.config(text="Status: Real-Time Cardiac Monitoring Active...", fg="#00D26A")
            self._playback_loop()

    def _playback_loop(self):
        """Simulates continuous live streaming ECG oscilloscope"""
        if not self.is_playing or self.filtered_signal is None:
            return

        N = len(self.filtered_signal)
        win = self.window_samples  # 4 seconds
        step = int(self.fs * 0.08)  # ~80ms scroll step (~12 FPS smooth rendering)

        start_idx = self.play_index
        end_idx = start_idx + win

        if end_idx >= N:
            self.play_index = 0
            start_idx = 0
            end_idx = win

        sig_chunk = self.filtered_signal[start_idx:end_idx]
        t_chunk = np.arange(start_idx, end_idx) / self.fs

        # Visible peaks in window
        peaks_in_win = self.r_peaks[(self.r_peaks >= start_idx) & (self.r_peaks < end_idx)]

        # Top Plot
        self.ax_ecg.clear()
        self._style_axis(self.ax_ecg, "Real-Time ECG Waveform Monitoring (Live Lead II)", "Amplitude (mV)")
        self.ax_ecg.plot(t_chunk, sig_chunk, color="#00D26A", linewidth=1.3)
        if len(peaks_in_win) > 0:
            self.ax_ecg.scatter(
                peaks_in_win / self.fs, self.filtered_signal[peaks_in_win],
                color="#FF4081", s=45, zorder=5
            )
        self.ax_ecg.set_xlim(t_chunk[0], t_chunk[-1])
        y_min = np.min(self.filtered_signal) - 0.2
        y_max = np.max(self.filtered_signal) + 0.2
        self.ax_ecg.set_ylim(y_min, y_max)

        # Bottom Plot
        if self.integrated_qrs is not None:
            int_chunk = self.integrated_qrs[start_idx:end_idx]
            self.ax_peaks.clear()
            self._style_axis(self.ax_peaks, "Real-Time QRS Detection & Wavelet Feature Tracking", "Energy")
            self.ax_peaks.plot(t_chunk, int_chunk, color="#61AFEF", linewidth=1.0)
            if len(peaks_in_win) > 0:
                self.ax_peaks.scatter(
                    peaks_in_win / self.fs, self.integrated_qrs[peaks_in_win],
                    color="#FF4081", s=35, zorder=5
                )
            self.ax_peaks.set_xlim(t_chunk[0], t_chunk[-1])
            self.ax_peaks.set_ylim(0, np.max(self.integrated_qrs) * 1.1 + 0.01)

        self.canvas.draw_idle()
        self.play_index += step

        # Schedule next frame (~65 ms)
        self.root.after(65, self._playback_loop)


def launch():
    root = tk.Tk()
    app = ECGMonitorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
