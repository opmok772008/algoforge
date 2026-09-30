"""
data/loaders.py — PhysioNet record loader with synthetic fallback.

Tries: MIT-BIH Arrhythmia (mitdb), CU Ventricular Tachyarrhythmia (cudb),
       MIT-BIH Malignant Ventricular Ectopy (vfdb).
Falls back to synthetic data if wfdb download fails (offline mode).

Split is BY RECORD (not by window) to prevent data leakage.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from biosig_fuzzy_pso.config import DEFAULT_CONFIG, EvalConfig, SAMPLE_RATE
from biosig_fuzzy_pso.data.synthetic import build_synthetic_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------
# MIT-BIH rhythm annotation codes -> internal label
RHYTHM_LABEL_MAP: Dict[str, int] = {
    "(N": 0,    # Normal sinus
    "(SB": 0,   # Sinus bradycardia
    "(ST": 0,   # Sinus tachycardia (non-lethal)
    "(VT": 1,   # Ventricular tachycardia
    "(VFL": 2,  # Ventricular flutter/fibrillation
    "(VF": 2,   # Ventricular fibrillation
    "(VFIB": 2, # Ventricular fibrillation (alias)
}

# Default label if rhythm annotation not found
DEFAULT_LABEL = 0

# Database identifiers and their PhysioNet DB names
DATABASES: Dict[str, Dict] = {
    "mitdb": {
        "pn_dir": "mitdb",
        "records": [
            "100", "101", "102", "103", "104", "105", "106", "107",
            "108", "109", "111", "112", "113", "114", "115", "116",
            "117", "118", "119", "121", "122", "123", "124", "200",
            "201", "202", "203", "205", "207", "208", "209", "210",
            "212", "213", "214", "215", "217", "219", "220", "221",
            "222", "223", "228", "230", "231", "232", "233", "234",
        ],
        "fs": 360,
    },
    "cudb": {
        "pn_dir": "cudb",
        "records": [
            "cu01", "cu02", "cu03", "cu04", "cu05", "cu06", "cu07",
            "cu08", "cu09", "cu10", "cu11", "cu12", "cu13", "cu14",
            "cu15", "cu16", "cu17", "cu18", "cu19", "cu20",
        ],
        "fs": 250,
    },
    "vfdb": {
        "pn_dir": "vfdb",
        "records": [
            "418", "419", "420", "421", "422", "423", "424", "425",
            "426", "427", "428", "429", "430", "431", "432", "433",
            "434", "435", "436", "437",
        ],
        "fs": 250,
    },
}


# ---------------------------------------------------------------------------
# Single-record loader
# ---------------------------------------------------------------------------

import os

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")

def _load_wfdb_record(
    record_name: str,
    pn_dir: str,
    target_fs: int = SAMPLE_RATE,
    max_duration_sec: float = 120.0,
) -> Optional[Dict]:
    """
    Load one PhysioNet record with wfdb, resampling to target_fs.
    Caches loaded records as .npz to disk for instant subsequent loads.

    Returns None on any failure.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{pn_dir}_{record_name}_{int(max_duration_sec)}_{target_fs}.npz")

    if os.path.exists(cache_path):
        try:
            with np.load(cache_path) as data:
                return {
                    "signal": data["signal"].astype(np.float32),
                    "r_peaks": data["r_peaks"].astype(np.int32),
                    "label": int(data["label"]),
                    "record": str(data["record"]),
                    "fs": int(data["fs"]),
                    "source": "physionet",
                }
        except Exception as exc:
            logger.debug("Failed reading cache %s: %s", cache_path, exc)

    try:
        import wfdb
        from scipy.signal import resample_poly
        from math import gcd

        # Load header to get src_fs
        header = wfdb.rdheader(record_name, pn_dir=pn_dir)
        src_fs = int(header.fs)
        sampto = int(max_duration_sec * src_fs) if max_duration_sec > 0 else None

        record = wfdb.rdrecord(record_name, pn_dir=pn_dir, sampfrom=0, sampto=sampto)
        ann = wfdb.rdann(record_name, "atr", pn_dir=pn_dir, sampfrom=0, sampto=sampto)

        ecg = record.p_signal[:, 0].astype(np.float32)  # first channel

        # Resample if necessary
        if src_fs != target_fs:
            g = gcd(target_fs, src_fs)
            up, down = target_fs // g, src_fs // g
            ecg = resample_poly(ecg, up, down).astype(np.float32)
            scale = target_fs / src_fs
            r_samples = (ann.sample * scale).astype(np.int32)
        else:
            r_samples = ann.sample.astype(np.int32)

        # R-peak annotations (normal and beat symbols)
        beat_symbols = set("NLRBAaJSVrFejnE/fQ?")
        r_peaks = np.array(
            [s for s, sym in zip(r_samples, ann.symbol) if sym in beat_symbols],
            dtype=np.int32,
        )

        # Rhythm label: scan rhythm annotations
        label = DEFAULT_LABEL
        for note in ann.aux_note:
            note_clean = note.strip().rstrip("\x00")
            if note_clean in RHYTHM_LABEL_MAP:
                label = RHYTHM_LABEL_MAP[note_clean]
                break

        res = {
            "signal": ecg,
            "r_peaks": r_peaks,
            "label": label,
            "record": record_name,
            "fs": target_fs,
            "source": "physionet",
        }

        # Save to disk cache
        try:
            np.savez_compressed(
                cache_path,
                signal=ecg,
                r_peaks=r_peaks,
                label=np.int32(label),
                record=record_name,
                fs=np.int32(target_fs),
            )
        except Exception:
            pass

        return res

    except Exception as exc:
        logger.debug("Failed to load %s/%s: %s", pn_dir, record_name, exc)
        return None


# ---------------------------------------------------------------------------
# Dataset loader (multiple databases)
# ---------------------------------------------------------------------------

def load_dataset(
    databases: Optional[List[str]] = None,
    max_records_per_db: int = 3,
    target_fs: int = SAMPLE_RATE,
    test_fraction: float = 0.3,
    seed: int = 42,
    duration_per_class_sec: float = 30.0,
    max_duration_sec: float = 120.0,
) -> Dict:
    """
    Load ECG records from PhysioNet databases, with automatic synthetic fallback.

    Parameters
    ----------
    databases         : list of keys from DATABASES; defaults to ["mitdb", "cudb"]
    max_records_per_db: maximum records to load per database (to save time)
    target_fs         : resample target sample rate
    test_fraction     : fraction of records held out for test (by record)
    seed              : RNG seed for train/test split
    max_duration_sec  : duration per record in seconds (default 120s)

    Returns
    -------
    dict with keys:
      'train', 'test': lists of record dicts
      'source': 'physionet' or 'synthetic'
    """
    if databases is None:
        databases = ["mitdb", "cudb"]

    rng = np.random.default_rng(seed)
    all_records: List[Dict] = []
    physionet_loaded = 0

    for db_name in databases:
        if db_name not in DATABASES:
            logger.warning("Unknown database: %s", db_name)
            continue
        db_info = DATABASES[db_name]
        record_list = list(db_info["records"])
        rng.shuffle(record_list)
        record_list = record_list[:max_records_per_db]

        for rec_name in record_list:
            rec = _load_wfdb_record(rec_name, db_info["pn_dir"], target_fs, max_duration_sec)
            if rec is not None:
                all_records.append(rec)
                physionet_loaded += 1
                logger.info("Loaded PhysioNet record: %s/%s", db_name, rec_name)

    if physionet_loaded == 0:
        logger.warning(
            "PhysioNet unavailable (offline?). "
            "Falling back to SYNTHETIC data. "
            "All metrics computed on synthetic signals."
        )
        synth = build_synthetic_dataset(
            duration_per_class_sec=duration_per_class_sec,
            seed=seed,
        )
        all_records = synth["records"]
        source = "synthetic"
    else:
        source = "physionet"

    # Split by record (never by window)
    indices = np.arange(len(all_records))
    rng.shuffle(indices)
    n_test = max(1, int(len(indices) * test_fraction))
    test_idx = set(indices[:n_test].tolist())
    train_records = [all_records[i] for i in range(len(all_records)) if i not in test_idx]
    test_records = [all_records[i] for i in test_idx]

    logger.info(
        "Dataset ready — source=%s  train=%d  test=%d",
        source, len(train_records), len(test_records),
    )

    return {
        "train": train_records,
        "test": test_records,
        "source": source,
    }
