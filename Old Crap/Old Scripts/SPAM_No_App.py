import numpy as np
import pyqtgraph as pg
import plotly.graph_objects as go
from pyqtgraph.Qt import QtGui, QtWidgets
from adriq.experiment import *
from adriq.Optomechanics import *
from adriq.tdc_functions import filter_trailing_zeros, compute_time_diffs, filter_runs
import csv
import os
from tqdm import tqdm
import time
import webbrowser
import itertools

# ...existing code...

calib_directory = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment Files and VIs\AOM calibration VI\Calibration_Files"

# ----------------------------
# Scan configuration (edit these)
# ----------------------------
PROBE_CHANNEL = "397c"          # channel whose amplitude you want to treat as "probe power"
SCAN_DETUNING_KEY = "397c"      # channel whose detuning you want to scan

# "Minus peak" detuning range (adjust to your feature)
DETUNING_START = -30.0
DETUNING_STOP = -5.0
DETUNING_STEP = 0.5
scan_detunings = np.arange(DETUNING_START, DETUNING_STOP + 1e-12, DETUNING_STEP)

probe_powers = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

N_PER_POINT = 3  # shots per (power, detuning) point
OUTDIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
OUTCSV = os.path.join(OUTDIR, f"minus_peak_scan_{time.strftime('%Y%m%d_%H%M%S')}.csv")

def hist_mean(hist: dict) -> float:
    """Histogram is assumed to be {counts:int -> occurrences:int}."""
    if not hist:
        return float("nan")
    tot = sum(hist.values())
    if tot <= 0:
        return float("nan")
    return sum(k * v for k, v in hist.items()) / tot

def build_sequence(dds_dict, pulse_sequencer, detuning_397c: float, probe_power: float):
    exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.08 * 2, N_Cycles=1e3)

    # Base detunings; only scan the minus-peak side for SCAN_DETUNING_KEY
    detuning_dict = {
        "854 SP1": 0,
        "854 SP2": 0,
        "397a": 0,
        "397c": detuning_397c,
        "866": 0,
        "866 OP": 0,
        "850 RP": 0,
        "866 RP": 35,
        "729 Temp1": 0,
    }
    exp_sequence.set_detunings(detuning_dict=detuning_dict)

    exp_sequence.set_trapping_parameters(
        trapping_detuning_dict={"397c": -50},
        trapping_amplitude_dict={"397c": 0.45},
    )

    exp_sequence.load_cooling(length=2000)

    # Keep your existing structure; adjust if your "minus peak" requires a different prep.
    exp_sequence.create_section(
        name="Shelving",
        duration=60,
        dds_functions={
            "397c": lambda t: 0.15,
            "850 RP": lambda t: 1,
        },
        pmt_gate_high=True,
    )

    # Measurement pulse; treat PROBE_CHANNEL amplitude as "probe power"
    amplitudes = {
        PROBE_CHANNEL: float(probe_power),
        "866 RP": 0.22,
    }
    exp_sequence.create_measurement_section(
        Length=3000,
        amplitudes_dict=amplitudes,
        threshold=4,
        pmt_gate_high=True,
    )

    exp_sequence.build_ram_arrays()
    exp_sequence.flash()
    return exp_sequence

# ----------------------------
# Hardware setup (unchanged intent)
# ----------------------------
dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")
pulse_sequencer = Pulse_Sequencer()

exp_runner = Experiment_Runner(
    dds_dict,
    pulse_sequencer,
    timeout=100,
    pmt_threshold=2000,
    expected_fluorescence=8000,
    pulse_expected_fluorescence=6000,
    sp_threshold=None,
    load_timeout=100,
    trigger_mode="ram",
    cavity_lock=False,
)
exp_runner.experiment_trap_depth = 0.55

# ----------------------------
# 2D scan: (probe power) x (minus-peak detuning)
# ----------------------------
with open(OUTCSV, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "probe_power",
            "detuning",
            "attempted",
            "valid",
            "valid_pct",
            "mean_counts",
        ],
    )
    w.writeheader()

    for pwr in probe_powers:
        for det in scan_detunings:
            # Build/flash for this point
            _seq = build_sequence(dds_dict, pulse_sequencer, detuning_397c=float(det), probe_power=float(pwr))

            exp_runner.clear_channels()
            exp_runner.start_experiment(N=N_PER_POINT)

            attempted = int(getattr(exp_runner, "N_Requested_Pulses", N_PER_POINT))
            valid = int(getattr(exp_runner, "N_Valid_Pulses", 0))
            valid_pct = 100.0 * valid / (attempted or 1)

            mean_counts = hist_mean(getattr(exp_runner, "measurement_hist", {}) or {})

            w.writerow(
                dict(
                    probe_power=float(pwr),
                    detuning=float(det),
                    attempted=attempted,
                    valid=valid,
                    valid_pct=float(valid_pct),
                    mean_counts=float(mean_counts),
                )
            )

print(f"Wrote scan results to: {OUTCSV}")

# Optional: quick plot (mean counts vs detuning) for each probe power
try:
    import matplotlib.pyplot as plt

    # Reload CSV into arrays for plotting (keeps it simple and matches what you saved)
    data = {}
    with open(OUTCSV, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            p = float(row["probe_power"])
            data.setdefault(p, {"det": [], "mean": [], "valid_pct": []})
            data[p]["det"].append(float(row["detuning"]))
            data[p]["mean"].append(float(row["mean_counts"]))
            data[p]["valid_pct"].append(float(row["valid_pct"]))

    plt.figure(figsize=(9, 5))
    for p in sorted(data.keys()):
        plt.plot(data[p]["det"], data[p]["mean"], marker="o", lw=1, label=f"P={p:g}")
    plt.xlabel(f"{SCAN_DETUNING_KEY} detuning (minus peak)")
    plt.ylabel("Mean counts (measurement window)")
    plt.title("Minus-peak scan repeated for probe powers")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Plot skipped: {e}")

# ...existing code...