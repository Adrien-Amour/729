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
import matplotlib.pyplot as plt  # added
import re
from scipy.optimize import curve_fit  # added

resonance = -23.25
probe_time_us = 20
probe_power = 0.15
def zeemanFS(J, S, L, B):
    muB = 9.274078e-24   # Bohr magneton in J/T
    hbar = 1.054572669125e-34  # Planck constant / 2pi in J·s
    gJ = 1 + (J*(J+1) + S*(S+1) - L*(L+1)) / (2*J*(J+1))
    w = gJ * muB * B / hbar      # angular frequency in rad/s
    return w/1E6 #convert to MHz
 
B = 1e-4  # Tesla
 
# w1 for J=1/2, S=1/2, L=0
w1 = zeemanFS(1/2, 1/2, 0, B)
# w5 for J=5/2, S=1/2, L=2
w5 = zeemanFS(5/2, 1/2, 2, B)
Delta_m2 = 2.5 * w5 - 0.5*w1
sensitivity = 2*Delta_m2/(2*np.pi)  # MHz/Gauss

dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

pulse_sequencer = Pulse_Sequencer()

exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.08,  N_Cycles=500)

exp_sequence.set_detunings(detuning_dict={"854 SP1": 0, "854 SP2": 0, "397a": 0, "397c": -20, "866": 0, "866 OP": 30, "850 RP": 0, "866 RP": 40, "729 Temp1": 0, "729 Temp2": 0})

# Use new cooling/trapping API and sections layout
exp_sequence.load_cooling(length=1000)
exp_sequence.load_trapping()

# Pump to stretch
exp_sequence.create_section(name="Pump To Stretch", duration=18, dds_functions={
    "397c": lambda t: 0.35,
    "866 OP": lambda t: 0.5,
    "854 SP1": lambda t: 0.2,
}, pmt_gate_high=True)

# Pump to ground (turn on after 1 μs)
exp_sequence.create_section(name="Pump To Ground", duration=4, dds_functions={
    "850 RP":  lambda t: 0.35 if t >= 1.0 else 0.0,
    "854 SP1": lambda t: 0.20 if t >= 1.0 else 0.0,
}, pmt_gate_high=True)

# Probe
exp_sequence.create_section(name="Probe", duration=probe_time_us, dds_functions={
    "729 Temp1": lambda t: probe_power,
}, pmt_gate_high=True)

# Standard measurement
exp_sequence.load_measurement()


exp_sequence.build_ram_arrays()
exp_sequence.flash()

exp_runner = Experiment_Runner(
    dds_dict,
    pulse_sequencer,
    timeout=100,
    pmt_threshold=2000,
    expected_fluorescence= 8000,
    pulse_expected_fluorescence= 6000,
    sp_threshold=None,
    load_timeout = 100,
    trigger_mode="ram",
    cavity_lock=False,
)


def Dark_Probability():
    # Only rerun; avoid edit_detunings/build/flash
    exp_runner.clear_channels()
    exp_runner.experiment_trap_depth = 0.6
    exp_runner.clear_channels()
    exp_runner.start_experiment(N=1)

    N_bright = exp_runner.bright_counts
    P_bright = N_bright / (exp_runner.N_Valid_Pulses or 1)
    P_dark = 1 - P_bright
    return P_dark

def next_spectrum_filename(directory=".", pad=3):
    pattern = re.compile(r"^spectrum_(\d+)\.csv$")
    max_n = 0
    try:
        for fname in os.listdir(directory):
            m = pattern.match(fname)
            if m:
                max_n = max(max_n, int(m.group(1)))
    except FileNotFoundError:
        pass
    n = max_n + 1
    return os.path.join(directory, f"spectrum_{n:0{pad}d}.csv")

# ---------- Modified: continuous alignment at fixed resonance ----------
plt.ion()
res_detuning = resonance  # stay fixed on resonance

# Set detuning once, then build/flash once
exp_sequence.edit_detunings(detuning_dict={"729 Temp1": res_detuning})
exp_sequence.build_ram_arrays()
exp_sequence.flash()

fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
line, = ax.plot([], [], 'o-', ms=3, label='P_dark')
ax.set_xlabel("Iteration")
ax.set_ylabel("P_dark")
ax.set_ylim(-0.05, 1.05)
ax.set_title(f"Alignment at {res_detuning:.3f} MHz")
ax.grid(True)

vals = []
i = 0
try:
    while True:
        p = Dark_Probability()
        vals.append(float(p))
        i += 1

        # show only the last 40 points
        start = max(0, i - 40)
        x = np.arange(start + 1, i + 1)
        y = np.array(vals[start:i])

        line.set_data(x, y)
        ax.set_xlim(max(1, i - 39), max(40, i + 1))
        ax.set_title(f"Alignment at {res_detuning:.3f} MHz | P_dark={p:.3f} | showing last {len(y)}")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.01)
except KeyboardInterrupt:
    print("Alignment loop stopped by user.")

plt.ioff()
plt.show()