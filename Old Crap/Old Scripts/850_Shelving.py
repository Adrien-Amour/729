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


# Create the dictionary of DDS instances
calib_directory = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment Files and VIs\AOM calibration VI\Calibration_Files"

dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

pulse_sequencer = Pulse_Sequencer()

exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.08*2,  N_Cycles=1E3)


exp_sequence.set_detunings(detuning_dict={"854 SP1": 0, "854 SP2": 0, "397a": 0, "397c": -20, "866": 0, "866 OP": 0, "850 RP": 0, "866 RP": 40, "729 Temp1": 0})
exp_sequence.set_trapping_parameters(
    trapping_detuning_dict={"397c": -50},
    trapping_amplitude_dict={"397c": 0.45}
)
exp_sequence.create_cooling_section(length=500, amplitude_dict={"397c":0.15,"866 RP":0.4, "854 SP1":1})
exp_sequence.create_section(name="Shelving", duration=60, dds_functions={
    "397c": lambda t: 0.15,
    "850 RP": lambda t: 1,
    # "866 RP": lambda t: 0.7,
}, pmt_gate_high=True) #First stage of Optical Pumping
# --- add 1 ms measurement pulse that forces 397c and 866 into cooling (pin=0) ---
exp_sequence.create_measurement_section(Length=1000, DDS=["397c", "866 RP"], threshold = 4, pmt_gate_high=True)

exp_sequence.build_ram_arrays()
exp_sequence.flash()

exp_runner = Experiment_Runner(
    dds_dict,
    pulse_sequencer,
    timeout=100,
    pmt_threshold=2000,
    expected_fluorescence= 8000,
    pulse_expected_fluorescence= 5000,
    sp_threshold=None,
    load_timeout = 100,
    trigger_mode="ram",
    cavity_lock=False,
)

exp_runner.experiment_trap_depth = 0.6

exp_runner.clear_channels()
exp_runner.start_experiment(N=10)

p_dark = exp_runner.bright_counts / (exp_runner.N_Valid_Pulses or 1)

dark_hist = exp_runner.measurement_hist




dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

pulse_sequencer = Pulse_Sequencer()

exp_sequence2 = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.08*2,  N_Cycles=1E3)


exp_sequence2.set_detunings(detuning_dict={"854 SP1": 0, "854 SP2": 0, "397a": 0, "397c": -20, "866": 0, "866 OP": 0, "850 RP": 0, "866 RP": 40, "729 Temp1": 0})
exp_sequence2.set_trapping_parameters(
    trapping_detuning_dict={"397c": -50},
    trapping_amplitude_dict={"397c": 0.4}
)
exp_sequence2.create_cooling_section(length=500, amplitude_dict={"397c":0.15,"866 RP":0.4, "854 SP1":1})
exp_sequence2.create_section(name="Shelving", duration=60, dds_functions={
    "397c": lambda t: 0.15,
    "850 RP": lambda t: 0,
    "866 RP": lambda t: 0.4,
}, pmt_gate_high=True) #First stage of Optical Pumping
# --- add 1 ms measurement pulse that forces 397c and 866 into cooling (pin=0) ---
exp_sequence2.create_measurement_section(Length=1000, DDS=["397c", "866 RP"], threshold = 0, pmt_gate_high=True)

exp_sequence2.build_ram_arrays()
exp_sequence2.flash()



exp_runner.clear_channels()
exp_runner.start_experiment(N=10)
exp_runner.plot_time_diffs_histogram("signal-f")
p_bright = exp_runner.bright_counts / (exp_runner.N_Valid_Pulses or 1)

bright_hist = exp_runner.measurement_hist

print(f"Dark Probability: {p_dark}")
print(f"Bright Probability: {p_bright}")

import numpy as np
import matplotlib.pyplot as plt
from math import exp, factorial

def poisson_fit_from_hist(hist):
    ks = np.array(sorted(hist.keys()), dtype=int)
    vs = np.array([hist[int(k)] for k in ks], dtype=float)
    total = vs.sum()
    nbar = (ks * vs).sum() / total if total > 0 else 0.0
    pmf = np.array([exp(-nbar) * (nbar**k) / factorial(int(k)) for k in ks])
    return ks, vs, total, nbar, pmf

# Use the histograms you collected
ks_d, vs_d, tot_d, nbar_d, pmf_d = poisson_fit_from_hist(dark_hist)
ks_b, vs_b, tot_b, nbar_b, pmf_b = poisson_fit_from_hist(bright_hist)

plt.figure(figsize=(9,5))
# Dark: grey, transparent
plt.bar(ks_d, vs_d, width=0.8, align='center', edgecolor='black', color='grey', alpha=0.35, label='Dark (data)')
plt.plot(ks_d, tot_d * pmf_d, color='grey', lw=2, label=f'Dark Poisson (n̄={nbar_d:.2f})')
# Bright: blue
plt.bar(ks_b, vs_b, width=0.6, align='center', edgecolor='black', color='tab:blue', alpha=0.7, label='Bright (data)')
plt.plot(ks_b, tot_b * pmf_b, color='tab:blue', lw=2, label=f'Bright Poisson (n̄={nbar_b:.2f})')

plt.xlabel('Counts in measurement window')
plt.ylabel('Occurrences')
plt.title('Bright vs Dark histograms with Poisson fits')
plt.grid(True, axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from math import exp, factorial

def poisson_fit_from_hist(hist):
    ks = np.array(sorted(hist.keys()), dtype=int)
    vs = np.array([hist[int(k)] for k in ks], dtype=float)
    total = vs.sum()
    nbar = (ks * vs).sum() / total if total > 0 else 0.0
    pmf = np.array([exp(-nbar) * (nbar**k) / factorial(int(k)) for k in ks])
    return ks, vs, total, nbar, pmf

# Use the histograms you collected
ks_d, vs_d, tot_d, nbar_d, pmf_d = poisson_fit_from_hist(dark_hist)
ks_b, vs_b, tot_b, nbar_b, pmf_b = poisson_fit_from_hist(bright_hist)

plt.figure(figsize=(9,5))
# Dark: grey, transparent
plt.bar(ks_d, vs_d, width=0.8, align='center', edgecolor='black', color='grey', alpha=0.35, label='Dark (data)')
plt.plot(ks_d, tot_d * pmf_d, color='grey', lw=2, label=f'Dark Poisson (n̄={nbar_d:.2f})')
# Bright: blue
plt.bar(ks_b, vs_b, width=0.6, align='center', edgecolor='black', color='tab:blue', alpha=0.7, label='Bright (data)')
plt.plot(ks_b, tot_b * pmf_b, color='tab:blue', lw=2, label=f'Bright Poisson (n̄={nbar_b:.2f})')

plt.xlabel('Counts in measurement window')
plt.ylabel('Occurrences')
plt.title('Bright vs Dark histograms with Poisson fits')
plt.grid(True, axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

thr = int(input("Enter threshold (counts): ").strip())
fidelity = 1 - 0.5 * (sum(v for k,v in dark_hist.items() if k >= thr)/sum(dark_hist.values()) + sum(v for k,v in bright_hist.items() if k < thr)/sum(bright_hist.values()))
print(fidelity)
thr = int(input("Enter threshold (counts): ").strip())
fidelity = 1 - 0.5 * (sum(v for k,v in dark_hist.items() if k >= thr)/sum(dark_hist.values()) + sum(v for k,v in bright_hist.items() if k < thr)/sum(bright_hist.values()))
print(fidelity)
