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
import matplotlib.pyplot as plt  # added
import re



dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

pulse_sequencer = Pulse_Sequencer()

exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.08,  N_Cycles=500)


exp_sequence.set_detunings(detuning_dict={"854 SP1": 0, "854 SP2": 0, "397a": 0, "397c": -20, "866": 0, "866 OP": 0, "850 RP": 0, "866 RP": 40, "729 Temp1": 0, "729 Temp2": 0})
exp_sequence.set_trapping_parameters(
    trapping_detuning_dict={"397c": -50},
    trapping_amplitude_dict={"397c": 0.4}
)
exp_sequence.create_cooling_section(length=500, amplitude_dict={"397c":0.15,"866 RP":0.4, "854 SP1":1})
# Create a Gaussian output function for DDS 397a

# Create op section 
exp_sequence.create_section(name="Pump To Ground", duration=2, dds_functions={
    "866 RP": lambda t: 0.7,
}, pmt_gate_high=True) #First stage of Optical Pumping
exp_sequence.create_section(name="Probe", duration=12, dds_functions={
    "729 Temp1": lambda t: 1,
}, pmt_gate_high=True) #First stage of Optical Pumping


# --- add 0.505 ms measurement pulse that forces 397c and 866 into cooling (pin=0) ---
exp_sequence.create_measurement_section(Length=1000, DDS=["397c", "866 RP"], pmt_gate_high=True, threshold=4)

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


def Dark_Probability(detuning):
    exp_sequence.edit_detunings(detuning_dict={"729 Temp1": detuning})
    exp_sequence.build_ram_arrays()
    exp_sequence.flash()
    exp_runner.clear_channels()
    exp_runner.experiment_trap_depth = 0.6

    exp_runner.clear_channels()
    exp_runner.start_experiment(N=3)

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

# Enable interactive plotting and prepare live plot
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], marker='o')  # start with empty data
ax.set_xlabel("Detuning (MHz)")
ax.set_ylabel("Fluorescence Counts")
ax.set_title("Fluorescence vs Detuning")
ax.grid(True)
plt.show(block=False)  # ensure non-blocking show

# Replace the detunings definition to include extra points
center = 22.97
span = 0.5
step = 0.002
detunings_plus = np.arange(center - span, center + span + 1e-9, step)
secular_f = 1.7  # MHz
center = -23.23
span = 0.5
step = 0.002
detunings_minus_carrier = np.arange(center - span, center + span + 1e-9, step)
detunings_minus_rsb = np.arange(center - span- secular_f, center + span - secular_f + 1e-9, step)
detunings_minus_bsb = np.arange(center - span+ secular_f, center + span + secular_f + 1e-9, step)

# Bridges every 0.1 MHz between carrier and sidebands (exclude endpoints to avoid duplicates)
bridge_step = 0.1  # MHz (100 kHz)
bridge_to_rsb = np.arange(center, center - secular_f, -bridge_step)[1:]  # from carrier down to RSB
bridge_to_bsb = np.arange(center, center + secular_f, bridge_step)[1:]   # from carrier up to BSB

detunings_minus = np.concatenate((
    detunings_minus_carrier,
    detunings_minus_rsb,
    detunings_minus_bsb,
    bridge_to_rsb,
    bridge_to_bsb
))
detunings_minus.sort()
# detunings = np.concatenate((detunings_minus, detunings_plus))
# detunings.sort()

Dark_Probability_values = []

for detuning in tqdm(detunings_minus, desc="Scanning detunings"):  # if UI still freezes, use tqdm(..., disable=True)
    P_dark = Dark_Probability(detuning)
    Dark_Probability_values.append(P_dark)
    # Update line data incrementally
    line.set_data(detunings_minus[:len(Dark_Probability_values)], Dark_Probability_values)
    ax.relim()
    ax.autoscale_view()

    fig.canvas.draw_idle()
    fig.canvas.flush_events()  # ensure GUI event processing
    plt.pause(0.01)  # small pause to allow GUI event loop to update


# Save CSV and figure with unique, iterating filenames
out_csv = next_spectrum_filename(".")
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["detuning_MHz", "P_dark"])
    for d, p in zip(detunings, Dark_Probability_values):
        writer.writerow([float(d), float(p)])
print(f"Saved: {out_csv}")




# Optionally keep window open after finishing the scan
plt.ioff()
plt.show()