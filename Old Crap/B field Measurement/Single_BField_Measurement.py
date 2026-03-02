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

def zeemanFS(J, S, L, B):
    muB = 9.274078e-24   # Bohr magneton in J/T
    hbar = 1.054572669125e-34  # Planck constant / 2pi in J·s
    gJ = 1 + (J*(J+1) + S*(S+1) - L*(L+1)) / (2*J*(J+1))
    w = gJ * muB * B / hbar      # angular frequency in rad/s
    return w/1E6 #convert to MHz
 
B = 1e-4  # Tesla to gauss
 
# w1 for J=1/2, S=1/2, L=0
w1 = zeemanFS(1/2, 1/2, 0, B)
# w5 for J=5/2, S=1/2, L=2
w5 = zeemanFS(5/2, 1/2, 2, B)
Delta_m2 = 2.5 * w5 - 0.5*w1
sensitivity = 2*Delta_m2/(2*np.pi)  # MHz/Gauss


def gaussian(amplitude, mu, sigma): #Sigma is Half Width 1/e height (multiply by 2*sqrt(ln(2)) to get FWHM ~1.665)
    return lambda t: amplitude * np.exp(-((t - mu)**2) / (sigma**2))

# sinc^2 model + fitter
def sinc2(x, A, x0, w, y0):
    # Uses normalized sinc: np.sinc(z) = sin(pi*z)/(pi*z)
    # Peak at x0; w controls width; A is amplitude; y0 is offset
    x = np.asarray(x, dtype=float)
    return y0 + A * np.sinc((x - x0) / w)**2

def fit_sinc2(x, y, x0_guess=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x0_guess is None:
        x0_guess = x[np.argmax(y)]
    A0 = (np.max(y) - np.min(y)) or 1e-3
    # width guess ~ span/10; avoid zero
    w0 = max((x.max() - x.min()) / 10.0, 1e-3)
    y00 = float(np.min(y))
    try:
        popt, pcov = curve_fit(
            sinc2, x, y,
            p0=[A0, float(x0_guess), w0, y00],
            maxfev=20000
        )
    except Exception:
        popt, pcov = [A0, float(x0_guess), w0, y00], np.full((4, 4), np.nan)
    return popt, pcov

dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

pulse_sequencer = Pulse_Sequencer()

exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.08,  N_Cycles=500)


exp_sequence.set_detunings(detuning_dict={"854 SP1": 0, "854 SP2": 0, "397a": 0, "397c": -20, "866": 0, "866 OP": 30, "850 RP": 0, "866 RP": 40, "729 Temp1": 0, "729 Temp2": 0})

exp_sequence.load_cooling(length=2000)
exp_sequence.load_trapping()

# Create op section 
exp_sequence.create_section(name="Pump To Stretch", duration=18, dds_functions={
    "397c": lambda t: 0.35,
    "866 OP": lambda t: 0.5,
    "854 SP1": lambda t: 0.2,
}, pmt_gate_high=True) #First stage of Optical Pumping

exp_sequence.create_section(name="Pump To Ground", duration=4, dds_functions={
    "850 RP":  lambda t: 0.35 if t >= 1.0 else 0.0,
    "854 SP1": lambda t: 0.20 if t >= 1.0 else 0.0,
}, pmt_gate_high=True) #First stage of Optical Pumping

exp_sequence.create_section(name="Probe", duration=15, dds_functions={
    "729 Temp1": lambda t: 0.1,
}, pmt_gate_high=True) #First stage of Optical Pumping

exp_sequence.load_measurement()


exp_sequence.build_ram_arrays()
exp_sequence.flash()

exp_runner = Experiment_Runner(
    dds_dict,
    pulse_sequencer,
    timeout=100,
    pmt_threshold=2000,
    expected_fluorescence= 8000,
    pulse_expected_fluorescence= 7000,
    sp_threshold=None,
    load_timeout = 100,
    trigger_mode="ram",
    cavity_lock=False,
)
exp_runner.measure_expected_fluorescence()



def Dark_Probability(detuning):
    exp_sequence.edit_detunings(detuning_dict={"729 Temp1": detuning})
    exp_sequence.build_ram_arrays()
    exp_sequence.flash()
    exp_runner.clear_channels()
    exp_runner.experiment_trap_depth = 0.55

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

# ---------- Modified: two scans, two subplots, fits, and splitting ----------
plt.ion()

# Define detuning ranges
center_plus = 22.97
center_minus = -23.35
span = 2
step = 0.005*4
detunings_plus = np.arange(center_plus - span, center_plus + span + 1e-9, step)
detunings_minus = np.arange(center_minus - span, center_minus + span + 1e-9, step)

fig, (ax_minus, ax_plus) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

def run_scan(detunings, ax, title):
    line, = ax.plot([], [], 'o', ms=3, label='data')
    ax.set_xlabel("Detuning (MHz)")
    ax.set_ylabel("P_dark")
    ax.set_title(title)
    ax.grid(True)
    vals = []
    for i, d in enumerate(tqdm(detunings, desc=f"Scanning {title}")):
        p = Dark_Probability(d)
        vals.append(p)
        line.set_data(detunings[:i+1], vals)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.01)
    return np.array(detunings, dtype=float), np.array(vals, dtype=float)

def save_spectrum(detunings, values):
    out_csv = next_spectrum_filename(".")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["detuning_MHz", "P_dark"])
        for d, p in zip(detunings, values):
            writer.writerow([float(d), float(p)])
    print(f"Saved: {out_csv}")
    return out_csv

# Run minus scan
dets_minus, vals_minus = run_scan(detunings_minus, ax_minus, "Minus detunings")
csv_minus = save_spectrum(dets_minus, vals_minus)

# Fit minus and overlay
popt_minus, _ = fit_sinc2(dets_minus, vals_minus, x0_guess=center_minus)
xfit_minus = np.linspace(dets_minus.min(), dets_minus.max(), 600)
ax_minus.plot(xfit_minus, sinc2(xfit_minus, *popt_minus), '-', lw=2, label='sinc^2 fit')
ax_minus.axvline(popt_minus[1], color='k', ls='--', alpha=0.6, label=f"Center = {popt_minus[1]:.6f} MHz")
ax_minus.legend(loc="best")

# Run plus scan
dets_plus, vals_plus = run_scan(detunings_plus, ax_plus, "Plus detunings")
csv_plus = save_spectrum(dets_plus, vals_plus)

# Fit plus and overlay
popt_plus, _ = fit_sinc2(dets_plus, vals_plus, x0_guess=center_plus)
xfit_plus = np.linspace(dets_plus.min(), dets_plus.max(), 600)
ax_plus.plot(xfit_plus, sinc2(xfit_plus, *popt_plus), '-', lw=2, label='sinc^2 fit')
ax_plus.axvline(popt_plus[1], color='k', ls='--', alpha=0.6, label=f"Center = {popt_plus[1]:.6f} MHz")
ax_plus.legend(loc="best")

# Splitting (printed in red on the figure)
splitting = abs(popt_plus[1] - popt_minus[1])
fig.suptitle(f"Frequency splitting: {splitting:.6f} MHz", color='red', fontsize=12)
print(f"Frequency splitting: {splitting:.6f} MHz")
Magnetic_Field_Gauss = splitting / sensitivity
print(f"Magnetic Field: {Magnetic_Field_Gauss:.6f} Gauss")
plt.show(block=False)
plt.ioff()
plt.show()