import os
import re
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

# --- Experiment framework imports (your existing stack) ---
from adriq.experiment import *
from adriq.Optomechanics import *

# ---------- Helpers ----------
def lorentzian(x, A, x0, gamma, y0):
    x = np.asarray(x, dtype=float)
    return y0 + A * (gamma**2) / ((x - x0)**2 + gamma**2)

def fit_lorentzian(x, y, x0_guess=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x0_guess is None:
        x0_guess = x[np.argmax(y)]
    A0 = (np.max(y) - np.min(y)) or 1e-3
    gamma0 = max((x.max() - x.min()) / 10.0, 1e-3)
    y00 = float(np.min(y))
    try:
        popt, pcov = curve_fit(lorentzian, x, y, p0=[A0, float(x0_guess), gamma0, y00], maxfev=20000)
    except Exception:
        popt, pcov = [A0, float(x0_guess), gamma0, y00], np.full((4, 4), np.nan)
    return popt, pcov

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

def save_spectrum(detunings, values, n_brights, n_valids, center_MHz):
    out_csv = next_spectrum_filename(".")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # Add N_bright, N_valid and center frequency
        writer.writerow(["detuning_MHz", "P_dark", "N_bright", "N_valid", "center_MHz"])
        for d, p, nb, nv in zip(detunings, values, n_brights, n_valids):
            writer.writerow([float(d), float(p), int(nb), int(nv), float(center_MHz)])
    print(f"Saved: {out_csv}")
    return out_csv

# ---------- Setup experiment ----------
dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

pulse_sequencer = Pulse_Sequencer()
exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.08, N_Cycles=500)

# Base detunings
exp_sequence.set_detunings(detuning_dict={
    "854 SP1": 0, "854 SP2": 0, "397a": 0, "397c": -20, "866": 0,
    "866 OP": 0, "850 RP": 0, "866 RP": 40, "729 Temp1": 0, "729 Temp2": 0
})

# Cooling / trapping / pumping / probe sequence
exp_sequence.load_cooling(length=1000)
exp_sequence.load_trapping()
exp_sequence.create_section(name="Pump To Ground", duration=2, dds_functions={
    "866 RP": lambda t: 0.2,
    "854 SP1": lambda t: 0.1,
}, pmt_gate_high=True)
exp_sequence.create_section(name="Probe", duration=25, dds_functions={
    "729 Temp1": lambda t: 0.1,
}, pmt_gate_high=True)
exp_sequence.load_measurement()

exp_sequence.build_ram_arrays()
exp_sequence.flash()

exp_runner = Experiment_Runner(
    dds_dict,
    pulse_sequencer,
    timeout=100,
    pmt_threshold=2000,
    expected_fluorescence=8000,
    pulse_expected_fluorescence=2400,
    sp_threshold=None,
    load_timeout=100,
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

    N_bright = int(exp_runner.bright_counts)
    N_valid = int(exp_runner.N_Valid_Pulses or 0)
    P_bright = N_bright / (N_valid or 1)
    P_dark = 1 - P_bright
    # Return raw counts too
    return P_dark, N_bright, N_valid

# ---------- Single scan around minus (±2 MHz) ----------
plt.ion()

center_minus = -23.25  # set your current best estimate here
span = 2.0
step = 0.02  # adjust for speed/resolution
detunings = np.arange(center_minus - span, center_minus + span + 1e-9, step)

fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
line, = ax.plot([], [], 'o', ms=3, label='data')
ax.set_xlabel("Detuning (MHz)")
ax.set_ylabel("P_dark")
ax.set_title("Minus detuning scan (±2 MHz)")
ax.grid(True)

values = []
n_brights = []
n_valids = []
for i, d in enumerate(tqdm(detunings, desc="Scanning minus")):
    p, nb, nv = Dark_Probability(d)
    values.append(p)
    n_brights.append(nb)
    n_valids.append(nv)
    line.set_data(detunings[:i+1], values)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)

detunings = np.asarray(detunings, dtype=float)
values = np.asarray(values, dtype=float)
n_brights = np.asarray(n_brights, dtype=int)
n_valids = np.asarray(n_valids, dtype=int)

csv_path = save_spectrum(detunings, values, n_brights, n_valids, center_minus)

# Fit and overlay
popt, _ = fit_lorentzian(detunings, values, x0_guess=center_minus)
xfit = np.linspace(detunings.min(), detunings.max(), 1200)
ax.plot(xfit, lorentzian(xfit, *popt), '-', lw=2, label='Lorentz fit')
ax.axvline(popt[1], color='k', ls='--', alpha=0.6, label=f"Center = {popt[1]:.6f} MHz")
ax.legend(loc="best")

print(f"Minus-line center (fit): {popt[1]:.6f} MHz")
print(f"Lorentz HWHM gamma: {popt[2]:.6f} MHz  (FWHM ~ {2*popt[2]:.6f} MHz)")

plt.show(block=False)
plt.ioff()
plt.show()