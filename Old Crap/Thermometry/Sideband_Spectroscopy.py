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
pi_time_bsb = 15  # microseconds
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


def gaussian(amplitude, mu, sigma): #Sigma is Half Width 1/e height (multiply by 2*sqrt(ln(2)) to get FWHM ~1.665)
    return lambda t: amplitude * np.exp(-((t - mu)**2) / (sigma**2))

# Lorentzian model + fitter (added)
def lorentzian(x, A, x0, gamma, y0):
    # y = y0 + A * gamma^2 / ((x - x0)^2 + gamma^2)
    x = np.asarray(x, dtype=float)
    return y0 + A * (gamma**2) / ((x - x0)**2 + gamma**2)

def fit_lorentzian(x, y, x0_guess=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x0_guess is None:
        x0_guess = x[np.argmax(y)]
    A0 = (np.max(y) - np.min(y)) or 1e-3
    gamma0 = max((x.max() - x.min())/10.0, 1e-3)
    y00 = float(np.min(y))
    try:
        popt, pcov = curve_fit(
            lorentzian, x, y,
            p0=[A0, float(x0_guess), gamma0, y00],
            maxfev=20000
        )
    except Exception:
        popt, pcov = [A0, float(x0_guess), gamma0, y00], np.full((4, 4), np.nan)
    return popt, pcov

# --- new: sum of two Lorentzians sharing baseline ---
def dual_lorentzian(x, A1, x01, g1, A2, x02, g2, y0):
    x = np.asarray(x, dtype=float)
    return (
        y0
        + A1 * (g1**2) / ((x - x01)**2 + g1**2)
        + A2 * (g2**2) / ((x - x02)**2 + g2**2)
    )

def fit_dual_lorentzian(x, y, x01_guess, x02_guess):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y0 = float(np.min(y))
    A = max((np.max(y) - y0), 1e-3)
    # width guess ~ span/10 or small floor
    width_guess = max((x.max() - x.min()) / 10.0, 1e-3)
    p0 = [A/2, float(x01_guess), width_guess, A/2, float(x02_guess), width_guess, y0]
    try:
        popt, pcov = curve_fit(dual_lorentzian, x, y, p0=p0, maxfev=30000)
    except Exception:
        popt, pcov = p0, np.full((7, 7), np.nan)
    return popt, pcov

dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

pulse_sequencer = Pulse_Sequencer()

exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.08,  N_Cycles=200)


exp_sequence.set_detunings(detuning_dict={"854 SP1": 0, "854 SP2": 0, "397a": 0, "397c": -18, "866": 0, "866 OP": 0, "850 RP": 0, "866 RP": 40, "729 Temp1": 0, "729 Temp2": 0})
exp_sequence.load_cooling(length=1000)
exp_sequence.load_trapping()
# Create op section 
exp_sequence.create_section(name="Pump To Ground", duration=2, dds_functions={
    "866 RP": lambda t: 0.7,
    "854 SP1": lambda t: 0.1,
}, pmt_gate_high=True) #First stage of Optical Pumping
exp_sequence.create_section(name="Probe", duration=pi_time_bsb, dds_functions={
    "729 Temp1": lambda t: 0.5,
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
    pulse_expected_fluorescence= 6000,
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
    exp_runner.experiment_trap_depth = 0.4

    exp_runner.clear_channels()
    exp_runner.start_experiment(N=3)

    N_bright = exp_runner.bright_counts
    P_bright = N_bright / (exp_runner.N_Valid_Pulses or 1)
    P_dark = 1 - P_bright
    return P_dark

def next_spectrum_filename(directory=".", pad=3, prefix="spectrum"):
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.csv$")
    max_n = 0
    try:
        for fname in os.listdir(directory):
            m = pattern.match(fname)
            if m:
                max_n = max(max_n, int(m.group(1)))
    except FileNotFoundError:
        pass
    n = max_n + 1
    return os.path.join(directory, f"{prefix}_{n:0{pad}d}.csv")

# ---------- Modified: two scans, two subplots, fits, and splitting ----------
plt.ion()

# Define detuning ranges centered on red/blue sidebands relative to the minus resonance
center_plus = 22.97
center_minus = -23.275
span = 0.3
step = 0.002

f_sec = 1.6
center_rsb = center_minus - f_sec
center_bsb = center_minus + f_sec

detunings_rsb = np.arange(center_rsb - span, center_rsb + span + 1e-9, step)
detunings_bsb = np.arange(center_bsb - span, center_bsb + span + 1e-9, step)

# replace the "minus/plus" axes with "RSB/BSB"
fig, (ax_rsb, ax_bsb) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

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

def save_spectrum(detunings, values, prefix):
    out_csv = next_spectrum_filename(".", prefix=prefix)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["detuning_MHz", "P_dark"])
        for d, p in zip(detunings, values):
            writer.writerow([float(d), float(p)])
    print(f"Saved: {out_csv}")
    # also save figure as PNG with matching index
    png_name = out_csv.replace(".csv", ".png")
    plt.savefig(png_name, dpi=150)
    print(f"Saved plot: {png_name}")
    return out_csv

# Run RSB scan
dets_rsb, vals_rsb = run_scan(detunings_rsb, ax_rsb, "RSB detunings")
csv_rsb = save_spectrum(dets_rsb, vals_rsb, prefix="rsb_spectrum")

# Fit RSB with two peaks and overlay
# initial guesses: two nearby peaks straddling the nominal RSB center
delta_guess = 0.06  # adjust if needed
popt_rsb, _ = fit_dual_lorentzian(dets_rsb, vals_rsb, x01_guess=center_rsb - delta_guess, x02_guess=center_rsb + delta_guess)
xfit_rsb = np.linspace(dets_rsb.min(), dets_rsb.max(), 800)
ax_rsb.plot(xfit_rsb, dual_lorentzian(xfit_rsb, *popt_rsb), '-', lw=2, label='Dual Lorentz fit')
# mark fitted centers
ax_rsb.axvline(popt_rsb[1], color='r', ls='--', alpha=0.6, label=f"RSB1 = {popt_rsb[1]:.6f} MHz")
ax_rsb.axvline(popt_rsb[4], color='m', ls='--', alpha=0.6, label=f"RSB2 = {popt_rsb[4]:.6f} MHz")
ax_rsb.legend(loc="best")

# Run BSB scan
dets_bsb, vals_bsb = run_scan(detunings_bsb, ax_bsb, "BSB detunings")
csv_bsb = save_spectrum(dets_bsb, vals_bsb, prefix="bsb_spectrum")

# Fit BSB with two peaks and overlay (supports possible non-degeneracy)
popt_bsb, _ = fit_dual_lorentzian(dets_bsb, vals_bsb, x01_guess=center_bsb - delta_guess, x02_guess=center_bsb + delta_guess)
xfit_bsb = np.linspace(dets_bsb.min(), dets_bsb.max(), 800)
ax_bsb.plot(xfit_bsb, dual_lorentzian(xfit_bsb, *popt_bsb), '-', lw=2, label='Dual Lorentz fit')
# mark fitted centers
ax_bsb.axvline(popt_bsb[1], color='r', ls='--', alpha=0.6, label=f"BSB1 = {popt_bsb[1]:.6f} MHz")
ax_bsb.axvline(popt_bsb[4], color='m', ls='--', alpha=0.6, label=f"BSB2 = {popt_bsb[4]:.6f} MHz")
ax_bsb.legend(loc="best")

# Splitting (printed in red on the figure)
# Example: use midpoints of the two fitted centers to define effective splitting
rsb_center_eff = 0.5 * (popt_rsb[1] + popt_rsb[4])
bsb_center_eff = 0.5 * (popt_bsb[1] + popt_bsb[4])
splitting = abs(bsb_center_eff - rsb_center_eff)
fig.suptitle(f"Frequency splitting (BSB-RSB eff.): {splitting:.6f} MHz", color='red', fontsize=12)
print(f"Frequency splitting (BSB-RSB eff.): {splitting:.6f} MHz")
Magnetic_Field_Gauss = splitting / sensitivity
print(f"Magnetic Field: {Magnetic_Field_Gauss:.6f} Gauss")
plt.show(block=False)
plt.ioff()
plt.show()