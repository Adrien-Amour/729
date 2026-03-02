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

    # Robust baseline + amplitude guesses (less sensitive to outliers)
    y10 = float(np.percentile(y, 10))
    y50 = float(np.percentile(y, 50))
    y90 = float(np.percentile(y, 90))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    yrng = max(y_max - y_min, 1e-6)

    y0_0 = y10
    A0 = max(y90 - y0_0, 1e-6)  # enforce non-negative amplitude guess

    # Center guess: max, optionally refined by quadratic interpolation (3-point)
    imax = int(np.argmax(y))
    x0_0 = float(x[imax]) if x0_guess is None else float(x0_guess)
    if 0 < imax < (len(x) - 1) and x0_guess is None:
        x1, x2, x3 = x[imax - 1], x[imax], x[imax + 1]
        y1, y2, y3 = y[imax - 1], y[imax], y[imax + 1]
        denom = (y1 - 2.0 * y2 + y3)
        if np.isfinite(denom) and abs(denom) > 1e-12:
            # vertex offset in index units
            delta = 0.5 * (y1 - y3) / denom
            # map offset onto x spacing (assumes near-uniform sampling locally)
            dx_local = (x3 - x1) * 0.5
            x0_0 = float(x2 + delta * dx_local)

    # Width guess from an approximate FWHM (if possible)
    # For sinc^2: half-max at |z| ≈ 0.443 (where (sinc(z))^2 = 0.5), so FWHM ≈ 2*0.443*w
    half_level = y0_0 + 0.5 * A0
    mask = y >= half_level
    w0 = None
    if np.any(mask) and np.any(~mask):
        idx = np.where(mask)[0]
        fwhm = float(x[idx[-1]] - x[idx[0]])
        if np.isfinite(fwhm) and fwhm > 0:
            w0 = fwhm / (2.0 * 0.443)

    if w0 is None:
        w0 = max((x.max() - x.min()) / 10.0, 1e-6)

    # Parameter bounds: enforce A>=0, w>0, and keep x0 within scan range
    dx_med = float(np.median(np.diff(np.sort(x)))) if len(x) > 1 else 1e-3
    w_lo = max(abs(dx_med) / 10.0, 1e-6)
    w_hi = max(x.max() - x.min(), w_lo * 10.0)

    A_hi = max(2.0 * yrng, A0 * 5.0)
    y0_lo = y_min - 0.5 * yrng
    y0_hi = y_max  # baseline shouldn't exceed the observed max for a positive peak model

    p0 = [A0, float(np.clip(x0_0, x.min(), x.max())), float(np.clip(w0, w_lo, w_hi)), float(np.clip(y0_0, y0_lo, y0_hi))]
    bounds = ([0.0, x.min(), w_lo, y0_lo], [A_hi, x.max(), w_hi, y0_hi])

    try:
        popt, pcov = curve_fit(
            sinc2, x, y,
            p0=p0,
            bounds=bounds,
            maxfev=40000
        )
    except Exception:
        popt, pcov = p0, np.full((4, 4), np.nan)

    return popt, pcov

dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

pulse_sequencer = Pulse_Sequencer()

exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.08,  N_Cycles=500)


exp_sequence.set_detunings(detuning_dict={"854 SP1": 0, "854 SP2": 0, "397a": 0, "397c": -20, "866": 0, "866 OP": 30, "850 RP": 0, "866 RP": 40, "729 Temp1": 0, "729 Temp2": 0})

exp_sequence.load_cooling(length=1000)
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

# --- Make probe power runtime-configurable (used by the Probe section) ---
probe_power = {"value": 0.01}  # mutable box so the lambda sees updates

exp_sequence.create_section(name="Probe", duration=25, dds_functions={
    "729 Temp1": lambda t: probe_power["value"],
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

def spectrum_filename_minus_power(power, directory="."):
    # e.g. spectrum_minus_p0p05_20260112_153012.csv
    tag = str(power).replace(".", "p")
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(directory, f"spectrum_minus_p{tag}_{ts}.csv")

# ---------- Modified: only minus scan, repeated for different probe powers ----------
plt.ion()

# Define detuning range (minus peak only)
center_minus = -23.35
span = 0.2
step = 0.005
detunings_minus = np.arange(center_minus - span, center_minus + span + 1e-9, step)

probe_powers = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
axes = np.ravel(axes)

def run_scan(detunings, ax, title, label):
    line, = ax.plot([], [], 'o', ms=3, label=label)
    ax.set_xlabel("Detuning (MHz)")
    ax.set_ylabel("P_dark")
    ax.set_title(title)
    ax.grid(True)
    vals = []
    for i, d in enumerate(tqdm(detunings, desc=title)):
        p = Dark_Probability(d)
        vals.append(p)
        line.set_data(detunings[:i+1], vals)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.01)
    return np.array(detunings, dtype=float), np.array(vals, dtype=float)

def save_spectrum(detunings, values, out_csv):
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["detuning_MHz", "P_dark"])
        for d, p in zip(detunings, values):
            writer.writerow([float(d), float(p)])
    print(f"Saved: {out_csv}")
    return out_csv

centers = []
for idx, (pwr, ax) in enumerate(zip(probe_powers, axes)):
    probe_power["value"] = float(pwr)

    title = f"Minus peak @ probe power = {pwr:g}"
    dets, vals = run_scan(detunings_minus, ax, title=title, label="data")

    out_csv = spectrum_filename_minus_power(pwr, directory=".")
    save_spectrum(dets, vals, out_csv)

    popt, _ = fit_sinc2(dets, vals, x0_guess=center_minus)
    xfit = np.linspace(dets.min(), dets.max(), 600)
    ax.plot(xfit, sinc2(xfit, *popt), '-', lw=2, label=f"sinc^2 fit (x0={popt[1]:.6f})")
    ax.axvline(popt[1], color='k', ls='--', alpha=0.6)
    ax.legend(loc="best")

    centers.append((pwr, popt[1]))
    print(f"Probe power {pwr:g}: fitted center = {popt[1]:.6f} MHz")

# If there are unused axes (shouldn't be), hide them
for ax in axes[len(probe_powers):]:
    ax.set_visible(False)

plt.show(block=False)
plt.ioff()
plt.show()