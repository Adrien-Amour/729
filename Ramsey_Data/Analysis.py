import os
import re
import glob
import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data_dir = r"c:\Users\probe\OneDrive - University of Sussex\Desktop\729\Ramsey_Data"
pattern = re.compile(r"ramsey_tprecess_(\d+)us_(\d{8}_\d{6})\.csv")

# Only analyze the most recent scan for these t_precess values (in microseconds).
# Edit this list to control which times are included.
TARGET_T_PRECESS_US = [10,40,80,120,160]

# =============================
# Ramsey fit functions
# =============================

def fit_func_full(phase, A, phi0, C):
    # For t=0 (C fixed = 1)
    return 0.5 * A * (1 + C*np.cos(2*np.deg2rad(phase) + phi0))


# =============================
# Find newest file for each t_precess
# =============================

files_by_t = {}

for fname in glob.glob(os.path.join(data_dir, "plus_ramsey_tprecess_*us_*.csv")):
    m = pattern.search(os.path.basename(fname))
    if m:
        t_precess = int(m.group(1))
        timestamp = m.group(2)
        files_by_t.setdefault(t_precess, []).append((timestamp, fname))

most_recent_files = []
target_t_set = set(TARGET_T_PRECESS_US)
missing_targets = sorted(target_t_set.difference(files_by_t.keys()))
if missing_targets:
    print(f"Warning: no files found for t_precess (us): {missing_targets}")

for t, files in files_by_t.items():
    if t not in target_t_set:
        continue
    files.sort()
    most_recent_files.append((t, files[-1][1]))

most_recent_files.sort()

# =============================
# Fit Ramsey fringes
# =============================
all_p_dark = []

for t, fname in most_recent_files:
    if t == 0:
        continue  # Skip t=0 files

    with open(fname, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            all_p_dark.append(float(row[1]))

A_est = 2 * np.mean(all_p_dark)
print(f"Estimated A from average: {A_est:.4f}")


t_list = []
C_list = []
C_err_list = []

plt.figure()



A0 = None
t_list = []
C_list = []
C_err_list = []

plt.figure()


for t, fname in most_recent_files:
    if t == 0:
        continue  # Skip t=0 files

    phases, p_dark, stderr = [], [], []

    with open(fname, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            phases.append(float(row[0]))
            p_dark.append(float(row[1]))
            stderr.append(float(row[2]))
    avg = np.mean(p_dark)
    print(f"t={t}us: average = {avg:.4f}")
    phases = np.array(phases)
    p_dark = np.array(p_dark)
    stderr = np.array(stderr)

    # 2. Fit for A and phi0 (no normalisation)
    def fit_func(phase, A, phi0, C):
        return fit_func_full(phase, A, phi0, C)

    p0 = [np.max(p_dark), 0.0, 1.0]  # Initial guess for A, phi0, and C

    popt, pcov = curve_fit(
        fit_func,
        phases,
        p_dark,
        sigma=stderr,
        p0=p0,
        absolute_sigma=True,
    )

    A, phi0, C = popt
    A_err = np.sqrt(np.diag(pcov))[0]

    fit_y = fit_func(phases, A, phi0, C)

    print(f"t={t}us: A={A:.4f}, C={C:.4f}")

    t_list.append(t)
    C_list.append(abs(C))          # Now C_list is the fitted C
    C_err_list.append(np.sqrt(np.diag(pcov))[2])  # Error on C

    label = f"t={t}us"

    eb = plt.errorbar(phases, p_dark, yerr=stderr, fmt='o', capsize=4, label=label)
    color = eb.lines[0].get_color()

    plt.plot(phases, fit_y, '--', color=color)
    plt.xlabel(r"$\mathrm{Phase}$ (deg)")
    plt.ylabel(r"Amplitude $A$")
    plt.title("Ramsey Scans and Fits")
    plt.legend()
    plt.tight_layout()
    # plt.show()

plt.xlabel(r"$\mathrm{Phase}$ (deg)")
plt.ylabel(r"Contrast $C/C_0$")
plt.title("Ramsey Scans and Fits")
plt.legend()
plt.tight_layout()
plt.show()

# =============================
# Contrast Decay Analysis
# =============================

t_vals = np.array(t_list)
C_vals = np.array(C_list)
C_errs = np.array(C_err_list)


# =============================
# Combined Gaussian + Exponential model
# C(t) = exp(-t/T2e - (t/T2g)^2)
# =============================

def combined_decay(C_0, t, T2e, T2g):
    return C_0 * np.exp(-t / T2e - (t / T2g)**2)


# ...existing code...

from scipy.optimize import curve_fit

# Fit C_vals vs t_vals to the combined_decay model
def combined_decay_fit_func(t, C_0, T2e, T2g):
    return combined_decay(C_0, t, T2e, T2g)

# Initial guesses: C_0=1, T2e=100, T2g=100 (adjust as needed)
p0 = [1.0, 100.0, 100.0]

popt, pcov = curve_fit(
    combined_decay_fit_func,
    t_vals,
    C_vals,
    sigma=C_errs,
    p0=p0,
    absolute_sigma=True,
    maxfev=10000
)

C_0_fit, T2e_fit, T2g_fit = popt
C_0_err, T2e_err, T2g_err = np.sqrt(np.diag(pcov))
# ...after the combined_decay fit and popt, pcov extraction...

T2e, T2g = popt[1], popt[2]
T2e_err, T2g_err = np.sqrt(np.diag(pcov))[1], np.sqrt(np.diag(pcov))[2]

print("\n=== Combined Ramsey Decay Fit ===")
print(f"T2_exp  = {T2e:.2f} ± {T2e_err:.2f} us")
print(f"T2_gaus = {T2g:.2f} ± {T2g_err:.2f} us")
linewidth = 2 / (T2e * 1E-6) # in Hz
print(f"Estimated linewidth from T2_exp: {linewidth*1E-3/(2*np.pi):.2f} (2pi) kHz")
sigma_delta = np.sqrt(2)/ (T2g * 1E-6) # in Hz
sigma_f = sigma_delta / (2*np.pi) # in Hz
print(f"Estimated sigma of Gaussian frequency distribution: {sigma_f*1E-3:.2f} (2pi) kHz")

def zeemanFS(J, S, L, B):
    muB = 9.274078e-24   # Bohr magneton in J/T
    hbar = 1.054572669125e-34  # Planck constant / 2pi in J·s
    gJ = 1 + (J*(J+1) + S*(S+1) - L*(L+1)) / (2*J*(J+1))
    w = gJ * muB * B / hbar      # angular frequency in rad/s
    return w/1E6 #convert to MHz

B = 1e-4  # Tesla to gauss
probe_amplitude = 0.1
# w1 for J=1/2, S=1/2, L=0
Delta_S = zeemanFS(1/2, 1/2, 0, B) / (2*np.pi)
# w5 for J=5/2, S=1/2, L=2
Delta_D = zeemanFS(5/2, 1/2, 2, B) / (2 * np.pi)
mS = -1/2
mD = +3/2
# mD=-5/2

sensitivity = np.abs(-mS*Delta_S + mD*Delta_D)
print(f"Frequency sensitivity to magnetic field fluctuations: {sensitivity:.2f} kHz/mG")
sigma_B =sigma_f*1E-3/sensitivity #convert to kHz and divide by sensitivity in kHz/mG to get mG

print(f"Estimated sigma of magnetic field fluctuations: {sigma_B:.2f} mG") 
# =============================
# Plot decay
# =============================

t_fit = np.linspace(0, np.max(t_vals), 500)

plt.figure()
plt.errorbar(t_vals, C_vals, yerr=C_errs, fmt='o', capsize=4)
plt.plot(
    t_fit,
    combined_decay(popt[0], t_fit, T2e, T2g),
    '--',
    label=r"Combined Fit: $e^{-t/T_{2e} - (t/T_{2g})^2}$"
)
plt.plot(
    t_fit,
    popt[0]*np.exp(-t_fit / T2e),
    ':',
    label=r"Dephasing: $e^{-t/T_{2e}}$"
)
plt.plot(
    t_fit,
    popt[0]*np.exp(-(t_fit / T2g)**2),
    ':',
    label=r"Gaussian $\Delta$ Noise: $e^{-(t/T_{2g})^2}$"
)

plt.xlabel(r"$t_\mathrm{precess}$ ($\mu$s)")
plt.ylabel(r"Contrast $C/C_0$")
plt.title("Ramsey Contrast Decay")
plt.legend()
plt.tight_layout()
plt.show()


# ----- Compute confidence band -----

def model(t, C0, T2e, T2g):
    return C0 * np.exp(-t/T2e - (t/T2g)**2)

C0, T2e, T2g = popt

# Evaluate model
y_fit = model(t_fit, C0, T2e, T2g)

# Compute Jacobian
exp_part = np.exp(-t_fit/T2e - (t_fit/T2g)**2)

dC0 = exp_part
dT2e = C0 * exp_part * (t_fit / T2e**2)
dT2g = C0 * exp_part * (2 * t_fit**2 / T2g**3)

J = np.vstack((dC0, dT2e, dT2g)).T

# Variance of fit
y_var = np.sum(J @ pcov * J, axis=1)
y_sigma = np.sqrt(y_var)

plt.figure()
plt.errorbar(t_vals, C_vals, yerr=C_errs, fmt='o', capsize=4, label="Data")

plt.plot(t_fit, y_fit, '--', label="Combined Fit")

# Add shaded 1-sigma region
plt.fill_between(
    t_fit,
    y_fit - y_sigma,
    y_fit + y_sigma,
    alpha=0.3,
    label="1σ Confidence"
)

plt.xlabel(r"$t_\mathrm{precess}$ ($\mu$s)")
plt.ylabel(r"Contrast $C/C_0$")
plt.title("Ramsey Contrast Decay")
plt.legend()
plt.tight_layout()
plt.show()

sensitivity_p = np.abs(-1/2*Delta_S + 3/2*Delta_D)
sensitivity_m = np.abs(-1/2*Delta_S - 5/2*Delta_D)
superposition_sensitivity = sensitivity_p + sensitivity_m
print(f"Superposition sensitivity: {superposition_sensitivity:.2f} (2pi) MHz/G or kHz/mG")
sigma_f_superposition_kHz = superposition_sensitivity * sigma_B
print(f"Estimated sigma of superposition frequency fluctuations: {sigma_f_superposition_kHz:.2f} (2pi) kHz")
sigma_delta_superposition_khz = sigma_f_superposition_kHz * 2 * np.pi
T2_superposition_us = np.sqrt(2) / (sigma_delta_superposition_khz * 1E-3)  # convert kHz to MHz
print(f"Estimated T2 for superposition: {T2_superposition_us:.2f} us")
interpolated_superposition_decay = model(t_fit, 1, np.inf, T2_superposition_us) 
fidelity = 0.5 * (1 + interpolated_superposition_decay)
plt.plot(
    t_fit,
    interpolated_superposition_decay,
    ':',
    label=r"Superposition Decay: $e^{-(t/T_{2g})^2}$"
)
plt.plot(
    t_fit,
    fidelity,
    ':',
    label=r"Superposition Fidelity: $0.5(1 + e^{-(t/T_{2g})^2})$"
)
plt.show()