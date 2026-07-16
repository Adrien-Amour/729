import numpy as np
import pandas as pd
import glob
import re
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ---------------------------
# 1. Find latest file
# ---------------------------
files = glob.glob("ramsey_detuning_*.csv")

def extract_number(f):
    m = re.search(r'ramsey_detuning_(\d+)\.csv', f)
    return int(m.group(1)) if m else -1

latest_file = max(files, key=extract_number)
print(f"Loading: {latest_file}")

# ---------------------------
# 2. Load data
# ---------------------------
df = pd.read_csv(latest_file)

detuning = df["detuning_MHz"].values
P = df["P_dark"].values
stderr = df["P_dark_stderr"].values

# Convert MHz → Hz for calculation
delta = detuning * 1e6

# ---------------------------
# 3. Physical Ramsey model
# ---------------------------
def ramsey_physical(delta, Omega0, T, phi, Pmax, delta0, t_pi2):
    """
    delta      : detuning in Hz
    Omega0     : Rabi frequency in rad/s
    T          : free precession time in s
    phi        : phase offset in rad
    Pmax       : maximum excitation probability (0 < Pmax <= 1)
    delta0     : resonance detuning offset in Hz
    t_pi2      : pulse duration of π/2 in s
    """
    d = delta - delta0
    Omega_eff = np.sqrt(Omega0**2 + (2*np.pi*d)**2)

    # Detuning-reduced excitation
    pulse_factor = (Omega0 / Omega_eff)**2 * np.sin(Omega_eff * t_pi2 / 2)**2

    return Pmax * pulse_factor * (1 + np.cos(2*np.pi*d*T + phi))

# ---------------------------
# 4. Initial guesses
# ---------------------------
Omega0_guess = np.pi / (2 * 7.2e-6)  # rad/s, from t_pi2 ~ 7.2us
T_guess = 40e-6                     # s
phi_guess = 0                        # rad
Pmax_guess = 0.95                     # max dark state probability
delta0_guess = np.mean(delta)         # Hz
t_pi2_guess = 7.2e-6                  # s

p0 = [Omega0_guess, T_guess, phi_guess, Pmax_guess, delta0_guess, t_pi2_guess]

# ---------------------------
# 5. Bounds
# ---------------------------
bounds = (
    [
        0.5 * Omega0_guess,  # Omega0
        20e-6,               # T
        -np.pi,              # phi
        0.8,                 # Pmax
        np.min(delta),       # delta0
        5e-6                 # t_pi2
    ],
    [
        2.0 * Omega0_guess,
        80e-6,
        np.pi,
        1.0,
        np.max(delta),
        15e-6
    ]
)

# ---------------------------
# 6. Fit
# ---------------------------
popt, pcov = curve_fit(
    ramsey_physical,
    delta,
    P,
    sigma=stderr,
    absolute_sigma=True,
    p0=p0,
    bounds=bounds,
    maxfev=50000
)

Omega0, T, phi, Pmax, delta0, t_pi2 = popt

print("\nFit results:")
print(f"Omega0 = {Omega0/(2*np.pi):.2f} Hz")
print(f"T      = {T*1e6:.2f} us")
print(f"phi    = {phi:.3f} rad")
print(f"Pmax   = {Pmax:.3f}")
print(f"delta0 = {delta0/1e6:.6f} MHz")
print(f"t_pi2  = {t_pi2*1e6:.2f} us")

# ---------------------------
# 7. Plot
# ---------------------------
xfit = np.linspace(delta.min(), delta.max(), 2000)
yfit = ramsey_physical(xfit, *popt)

plt.errorbar(detuning, P, yerr=stderr, fmt='o', label='Data')
plt.plot(xfit/1e6, yfit, label='Fit')
plt.xlabel("Detuning (MHz)")
plt.ylabel("P_dark")
plt.legend()
plt.title("Ramsey Fit (Ω_eff + flexible t_pi2)")
plt.show()