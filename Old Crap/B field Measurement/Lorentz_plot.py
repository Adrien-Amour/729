import os
import re
import csv
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -------- file finding --------
SPECTRUM_PATTERN = re.compile(r"^spectrum_(\d+)\.csv$", re.IGNORECASE)

def find_most_recent_files(directory: str, pattern: re.Pattern, count: int = 2) -> List[str]:
    candidates = []
    try:
        for fname in os.listdir(directory):
            m = pattern.match(fname)
            if m:
                idx = int(m.group(1))
                candidates.append((idx, os.path.join(directory, fname)))
    except FileNotFoundError:
        pass
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [path for _, path in candidates[:count]]

def load_spectrum_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    detuning = []
    p_dark = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        det_idx, p_idx = 0, 1
        if header:
            try:
                det_idx = header.index("detuning_MHz")
            except ValueError:
                det_idx = 0
            try:
                p_idx = header.index("P_dark")
            except ValueError:
                p_idx = 1
        for row in reader:
            if not row or len(row) <= max(det_idx, p_idx):
                continue
            try:
                detuning.append(float(row[det_idx]))
                p_dark.append(float(row[p_idx]))
            except ValueError:
                continue
    d = np.array(detuning, dtype=float)
    y = np.array(p_dark, dtype=float)
    order = np.argsort(d)
    return d[order], y[order]

# -------- physics: sensitivity from your Single_BField_Measurement --------
def zeemanFS(J, S, L, B_T):
    muB = 9.274078e-24   # J/T
    hbar = 1.054572669125e-34  # J·s
    gJ = 1 + (J*(J+1) + S*(S+1) - L*(L+1)) / (2*J*(J+1))
    w = gJ * muB * B_T / hbar      # rad/s
    return w/1E6  # MHz

# Precompute MHz/Gauss sensitivity for |m| change of 2 between S1/2 and D5/2 branches (your formula)
# B in Tesla; 1 G = 1e-4 T
B_ref_T = 1e-4
w1 = zeemanFS(1/2, 1/2, 0, B_ref_T)
w5 = zeemanFS(5/2, 1/2, 2, B_ref_T)
Delta_m2 = 2.5 * w5 - 0.5 * w1
SENSITIVITY_MHz_per_G = 2 * Delta_m2 / (2 * np.pi)  # MHz/G

# -------- fitting --------
def lorentzian(x, A, x0, gamma, y0):
    x = np.asarray(x, dtype=float)
    return y0 + A * (gamma**2) / ((x - x0)**2 + gamma**2)

def fit_lorentzian(x, y, x0_guess=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x0_guess is None:
        x0_guess = float(x[np.argmax(y)])
    A0 = max((np.max(y) - np.min(y)), 1e-6)
    gamma0 = max((x.max() - x.min()) / 10.0, 1e-6)
    y00 = float(np.min(y))
    try:
        popt, pcov = curve_fit(lorentzian, x, y, p0=[A0, x0_guess, gamma0, y00], maxfev=20000)
    except Exception:
        popt, pcov = [A0, x0_guess, gamma0, y00], np.full((4, 4), np.nan)
    return popt, pcov

# --- new: sinc^2 lineshape and fitting ---
def sinc2(x, A, x0, w, y0):
    """
    y = y0 + A * sinc((x - x0)/w)^2
    np.sinc uses sin(pi z)/(pi z), which is convenient for Fourier-limited pulses.
    """
    x = np.asarray(x, dtype=float)
    z = (x - x0) / max(w, 1e-12)
    return y0 + A * np.sinc(z) ** 2

def fit_sinc2(x, y, x0_guess=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x0_guess is None:
        x0_guess = float(x[np.argmax(y)])
    A0 = max((np.max(y) - np.min(y)), 1e-6)
    # Initial width guess: similar scale as Lorentzian gamma guess
    w0 = max((x.max() - x.min()) / 10.0, 1e-6)
    y00 = float(np.min(y))
    try:
        popt, pcov = curve_fit(sinc2, x, y, p0=[A0, x0_guess, w0, y00], maxfev=20000)
    except Exception:
        popt, pcov = [A0, x0_guess, w0, y00], np.full((4, 4), np.nan)
    return popt, pcov

def main():
    directory = os.getcwd()
    if len(sys.argv) > 1:
        directory = sys.argv[1]

    spectra = find_most_recent_files(directory, SPECTRUM_PATTERN, count=2)
    if len(spectra) < 2:
        print("Need at least two spectrum_###.csv files.")
        if spectra:
            print(f"Found only one: {spectra[0]}")
        sys.exit(1)

    (d1, p1) = load_spectrum_csv(spectra[0])
    (d2, p2) = load_spectrum_csv(spectra[1])

    # Fit both
    popt1, _ = fit_lorentzian(d1, p1)
    popt2, _ = fit_lorentzian(d2, p2)
    # Also fit sinc^2
    spopt1, _ = fit_sinc2(d1, p1, x0_guess=popt1[1])
    spopt2, _ = fit_sinc2(d2, p2, x0_guess=popt2[1])

    # --- new: compute linewidths and probe times ---
    # Lorentzian gamma is HWHM; FWHM = 2*gamma
    fwhm1 = 2.0 * float(popt1[2])
    fwhm2 = 2.0 * float(popt2[2])
    # For sinc^2: first zero at (x-x0)/w = 1 -> w ≈ 1/T (with x in MHz)
    # T (seconds) = 1 / (w * 1e6); report in microseconds
    T1_us = 1.0 / (abs(float(spopt1[2])) * 1e6) * 1e6 if spopt1[2] != 0 else np.nan
    T2_us = 1.0 / (abs(float(spopt2[2])) * 1e6) * 1e6 if spopt2[2] != 0 else np.nan

    center1 = popt1[1]
    center2 = popt2[1]
    splitting_MHz = abs(center2 - center1)
    B_field_G = splitting_MHz / SENSITIVITY_MHz_per_G

    # --- new: first figure: Lorentzian only ---
    figL, (ax1L, ax2L) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    xfit1 = np.linspace(d1.min(), d1.max(), 600)
    ax1L.plot(d1, p1, 'o', ms=3, label=os.path.basename(spectra[0]))
    ax1L.plot(xfit1, lorentzian(xfit1, *popt1), '-', lw=2, label='Lorentz fit')
    ax1L.axvline(center1, color='k', ls=':', alpha=0.6, label=f"Center = {center1:.6f} MHz")
    ax1L.set_xlabel("Detuning (MHz)")
    ax1L.set_ylabel("P_dark")
    ax1L.set_title("Most recent spectrum (Lorentz)")
    ax1L.grid(True, alpha=0.3)
    ax1L.legend(loc="best")

    xfit2 = np.linspace(d2.min(), d2.max(), 600)
    ax2L.plot(d2, p2, 'o', ms=3, label=os.path.basename(spectra[1]))
    ax2L.plot(xfit2, lorentzian(xfit2, *popt2), '-', lw=2, label='Lorentz fit')
    ax2L.axvline(center2, color='k', ls=':', alpha=0.6, label=f"Center = {center2:.6f} MHz")
    ax2L.set_xlabel("Detuning (MHz)")
    ax2L.set_ylabel("P_dark")
    ax2L.set_title("Second most recent spectrum (Lorentz)")
    ax2L.grid(True, alpha=0.3)
    ax2L.legend(loc="best")

    figL.suptitle(
        f"Splitting = {splitting_MHz:.6f} MHz | B = {B_field_G:.6f} G\n"
        f"Lorentz centers: {center1:.6f}, {center2:.6f} | FWHM: {fwhm1:.6f} MHz, {fwhm2:.6f} MHz",
        color='red', fontsize=12
    )
    plt.show()

    # --- new: second figure: sinc^2 only ---
    figS, (ax1S, ax2S) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax1S.plot(d1, p1, 'o', ms=3, label=os.path.basename(spectra[0]))
    ax1S.plot(xfit1, sinc2(xfit1, *spopt1), '--', lw=2, label='sinc^2 fit')
    ax1S.axvline(spopt1[1], color='gray', ls='--', alpha=0.6, label=f"Center = {spopt1[1]:.6f} MHz")
    ax1S.set_xlabel("Detuning (MHz)")
    ax1S.set_ylabel("P_dark")
    ax1S.set_title("Dark State Population Against Prob Detuning")
    ax1S.grid(True, alpha=0.3)
    # ax1S.legend(loc="best")

    ax2S.plot(d2, p2, 'o', ms=3, label=os.path.basename(spectra[1]))
    ax2S.plot(xfit2, sinc2(xfit2, *spopt2), '--', lw=2, label='sinc^2 fit')
    ax2S.axvline(spopt2[1], color='gray', ls='--', alpha=0.6, label=f"Center = {spopt2[1]:.6f} MHz")
    ax2S.set_xlabel("Detuning (MHz)")
    ax2S.set_ylabel("P_dark")
    ax2S.set_title("Dark State Population Against Prob Detuning")
    ax2S.grid(True, alpha=0.3)
    # ax2S.legend(loc="best")

    figS.suptitle(
        f"sinc^2 centers: {spopt1[1]:.6f}, {spopt2[1]:.6f} | Probe times: {T1_us:.3f} us, {T2_us:.3f} us",
        color='red', fontsize=12
    )
    plt.show()

    print(f"Centers (Lorentz): {center1:.6f} MHz, {center2:.6f} MHz")
    print(f"Centers (sinc^2): {spopt1[1]:.6f} MHz, {spopt2[1]:.6f} MHz")
    print(f"Lorentzian FWHM: {fwhm1:.6f} MHz (recent), {fwhm2:.6f} MHz (second)")
    print(f"Estimated probe time from sinc^2: {T1_us:.3f} us (recent), {T2_us:.3f} us (second)")
    print(f"Splitting: {splitting_MHz:.6f} MHz")
    print(f"B field: {B_field_G:.6f} Gauss")

if __name__ == "__main__":
    main()