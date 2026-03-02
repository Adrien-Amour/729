#!/usr/bin/env python3
"""
Fit the most recent spectrum_*.csv to a Lorentzian with offset:
    y(x) = y0 + A * gamma^2 / ((x - x0)^2 + gamma^2)

Usage (Windows, from the folder with CSVs):
    python spectroscopy_fit.py
Optional:
    python spectroscopy_fit.py --dir "C:\path\to\data" --pattern "spectrum_*.csv"
"""
import argparse
import csv
import os
import re
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def lorentz(x, y0, A, x0, gamma):
    return y0 + A * (gamma**2) / ((x - x0)**2 + gamma**2)


def find_latest_csv(search_dir: str, pattern: str) -> str:
    # Prefer numeric suffix (e.g., spectrum_N.csv). Fallback to latest mtime.
    rx = None
    numeric_candidates = []
    try:
        m = re.match(r"^(.*)_\*\.csv$", pattern)
        if m:
            base = re.escape(m.group(1))
            rx = re.compile(rf"^{base}_(\d+)\.csv$")
        else:
            rx = re.compile(r".*?(\d+)\.csv$")
    except re.error:
        rx = None

    for fname in os.listdir(search_dir):
        if not fname.lower().endswith(".csv"):
            continue
        if not _matches_glob_like(fname, pattern):
            continue
        if rx:
            mm = rx.match(fname)
            if mm:
                try:
                    n = int(mm.group(1))
                    numeric_candidates.append((n, fname))
                except ValueError:
                    pass

    if numeric_candidates:
        fname = max(numeric_candidates, key=lambda t: t[0])[1]
        return os.path.join(search_dir, fname)

    matches = [
        os.path.join(search_dir, f)
        for f in os.listdir(search_dir)
        if f.lower().endswith(".csv") and _matches_glob_like(f, pattern)
    ]
    if not matches:
        raise FileNotFoundError(f"No CSVs matching '{pattern}' in {search_dir}")
    return max(matches, key=os.path.getmtime)


def _matches_glob_like(name: str, pattern: str) -> bool:
    # Tiny '*' matcher
    if pattern == "*":
        return True
    parts = pattern.split("*")
    if len(parts) == 1:
        return name == pattern
    i = 0
    if parts[0]:
        if not name.startswith(parts[0]):
            return False
        i = len(parts[0])
    for mid in parts[1:-1]:
        if not mid:
            continue
        j = name.find(mid, i)
        if j < 0:
            return False
        i = j + len(mid)
    suf = parts[-1]
    if suf:
        return name[i:].endswith(suf)
    return True


def read_xy_from_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    with open(path, "r", newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            if str(row[0]).strip().startswith("#"):
                continue
            try:
                x = float(row[0])
                y = float(row[1])
            except (ValueError, IndexError):
                continue
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError(f"No numeric (x,y) rows found in {path}")
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def initial_guess(x: np.ndarray, y: np.ndarray):
    y0 = float(np.median(y))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    # Auto-detect peak or dip
    if (y_max - y0) >= (y0 - y_min):
        A0 = (y_max - y_min) or 1.0
        x0 = float(x[np.argmax(y)])
    else:
        A0 = -((y_max - y_min) or 1.0)
        x0 = float(x[np.argmin(y)])
    xr = float(np.max(x) - np.min(x)) or 1.0
    gamma0 = xr / 10.0
    return y0, A0, x0, gamma0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Directory to search for CSVs")
    ap.add_argument("--pattern", default="spectrum_*.csv",
                    help="Glob-like pattern for files (use * as wildcard)")
    args = ap.parse_args()

    csv_path = find_latest_csv(args.dir, args.pattern)
    print(f"Using: {csv_path}")

    x, y = read_xy_from_csv(csv_path)

    p0 = initial_guess(x, y)
    xr = float(np.max(x) - np.min(x)) or 1.0
    bounds = (
        [-np.inf, -np.inf, np.min(x) - xr, 1e-9],   # y0, A, x0, gamma
        [ np.inf,  np.inf, np.max(x) + xr,  np.inf],
    )

    try:
        popt, pcov = curve_fit(lorentz, x, y, p0=p0, bounds=bounds, maxfev=20000)
        perr = np.sqrt(np.diag(pcov)) if (pcov is not None and pcov.size) else [np.nan]*len(popt)
    except Exception as e:
        print(f"Fit failed: {e}")
        return

    y0, A, x0, gamma = popt
    sy0, sA, sx0, sgamma = perr
    fwhm = 2.0 * gamma
    sfwhm = 2.0 * sgamma

    print("Fit results:")
    print(f"  y0    = {y0:.6g} ± {sy0:.2g}")
    print(f"  A     = {A:.6g} ± {sA:.2g}")
    print(f"  x0    = {x0:.6g} ± {sx0:.2g}")
    print(f"  gamma = {gamma:.6g} ± {sgamma:.2g}  (FWHM = {fwhm:.6g} ± {sfwhm:.2g})")

    # Plot
    xf = np.linspace(np.min(x), np.max(x), 1000)
    yf = lorentz(xf, *popt)

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, "o", label="Data")
    plt.plot(xf, yf, "-", label="Lorentz fit")
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("P_dark")
    plt.title(f"Lorentzian fit: {os.path.basename(csv_path)}")
    plt.grid(True)
    plt.legend()

    base, _ = os.path.splitext(csv_path)
    out_png = base + "_lorentz_fit.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()

    # Save parameters
    out_txt = base + "_lorentz_fit.txt"
    with open(out_txt, "w", newline="") as f:
        f.write("# Lorentzian fit: y = y0 + A * gamma^2 / ((x - x0)^2 + gamma^2)\n")
        f.write(f"file  = {os.path.basename(csv_path)}\n")
        f.write(f"y0    = {y0:.12g}  +/- {sy0:.3g}\n")
        f.write(f"A     = {A:.12g}  +/- {sA:.3g}\n")
        f.write(f"x0    = {x0:.12g}  +/- {sx0:.3g}\n")
        f.write(f"gamma = {gamma:.12g}  +/- {sgamma:.3g}\n")
        f.write(f"FWHM  = {fwhm:.12g}  +/- {sfwhm:.3g}\n")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()