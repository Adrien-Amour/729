import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def find_latest_csv(prefix, directory="."):
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.csv$")
    latest_idx, latest_path = -1, None
    try:
        for fname in os.listdir(directory):
            m = pattern.match(fname)
            if m:
                idx = int(m.group(1))
                if idx > latest_idx:
                    latest_idx = idx
                    latest_path = os.path.join(directory, fname)
    except FileNotFoundError:
        pass
    return latest_path

def load_csv(path):
    dets, vals = [], []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            try:
                dets.append(float(row[0]))
                vals.append(float(row[1]))
            except Exception:
                continue
    return np.array(dets, dtype=float), np.array(vals, dtype=float)

def dual_lorentzian(x, A1, x01, g1, A2, x02, g2, y0):
    x = np.asarray(x, dtype=float)
    return y0 + A1*(g1**2)/((x-x01)**2+g1**2) + A2*(g2**2)/((x-x02)**2+g2**2)

def fit_dual(x, y, x01_guess=None, x02_guess=None, delta_guess=0.1):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y0 = float(np.min(y))
    A = max((np.max(y) - y0), 1e-3)
    width_guess = max((x.max() - x.min())/10.0, 1e-3)
    if x01_guess is None or x02_guess is None:
        # place two peaks ±delta_guess around the global maximum
        idx_max = int(np.argmax(y))
        x0 = x[idx_max]
        x01_guess = x0 - float(delta_guess)
        x02_guess = x0 + float(delta_guess)
    p0 = [A/2, float(x01_guess), width_guess, A/2, float(x02_guess), width_guess, y0]
    try:
        popt, _ = curve_fit(dual_lorentzian, x, y, p0=p0, maxfev=30000)
    except Exception:
        popt = p0
    return popt

# --- new: joint fit of RSB and BSB with shared baseline y0 ---
def fit_dual_shared_baseline(x_rsb, y_rsb, x_bsb, y_bsb, rsb_x01, rsb_x02, bsb_x01, bsb_x02, width_guess=None):
    x_rsb = np.asarray(x_rsb, dtype=float)
    y_rsb = np.asarray(y_rsb, dtype=float)
    x_bsb = np.asarray(x_bsb, dtype=float)
    y_bsb = np.asarray(y_bsb, dtype=float)

    if width_guess is None:
        width_guess = max((x_rsb.max() - x_rsb.min()) / 10.0, (x_bsb.max() - x_bsb.min()) / 10.0, 1e-3)

    # amplitudes from ranges; shared baseline initial as global min
    y0_init = float(min(y_rsb.min(), y_bsb.min()))
    A_rsb = max((y_rsb.max() - y0_init), 1e-3)
    A_bsb = max((y_bsb.max() - y0_init), 1e-3)

    # params: [A1r, x01r, g1r, A2r, x02r, g2r, A1b, x01b, g1b, A2b, x02b, g2b, y0_shared]
    p0 = [
        A_rsb/2, float(rsb_x01), width_guess, A_rsb/2, float(rsb_x02), width_guess,
        A_bsb/2, float(bsb_x01), width_guess, A_bsb/2, float(bsb_x02), width_guess,
        y0_init
    ]

    # build concatenated x with indices to split inside model
    n_rsb = x_rsb.size
    x_all = np.concatenate([x_rsb, x_bsb])

    def model_joint(x_all, A1r, x01r, g1r, A2r, x02r, g2r, A1b, x01b, g1b, A2b, x02b, g2b, y0_shared):
        xr = x_all[:n_rsb]
        xb = x_all[n_rsb:]
        yr = y0_shared + A1r*(g1r**2)/((xr - x01r)**2 + g1r**2) + A2r*(g2r**2)/((xr - x02r)**2 + g2r**2)
        yb = y0_shared + A1b*(g1b**2)/((xb - x01b)**2 + g1b**2) + A2b*(g2b**2)/((xb - x02b)**2 + g2b**2)
        return np.concatenate([yr, yb])

    y_all = np.concatenate([y_rsb, y_bsb])
    try:
        popt, pcov = curve_fit(model_joint, x_all, y_all, p0=p0, maxfev=50000)
    except Exception:
        popt = p0
        pcov = np.full((len(p0), len(p0)), np.nan)

    # unpack results
    params_rsb = popt[0:6]  # A1r,x01r,g1r,A2r,x02r,g2r
    params_bsb = popt[6:12] # A1b,x01b,g1b,A2b,x02b,g2b
    y0_shared = popt[12]
    return params_rsb, params_bsb, y0_shared, pcov

def main():
    rsb_csv = find_latest_csv("rsb_spectrum", ".")
    bsb_csv = find_latest_csv("bsb_spectrum", ".")
    if not rsb_csv and not bsb_csv:
        print("No rsb_spectrum_*.csv or bsb_spectrum_*.csv files found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    if rsb_csv:
        x_rsb, y_rsb = load_csv(rsb_csv)
        ax = axes[0]
        ax.plot(x_rsb, y_rsb, "o", ms=3, label="RSB data")
        ax.set_title(f"RSB: {os.path.basename(rsb_csv)}")
        ax.set_xlabel("Detuning (MHz)")
        ax.set_ylabel("P_dark")
        ax.grid(True)
    else:
        axes[0].set_visible(False)

    if bsb_csv:
        x_bsb, y_bsb = load_csv(bsb_csv)
        ax = axes[1]
        ax.plot(x_bsb, y_bsb, "o", ms=3, label="BSB data")
        ax.set_title(f"BSB: {os.path.basename(bsb_csv)}")
        ax.set_xlabel("Detuning (MHz)")
        ax.set_ylabel("P_dark")
        ax.grid(True)
    else:
        axes[1].set_visible(False)

    # Perform joint fit with shared baseline
    rsb_x01, rsb_x02 = -24.0, -24.42
    bsb_x01, bsb_x02 = -22.5, -22.14
    params_rsb, params_bsb, y0_shared, _ = fit_dual_shared_baseline(
        x_rsb, y_rsb, x_bsb, y_bsb,
        rsb_x01, rsb_x02,
        bsb_x01, bsb_x02
    )

    # Plot fitted curves using shared baseline
    xfit_rsb = np.linspace(x_rsb.min(), x_rsb.max(), 800)
    xfit_bsb = np.linspace(x_bsb.min(), x_bsb.max(), 800)
    axes[0].plot(
        xfit_rsb,
        dual_lorentzian(xfit_rsb, *params_rsb, y0_shared),
        "-", lw=2, label="Fit"
    )
    axes[1].plot(
        xfit_bsb,
        dual_lorentzian(xfit_bsb, *params_bsb, y0_shared),
        "-", lw=2, label="Fit"
    )
    # mark fitted centers
    rsb_c1, rsb_c2 = params_rsb[1], params_rsb[4]
    bsb_c1, bsb_c2 = params_bsb[1], params_bsb[4]
    axes[0].axvline(rsb_c1, color="r", ls="--", alpha=0.6, label=f"RSB1={rsb_c1:.5f}")
    axes[0].axvline(rsb_c2, color="m", ls="--", alpha=0.6, label=f"RSB2={rsb_c2:.5f}")
    axes[1].axvline(bsb_c1, color="r", ls="--", alpha=0.6, label=f"BSB1={bsb_c1:.5f}")
    axes[1].axvline(bsb_c2, color="m", ls="--", alpha=0.6, label=f"BSB2={bsb_c2:.5f}")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")

    # --- compute inner/outer secular frequencies with your convention and shared baseline ---
    # RSB: inner closer to carrier => more positive detuning (max), outer => min
    inner_rsb = max(rsb_c1, rsb_c2)
    outer_rsb = min(rsb_c1, rsb_c2)
    # BSB: inner closer to carrier => more negative detuning (min), outer => max
    inner_bsb = min(bsb_c1, bsb_c2)
    outer_bsb = max(bsb_c1, bsb_c2)

    secular_inner = (inner_bsb - inner_rsb) / 2.0
    secular_outer = (outer_bsb - outer_rsb) / 2.0

    print(f"Shared baseline y0: {y0_shared:.6f}")
    print(f"Inner secular frequency: {secular_inner:.6f} MHz (inner BSB - inner RSB)/2")
    print(f"Outer secular frequency: {secular_outer:.6f} MHz (outer BSB - outer RSB)/2")

    # Carrier estimates from pairs
    carrier_from_inner = 0.5 * (inner_bsb + inner_rsb)
    carrier_from_outer = 0.5 * (outer_bsb + outer_rsb)
    print(f"Carrier (from inner peaks): {carrier_from_inner:.6f} MHz")
    print(f"Carrier (from outer peaks): {carrier_from_outer:.6f} MHz")

    fig.suptitle(
        f"Shared Background={y0_shared:.4f} | Inner secular={secular_inner:.6f} MHz | Outer secular={secular_outer:.6f} MHz\n",
        color="blue", fontsize=10
    )

    out_png = "latest_spectra.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved figure: {out_png}")
    plt.show()

if __name__ == "__main__":
    main()
