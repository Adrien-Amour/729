import sys
import json
import os
import csv
import pprint

# Force pyqtgraph to use the same Qt binding as adriq.Custom_PyQT (PyQt5)
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from adriq.experiment import _load_presets_dict, load_dds_dict, Pulse_Sequencer, Experiment_Builder, Experiment_Runner, upsert_autogen_function
from datetime import datetime
# Prefer Custom_PyQT spinboxes; fall back to QtWidgets if unavailable.
try:
    from adriq.Custom_PyQT import CursorIntSpinBox, CursorDoubleSpinBox
except Exception:
    CursorIntSpinBox = QtWidgets.QSpinBox
    CursorDoubleSpinBox = QtWidgets.QDoubleSpinBox

# --- Qt binding compatibility (PyQt* vs PySide*) ---
Signal = getattr(QtCore, "pyqtSignal", None) or getattr(QtCore, "Signal", None)
Slot = getattr(QtCore, "pyqtSlot", None) or getattr(QtCore, "Slot", None)
if Signal is None or Slot is None:
    raise RuntimeError("Could not find Qt Signal/Slot (unsupported Qt binding via pyqtgraph.Qt).")

# ----------------------------
# User-editable constants (top-of-file “advanced options”)
# ----------------------------
DDS_CONFIG_PATH = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg"
MEAS_PRESET_START_DIR = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\ps_presets"
PRESETS_PY_PATH = os.path.join(MEAS_PRESET_START_DIR, "presets.py")
DEFAULT_MEASUREMENT_JSON_PATH = os.path.join(MEAS_PRESET_START_DIR, "measurement.json")  # NEW

# 397c is cooling detuning: must be consistent across dark/bright
COOLING_DETUNING_397C = -14.0
COOLING_DETUNING_397B = -14.0
STATE_PREP_LENGTH_US = 100
# Common trap/cycle settings
TRAP_AMP_397C = 0.45
N_CYCLES = 500  # was 1e3 (float); keep same value but as int for UI + builder

# Shelving section defaults (edit here if needed)
DARK_SHELVING = {"397b": 1.0, "850 RP": 1.0} # Pump to D5/2
BRIGHT_SHELVING = {"866 RP": 0.2, "854 SP1": 0.3} #Pump to S1/2

# State readout threshold used by Ion_State_Discrimination inside Experiment_Runner
STATE_READOUT_THRESHOLD = 4

# Temporal histogram settings (uses Experiment_Runner.get_time_diffs, units: µs)
TIME_DIFFS_MODE = "signal-f"
TIME_HIST_LOWER_US = None
TIME_HIST_UPPER_US = None  # was 80.0; None == full trace
TIME_HIST_N_BINS = 400     # fixed number of points/bins in the histogram

# Poisson-fit plotting cap (avoid silly-large arrays if something goes wrong)
POISSON_KMAX_CAP = 500

# ----------------------------
# Analysis helpers
# ----------------------------
def _hist_dict_to_xy(hist: dict[int, int]):
    """Return sorted x,y arrays from a {bin: count} dict."""
    if not hist:
        return np.array([], dtype=int), np.array([], dtype=float)
    xs = np.array(sorted(int(k) for k in hist.keys()), dtype=int)
    ys = np.array([float(hist[int(k)]) for k in xs], dtype=float)
    return xs, ys


def compute_fidelity_from_hists(dark_hist: dict, bright_hist: dict, thr: int) -> float:
    """Fidelity = 1 - 0.5*(P(dark mis-ID as bright) + P(bright mis-ID as dark))."""
    if not dark_hist or not bright_hist:
        return float("nan")
    d_tot = sum(dark_hist.values()) or 1
    b_tot = sum(bright_hist.values()) or 1
    p_dark_to_bright = sum(v for k, v in dark_hist.items() if int(k) >= int(thr)) / d_tot
    p_bright_to_dark = sum(v for k, v in bright_hist.items() if int(k) < int(thr)) / b_tot
    return 1.0 - 0.5 * (p_dark_to_bright + p_bright_to_dark)


def poisson_fit_from_hist(hist: dict[int, int], *, ks: np.ndarray | None = None):
    """
    Return (ks, expected_counts, nbar) where expected_counts = total * PoissonPMF(nbar).
    If ks is provided, evaluate the fit over that shared ks range.
    Uses log-factorial for stability.
    """
    if not hist:
        if ks is None:
            return np.array([], dtype=int), np.array([], dtype=float), float("nan")
        ks = np.asarray(ks, dtype=int)
        return ks, np.zeros_like(ks, dtype=float), float("nan")

    from math import lgamma, log

    x, y = _hist_dict_to_xy(hist)
    total = float(y.sum())
    if total <= 0:
        if ks is None:
            return np.array([], dtype=int), np.array([], dtype=float), float("nan")
        ks = np.asarray(ks, dtype=int)
        return ks, np.zeros_like(ks, dtype=float), float("nan")

    nbar = float((x * y).sum() / total)

    if ks is None:
        kmax = int(x.max()) if x.size else 0
        kmax = min(kmax, int(POISSON_KMAX_CAP))
        ks = np.arange(0, kmax + 1, dtype=int)
    else:
        ks = np.asarray(ks, dtype=int)

    if nbar <= 0:
        expected = np.zeros_like(ks, dtype=float)
        if np.any(ks == 0):
            expected[ks == 0] = total
        return ks, expected, 0.0

    log_nbar = log(nbar)
    logpmf = np.array([-nbar + (int(k) * log_nbar) - lgamma(int(k) + 1) for k in ks], dtype=float)
    expected = total * np.exp(logpmf)
    return ks, expected, nbar


def time_diffs_to_hist_xy(
    time_diffs_by_channel: dict,
    *,
    lower_us=None,
    upper_us=None,
    n_bins: int | None = None,
    bin_width_us: float | None = None,
):
    """Combine all time diffs (µs) and histogram them. Returns (bin_edges, counts)."""
    if not time_diffs_by_channel:
        return np.array([], dtype=float), np.array([], dtype=float)

    arrs = []
    for v in time_diffs_by_channel.values():
        try:
            a = np.asarray(v, dtype=float)
        except Exception:
            continue
        if a.size:
            arrs.append(a)
    if not arrs:
        return np.array([], dtype=float), np.array([], dtype=float)

    all_d = np.concatenate(arrs)
    if lower_us is not None:
        all_d = all_d[all_d >= float(lower_us)]
    if upper_us is not None:
        all_d = all_d[all_d <= float(upper_us)]
    if all_d.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    hi = float(upper_us) if upper_us is not None else float(np.max(all_d))
    lo = float(lower_us) if lower_us is not None else float(np.min(all_d))
    if hi <= lo:
        return np.array([], dtype=float), np.array([], dtype=float)

    if n_bins is not None:
        n_bins = int(n_bins)
        if n_bins <= 0:
            return np.array([], dtype=float), np.array([], dtype=float)
        edges = np.linspace(lo, hi, n_bins + 1, dtype=float)
    else:
        if bin_width_us is None:
            bin_width_us = 1.0
        edges = np.arange(lo, hi + float(bin_width_us), float(bin_width_us), dtype=float)

    counts, edges = np.histogram(all_d, bins=edges)
    return edges, counts.astype(float)


# ----------------------------
# NEW: measurement-window joint fit helpers (shared N_scatter + N_background)
# ----------------------------
def _extract_window_xy_from_edges_counts(
    edges: np.ndarray,
    counts: np.ndarray,
    *,
    t0_us: float,
    t1_us: float,
):
    """Return (t_rel_us, y) using bin centers in [t0_us, t1_us], with t_rel=0 at t0_us."""
    edges = np.asarray(edges, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if edges.size < 2 or counts.size < 1:
        return np.array([], dtype=float), np.array([], dtype=float)

    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (centers >= float(t0_us)) & (centers <= float(t1_us))
    if not np.any(mask):
        return np.array([], dtype=float), np.array([], dtype=float)

    t_rel = centers[mask] - float(t0_us)
    y = counts[mask]
    ok = np.isfinite(t_rel) & np.isfinite(y)
    return t_rel[ok], y[ok]


def _avg_first_n(y: np.ndarray, n: int = 3) -> float:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return float("nan")
    n = int(min(max(n, 1), y.size))
    return float(np.mean(y[:n]))


def _initial_gamma_bright(*, N_scatter0: float, N_background0: float, y_end: float, T_us: float) -> float:
    # y_end = N_bg + N_scatter*exp(-gamma*T)
    if not (np.isfinite(N_scatter0) and np.isfinite(N_background0) and np.isfinite(y_end) and np.isfinite(T_us)) or T_us <= 0:
        return float("nan")
    r = (float(y_end) - float(N_background0)) / float(N_scatter0) if N_scatter0 > 0 else float("nan")
    r = float(np.clip(r, 1e-12, 1.0))
    return float(-np.log(r) / float(T_us))


def _initial_gamma_dark(*, N_scatter0: float, N_background0: float, y_end: float, T_us: float) -> float:
    # y_end = N_bg + N_scatter*(1-exp(-gamma*T))
    if not (np.isfinite(N_scatter0) and np.isfinite(N_background0) and np.isfinite(y_end) and np.isfinite(T_us)) or T_us <= 0:
        return float("nan")
    r = 1.0 - (float(y_end) - float(N_background0)) / float(N_scatter0) if N_scatter0 > 0 else float("nan")
    r = float(np.clip(r, 1e-12, 1.0))
    return float(-np.log(r) / float(T_us))


def fit_shared_scatter_background_models(
    t_b_us: np.ndarray,
    y_b: np.ndarray,
    t_d_us: np.ndarray,
    y_d: np.ndarray,
    *,
    T_us: float,
    N_scatter0: float,
    N_background0: float,
    gamma_b0: float,
    gamma_d0: float,
    grid_points: int = 60,
    refine_steps: int = 3,
):
    """
    Joint fit with shared N_scatter + N_background, separate gammas.
    Bright: y = N_bg + N_s * exp(-gamma_b t)
    Dark:   y = N_bg + N_s * (1 - exp(-gamma_d t))
    """
    t_b = np.asarray(t_b_us, dtype=float)
    y_b = np.asarray(y_b, dtype=float)
    t_d = np.asarray(t_d_us, dtype=float)
    y_d = np.asarray(y_d, dtype=float)
    if t_b.size < 3 or t_d.size < 3:
        return {"ok": False, "reason": "Not enough in-window bins to fit."}

    nb, nd = int(y_b.size), int(y_d.size)
    n = nb + nd
    Sy = float(np.sum(y_b) + np.sum(y_d))
    Sy2 = float(np.sum(y_b * y_b) + np.sum(y_d * y_d))

    def solve_shared(Sx: float, Sxx: float, Sxy: float):
        den = n * Sxx - Sx * Sx
        if not np.isfinite(den) or abs(den) < 1e-18:
            return float("nan"), float("nan")
        Ns = (n * Sxy - Sx * Sy) / den
        Nb = (Sy - Ns * Sx) / float(n)
        return float(Ns), float(Nb)

    def rss_from_sums(Sx: float, Sxx: float, Sxy: float, Ns: float, Nb: float):
        return float(
            Sy2
            + n * (Nb * Nb)
            + (Ns * Ns) * Sxx
            + 2.0 * Nb * Ns * Sx
            - 2.0 * Nb * Sy
            - 2.0 * Ns * Sxy
        )

    def finite_pos(x: float, fallback: float) -> float:
        x = float(x)
        return x if (np.isfinite(x) and x > 0) else float(fallback)

    gamma_b0 = finite_pos(gamma_b0, 1e-3)
    gamma_d0 = finite_pos(gamma_d0, 1e-3)

    gb_lo, gb_hi = 0.0, max(1e-6, gamma_b0 * 10.0)
    gd_lo, gd_hi = 0.12*2*np.pi*1E-6, max(1e-6, gamma_d0 * 10.0)

    best = {"rss": float("inf"), "gamma_b": float("nan"), "gamma_d": float("nan"), "N_scatter": float("nan"), "N_background": float("nan")}

    for _ in range(int(max(1, refine_steps))):
        gb_grid = np.linspace(gb_lo, gb_hi, int(max(10, grid_points)), dtype=float)
        gd_grid = np.linspace(gd_lo, gd_hi, int(max(10, grid_points)), dtype=float)

        Eb = np.exp(-np.outer(gb_grid, t_b))        # (G, nb)
        xb = Eb
        Sx_b = np.sum(xb, axis=1)
        Sxx_b = np.sum(xb * xb, axis=1)
        Sxy_b = np.dot(xb, y_b)

        Ed = np.exp(-np.outer(gd_grid, t_d))        # (G, nd)
        xd = 1.0 - Ed
        Sx_d = np.sum(xd, axis=1)
        Sxx_d = np.sum(xd * xd, axis=1)
        Sxy_d = np.dot(xd, y_d)

        for i, gb in enumerate(gb_grid):
            for j, gd in enumerate(gd_grid):
                Sx = float(Sx_b[i] + Sx_d[j])
                Sxx = float(Sxx_b[i] + Sxx_d[j])
                Sxy = float(Sxy_b[i] + Sxy_d[j])

                Ns, Nb = solve_shared(Sx, Sxx, Sxy)
                if not np.isfinite(Ns) or not np.isfinite(Nb) or Ns < 0:
                    continue

                rss = rss_from_sums(Sx, Sxx, Sxy, Ns, Nb)
                if rss < best["rss"]:
                    best = {"rss": float(rss), "gamma_b": float(gb), "gamma_d": float(gd), "N_scatter": float(Ns), "N_background": float(Nb)}

        # refine around best
        gb_c, gd_c = best["gamma_b"], best["gamma_d"]
        gb_span = max(1e-6, 0.35 * (gb_hi - gb_lo))
        gd_span = max(1e-6, 0.35 * (gd_hi - gd_lo))
        gb_lo, gb_hi = max(0.0, gb_c - gb_span), gb_c + gb_span
        gd_lo, gd_hi = max(0.0, gd_c - gd_span), gd_c + gd_span

    if not np.isfinite(best["gamma_b"]) or not np.isfinite(best["gamma_d"]):
        return {"ok": False, "reason": "Fit failed (no finite solution found)."}

    Ns, Nb, gb, gd = best["N_scatter"], best["N_background"], best["gamma_b"], best["gamma_d"]
    yfit_b = Nb + Ns * np.exp(-gb * t_b)
    yfit_d = Nb + Ns * (1.0 - np.exp(-gd * t_d))

    return {
        "ok": True,
        "N_scatter": float(Ns),
        "N_background": float(Nb),
        "gamma_b": float(gb),
        "gamma_d": float(gd),
        "yfit_b": np.asarray(yfit_b, dtype=float),
        "yfit_d": np.asarray(yfit_d, dtype=float),
        "init": {"N_scatter0": float(N_scatter0), "N_background0": float(N_background0), "gamma_b0": float(gamma_b0), "gamma_d0": float(gamma_d0)},
    }


# ----------------------------
# Experiment helpers
# ----------------------------
def build_sequence(
    dds_dict,
    pulse_sequencer,
    *,
    detuning_866rp: float,
    meas_length: int,
    meas_amp_397c: float,
    meas_amp_397b: float,
    meas_amp_866rp: float,
    shelving_dds_functions: dict,
    n_cycles: int,
    threshold: int,  # NEW
    name="SPAM",
):
    """
    Build + flash one Experiment_Builder sequence.
    Constraint: cooling length == measurement length.
    Also enforces one shared 397c cooling detuning for both bright/dark.
    """
    exp = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.08 * 2, N_Cycles=int(n_cycles), external_trigger=False)

    # Load cooling preset (amps/dets), but enforce 397c + 866RP afterwards.
    exp.load_cooling(length=int(meas_length))
    exp.edit_detunings({"866 RP": float(detuning_866rp)})

    exp.load_trapping()

    exp.create_section(
        name=f"{name}:Shelving",
        duration=STATE_PREP_LENGTH_US,
        dds_functions=shelving_dds_functions,
        pmt_gate_high=True,
    )

    exp.create_measurement_section(
        Length=int(meas_length),
        amplitudes_dict={"397c": float(meas_amp_397c), "397b": float(meas_amp_397b), "866 RP": float(meas_amp_866rp)},
        threshold=int(threshold),  # CHANGED (was STATE_READOUT_THRESHOLD)
        pmt_gate_high=True,
    )

    exp.build_ram_arrays()
    # exp.plot_amplitude_arrays()
    exp.flash()
    return exp


def run_sequence(dds_dict, pulse_sequencer, *, seq_kwargs: dict, n_runs: int):
    """Build/flash a sequence, run it, and return (measurement_hist, time_diffs_by_channel, stats_dict)."""
    # Build+flash sequence
    _ = build_sequence(dds_dict, pulse_sequencer, **seq_kwargs)

    exp_runner = Experiment_Runner(
        dds_dict,
        pulse_sequencer,
        timeout=100,
        pmt_threshold=2000,
        expected_fluorescence=3000,
        pulse_expected_fluorescence=3000,
        sp_threshold=None,
        load_timeout=100,
        trigger_mode="ram",
        cavity_lock=False,
    )
    # Avoid inventing/assuming attributes on the runner.
    if hasattr(exp_runner, "experiment_trap_depth"):
        exp_runner.experiment_trap_depth = 0.78

    exp_runner.clear_channels()
    exp_runner.measure_expected_fluorescence()
    exp_runner.start_experiment(N=int(n_runs))

    meas_hist = getattr(exp_runner, "measurement_hist", {}) or {}

    # Temporal diffs: only call if present.
    if hasattr(exp_runner, "get_time_diffs"):
        time_diffs = exp_runner.get_time_diffs(
            TIME_DIFFS_MODE,
            lower_cutoff=TIME_HIST_LOWER_US,
            upper_cutoff=TIME_HIST_UPPER_US
        ) or {}
    else:
        time_diffs = {}

    # Avoid guessing non-existent runner attributes; compute stats from known quantities.
    attempted = int(n_runs)
    valid = int(sum(meas_hist.values())) if meas_hist else 0

    # NEW: runner-derived P_dark / P_bright (best-effort; uses exp_runner values if present)
    n_valid_pulses_runner = (
        getattr(exp_runner, "N_Valid_Pulses", None)
        or getattr(exp_runner, "N_valid_pulses", None)
        or getattr(exp_runner, "n_valid_pulses", None)
        or 0
    )
    N_bright = exp_runner.bright_counts
    p_bright_runner = N_bright / (exp_runner.N_Valid_Pulses or 1)
    p_dark_runner = 1.0 - p_bright_runner
    # print(f"\033[91mRunner-derived: N_valid_pulses={n_valid_pulses_runner}, N_bright={N_bright}, P_bright={p_bright_runner:.4f}, P_dark={p_dark_runner:.4f}\033[0m")
    

    denom = float(n_valid_pulses_runner or 1)

    stats = {
        "attempted": attempted,
        "valid": valid,
        "n_valid_pulses_runner": int(n_valid_pulses_runner or 0),
        "p_dark_runner": p_dark_runner,
        "p_bright_runner": p_bright_runner,
    }
    return meas_hist, time_diffs, stats


# ----------------------------
# Qt worker
# ----------------------------
class SpamWorker(QtCore.QObject):
    finished = Signal(dict)  # payload
    failed = Signal(str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config

    @Slot()
    def run(self):
        try:
            dds_dict = load_dds_dict("ram", DDS_CONFIG_PATH)
            pulse_sequencer = Pulse_Sequencer()

            dark_seq = dict(
                detuning_866rp=self.config["detuning_866rp"],
                meas_length=self.config["meas_length"],
                meas_amp_397c=self.config["meas_amp_397c"],
                meas_amp_397b=self.config["meas_amp_397b"],
                meas_amp_866rp=self.config["meas_amp_866rp"],
                n_cycles=self.config["n_cycles"],
                threshold=self.config["threshold"],  # NEW
                shelving_dds_functions={
                    "397b": lambda t: float(DARK_SHELVING["397b"]) if t < 0.95 * STATE_PREP_LENGTH_US else 0.0,
                    "850 RP": lambda t: float(DARK_SHELVING["850 RP"]) if t < 0.95 * STATE_PREP_LENGTH_US else 0.0,
                    
                },
                name="Dark",
            )
            dark_hist, dark_tdiffs, dark_stats = run_sequence(
                dds_dict, pulse_sequencer, seq_kwargs=dark_seq, n_runs=self.config["n_runs"]
            )

            bright_seq = dict(
                detuning_866rp=self.config["detuning_866rp"],
                meas_length=self.config["meas_length"],
                meas_amp_397c=self.config["meas_amp_397c"],
                meas_amp_397b=self.config["meas_amp_397b"],
                meas_amp_866rp=self.config["meas_amp_866rp"],
                n_cycles=self.config["n_cycles"],
                threshold=self.config["threshold"],  # NEW
                shelving_dds_functions={
                    "866 RP": lambda t: float(BRIGHT_SHELVING["866 RP"]) if t < 0.95 * STATE_PREP_LENGTH_US else 0.0,
                    "854 SP1": lambda t: float(BRIGHT_SHELVING["854 SP1"]) if t < 0.95 * STATE_PREP_LENGTH_US else 0.0,
                },
                name="Bright",
            )
            bright_hist, bright_tdiffs, bright_stats = run_sequence(
                dds_dict, pulse_sequencer, seq_kwargs=bright_seq, n_runs=self.config["n_runs"]
            )

            self.finished.emit(
                {
                    "dark_hist": dark_hist,
                    "bright_hist": bright_hist,
                    "dark_time_diffs": dark_tdiffs,
                    "bright_time_diffs": bright_tdiffs,
                    "dark_stats": dark_stats,
                    "bright_stats": bright_stats,
                }
            )
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


# ----------------------------
# Qt UI
# ----------------------------
class SpamApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPAM (Bright/Dark)")

        self._last_dark_hist = {}
        self._last_bright_hist = {}
        self._last_dark_tdiffs = {}
        self._last_bright_tdiffs = {}
        
        #store the number of valid runs, to normalise data from bright/dark
        self._last_dark_valid = 0
        self._last_bright_valid = 0

        # NEW: keep last time-histogram arrays for fitting
        self._last_dark_time_edges = np.array([], dtype=float)
        self._last_dark_time_counts = np.array([], dtype=float)
        self._last_bright_time_edges = np.array([], dtype=float)
        self._last_bright_time_counts = np.array([], dtype=float)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QGridLayout(central)

        # Controls
        ctrl = QtWidgets.QGroupBox("Controls")
        ctrl_l = QtWidgets.QFormLayout(ctrl)

        self.n_runs = CursorIntSpinBox()
        self.n_runs.setRange(1, 1_000_000)
        self.n_runs.setValue(25)

        self.detuning_866rp = CursorDoubleSpinBox()
        self.detuning_866rp.setRange(-500.0, 500.0)
        self.detuning_866rp.setDecimals(3)
        self.detuning_866rp.setValue(35.0)

        self.n_cycles = CursorIntSpinBox()
        self.n_cycles.setRange(1, 10_000_000)
        self.n_cycles.setValue(int(N_CYCLES))

        self.meas_length = CursorIntSpinBox()
        self.meas_length.setRange(1, 10_000_000)
        self.meas_length.setValue(5000)

        self.meas_amp_397c = CursorDoubleSpinBox()
        self.meas_amp_397c.setRange(0.0, 1.0)
        self.meas_amp_397c.setSingleStep(0.01)
        self.meas_amp_397c.setValue(0.2)

        self.meas_amp_397b = CursorDoubleSpinBox()
        self.meas_amp_397b.setRange(0.0, 1.0)
        self.meas_amp_397b.setSingleStep(0.01)
        self.meas_amp_397b.setValue(0.0)

        self.meas_amp_866rp = CursorDoubleSpinBox()
        self.meas_amp_866rp.setRange(0.0, 1.0)
        self.meas_amp_866rp.setSingleStep(0.01)
        self.meas_amp_866rp.setValue(0.04)

        self.threshold = CursorIntSpinBox()
        self.threshold.setRange(0, 1_000_000)
        self.threshold.setValue(4)
        self.threshold.valueChanged.connect(self._on_threshold_changed)

        self.fidelity_out = QtWidgets.QLineEdit()
        self.fidelity_out.setReadOnly(True)

        # NEW: show the two error probabilities alongside fidelity
        self.p_dark_to_bright_out = QtWidgets.QLineEdit()
        self.p_dark_to_bright_out.setReadOnly(True)
        self.p_bright_to_dark_out = QtWidgets.QLineEdit()
        self.p_bright_to_dark_out.setReadOnly(True)

        # NEW: fit readout (4 boxes, like fidelity row; will be laid out 2x2)
        self.time_fit_r_bright_out = QtWidgets.QLineEdit(); self.time_fit_r_bright_out.setReadOnly(True)
        self.time_fit_r_dark_out   = QtWidgets.QLineEdit(); self.time_fit_r_dark_out.setReadOnly(True)
        self.time_fit_gb_out       = QtWidgets.QLineEdit(); self.time_fit_gb_out.setReadOnly(True)
        self.time_fit_gd_out       = QtWidgets.QLineEdit(); self.time_fit_gd_out.setReadOnly(True)

        # NEW: runner-derived P_dark/P_bright readout (per experiment run)
        self.runner_p_dark_run_out = QtWidgets.QLineEdit(); self.runner_p_dark_run_out.setReadOnly(True)
        self.runner_p_bright_run_out = QtWidgets.QLineEdit(); self.runner_p_bright_run_out.setReadOnly(True)

        self.run_btn = QtWidgets.QPushButton("Run SPAM Experiment")
        self.run_btn.clicked.connect(self._start_run)

        self.save_preset_btn = QtWidgets.QPushButton("Save measurement preset...")
        self.save_preset_btn.clicked.connect(self._save_measurement_preset)

        self.save_data_btn = QtWidgets.QPushButton("Save data (CSV)...")
        self.save_data_btn.clicked.connect(self._save_data_csv)

        self.status = QtWidgets.QLabel(
            f"Idle | 397c detuning fixed at {COOLING_DETUNING_397C} MHz | DDS cfg: {DDS_CONFIG_PATH}"
        )

        ctrl_l.addRow("N runs", self.n_runs)
        ctrl_l.addRow("866 RP detuning", self.detuning_866rp)
        ctrl_l.addRow("N cycles", self.n_cycles)
        ctrl_l.addRow("Measurement length (µs)", self.meas_length)
        ctrl_l.addRow("Measurement amp 397c", self.meas_amp_397c)
        ctrl_l.addRow("Measurement amp 397b", self.meas_amp_397b)
        ctrl_l.addRow("Measurement amp 866 RP", self.meas_amp_866rp)
        ctrl_l.addRow("Threshold (counts)", self.threshold)

        # REPLACE the single fidelity field with 3 columns in one row
        self._spam_metrics_row = QtWidgets.QWidget()
        _spam_metrics_l = QtWidgets.QHBoxLayout(self._spam_metrics_row)
        _spam_metrics_l.setContentsMargins(0, 0, 0, 0)
        _spam_metrics_l.addWidget(self.fidelity_out)
        _spam_metrics_l.addWidget(self.p_dark_to_bright_out)
        _spam_metrics_l.addWidget(self.p_bright_to_dark_out)
        ctrl_l.addRow("Fid | P(D→B) | P(B→D)", self._spam_metrics_row)

        # NEW: extra row with P_dark/P_bright from exp_runner (not post-processing)
        self._runner_p_row = QtWidgets.QWidget()
        _runner_p_l = QtWidgets.QHBoxLayout(self._runner_p_row)
        _runner_p_l.setContentsMargins(0, 0, 0, 0)
        _runner_p_l.addWidget(self.runner_p_dark_run_out)
        _runner_p_l.addWidget(self.runner_p_bright_run_out)
        ctrl_l.addRow("Runner P (Dark run | Bright run)", self._runner_p_row)

        # CHANGED: was ctrl_l.addRow("Time fit (shared)", self.time_fit_out)
        self._time_fit_row = QtWidgets.QWidget()
        _time_fit_g = QtWidgets.QGridLayout(self._time_fit_row)
        _time_fit_g.setContentsMargins(0, 0, 0, 0)
        _time_fit_g.setHorizontalSpacing(6)
        _time_fit_g.setVerticalSpacing(6)
        _time_fit_g.addWidget(self.time_fit_r_bright_out, 0, 0)
        _time_fit_g.addWidget(self.time_fit_r_dark_out,   0, 1)
        _time_fit_g.addWidget(self.time_fit_gb_out,       1, 0)
        _time_fit_g.addWidget(self.time_fit_gd_out,       1, 1)
        ctrl_l.addRow("Time fit (shared)", self._time_fit_row)

        ctrl_l.addRow(self.run_btn)
        ctrl_l.addRow(self.save_preset_btn)
        ctrl_l.addRow(self.save_data_btn)
        ctrl_l.addRow("Status", self.status)

        # Plots
        self.dark_time_plot = pg.PlotWidget(title="Dark PMT Count Histogram")
        self.bright_time_plot = pg.PlotWidget(title="Bright PMT Count Histogram")
        self.spam_plot = pg.PlotWidget(title="SPAM Histogram")

        for p in (self.dark_time_plot, self.bright_time_plot, self.spam_plot):
            p.showGrid(x=True, y=True, alpha=0.3)

        # make sure interactive zoom is enabled (pyqtgraph default, but keep explicit)
        self.dark_time_plot.setMouseEnabled(x=True, y=True)
        self.bright_time_plot.setMouseEnabled(x=True, y=True)

        # SPAM cutoff line (dashed red)
        self._cut_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen("r", width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self._cut_line.setPos(self._cut_line_x())

        # NEW: apply defaults from measurement.json (if present) after _cut_line exists
        self._apply_defaults_from_measurement_json(DEFAULT_MEASUREMENT_JSON_PATH)
        self._apply_defaults_from_presets_py(PRESETS_PY_PATH, preset="measurement")

        root.addWidget(ctrl, 0, 0, 1, 1)
        root.addWidget(self.dark_time_plot, 0, 1, 1, 2)
        root.addWidget(self.bright_time_plot, 1, 1, 1, 2)
        root.addWidget(self.spam_plot, 1, 0, 1, 1)

        # Thread wiring
        self._thread = None
        self._worker = None

    def _collect_config(self) -> dict:
        return {
            "n_runs": int(self.n_runs.value()),
            "detuning_866rp": float(self.detuning_866rp.value()),
            "n_cycles": int(self.n_cycles.value()),
            "meas_length": int(self.meas_length.value()),
            "meas_amp_397c": float(self.meas_amp_397c.value()),
            "meas_amp_397b": float(self.meas_amp_397b.value()),
            "meas_amp_866rp": float(self.meas_amp_866rp.value()),
            "threshold": int(self.threshold.value()),  # NEW: use UI box at run time
        }

    def _start_run(self):
        if self._thread is not None:
            return
        self.run_btn.setEnabled(False)
        self.status.setText("Running...")

        self._thread = QtCore.QThread(self)
        self._worker = SpamWorker(self._collect_config())
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    def _cleanup_thread(self):
        self._worker = None
        self._thread = None
        self.run_btn.setEnabled(True)

    def _on_failed(self, msg: str):
        self.status.setText(f"Failed: {msg}")

    def _on_finished(self, payload: dict):
        self.status.setText("Done")

        self._last_dark_hist = payload.get("dark_hist", {}) or {}
        self._last_bright_hist = payload.get("bright_hist", {}) or {}
        self._last_dark_tdiffs = payload.get("dark_time_diffs", {}) or {}
        self._last_bright_tdiffs = payload.get("bright_time_diffs", {}) or {}
        
        d = payload.get("dark_stats", {}) or {}
        b = payload.get("bright_stats", {}) or {}
        self._last_dark_valid = int(d.get("valid", 0) or 0)
        self._last_bright_valid = int(b.get("valid", 0) or 0)

        # CHANGED: show ONLY runner-derived P_bright for each experiment run
        def _fmt_runner_p(stats: dict) -> str:
            try:
                pb = float((stats or {}).get("p_bright_runner", float("nan")))
                return "nan" if not np.isfinite(pb) else f"{pb:.6g}"
            except Exception:
                return "nan"

        self.runner_p_dark_run_out.setText(_fmt_runner_p(d))
        self.runner_p_bright_run_out.setText(_fmt_runner_p(b))

        # NEW: store edges+counts so we can fit the measurement window
        self._last_dark_time_edges, self._last_dark_time_counts = self._plot_time_diffs(
            self.dark_time_plot, self._last_dark_tdiffs, pen=pg.mkPen("w", width=2)
        )
        self._last_bright_time_edges, self._last_bright_time_counts = self._plot_time_diffs(
            self.bright_time_plot, self._last_bright_tdiffs, pen=pg.mkPen("y", width=2)
        )

        self._plot_spam_hist(self._last_dark_hist, self._last_bright_hist)

        # NEW: perform joint fit + overlay curves
        self._fit_and_overlay_time_models()

        # Optional run stats in the status line
        d = payload.get("dark_stats", {})
        b = payload.get("bright_stats", {})
        self.status.setText(
            f"Done | Dark valid {d.get('valid',0)}/{d.get('attempted',0)} | Bright valid {b.get('valid',0)}/{b.get('attempted',0)}"
        )

        self._update_fidelity_box()

    def _measurement_window_bounds_us(self) -> tuple[float, float]:
        # Trigger is at end of cooling; measurement starts after 60 µs shelving.
        t0 = 60.0
        t1 = t0 + float(self.meas_length.value())
        return t0, t1

    def _add_measurement_window_lines(self, plot: pg.PlotWidget):
        t0, t1 = self._measurement_window_bounds_us()
        pen = pg.mkPen((0, 255, 0), width=2, style=QtCore.Qt.PenStyle.DashLine)
        for t in (t0, t1):
            line = pg.InfiniteLine(angle=90, movable=False, pen=pen)
            line.setPos(t)
            plot.addItem(line)

    def _plot_time_diffs(self, plot: pg.PlotWidget, time_diffs_by_channel: dict, pen):
        plot.clear()
        edges, counts = time_diffs_to_hist_xy(
            time_diffs_by_channel,
            lower_us=TIME_HIST_LOWER_US,
            upper_us=TIME_HIST_UPPER_US,
            n_bins=TIME_HIST_N_BINS,
        )
        if edges.size == 0 or counts.size == 0:
            plot.addItem(pg.TextItem("No time diffs returned by get_time_diffs(...).", color="r"))
            return np.array([], dtype=float), np.array([], dtype=float)  # NEW

        plot.plot(edges, counts, stepMode=True, fillLevel=0, brush=(100, 100, 255, 60), pen=pen)
        self._add_measurement_window_lines(plot)

        plot.setLabel("bottom", "Time (µs)")
        plot.setLabel("left", "Occurrences")
        return edges, counts  # NEW

    def _plot_spam_hist(self, dark_hist: dict, bright_hist: dict):
        self.spam_plot.clear()
        self.spam_plot.addLegend()

        xd, yd = _hist_dict_to_xy(dark_hist)
        xb, yb = _hist_dict_to_xy(bright_hist)

        if xd.size:
            self.spam_plot.addItem(pg.BarGraphItem(x=xd, height=yd, width=0.8, brush=(160, 160, 160, 90), pen="k"))
        if xb.size:
            self.spam_plot.addItem(pg.BarGraphItem(x=xb, height=yb, width=0.6, brush=(50, 120, 255, 160), pen="k"))

        # Poisson fits: force both to run over the same ks span
        kmax_d = int(xd.max()) if xd.size else 0
        kmax_b = int(xb.max()) if xb.size else 0
        kmax = min(max(kmax_d, kmax_b), int(POISSON_KMAX_CAP))
        ks_shared = np.arange(0, kmax + 1, dtype=int) if kmax > 0 else np.array([0], dtype=int)

        ks_d, exp_d, nbar_d = poisson_fit_from_hist(dark_hist, ks=ks_shared)
        ks_b, exp_b, nbar_b = poisson_fit_from_hist(bright_hist, ks=ks_shared)

        if ks_d.size and np.isfinite(nbar_d):
            self.spam_plot.plot(ks_d, exp_d, pen=pg.mkPen((120, 120, 120), width=2), name=f"Dark fit n̄={nbar_d:.2f}")
        if ks_b.size and np.isfinite(nbar_b):
            self.spam_plot.plot(ks_b, exp_b, pen=pg.mkPen((50, 120, 255), width=2), name=f"Bright fit n̄={nbar_b:.2f}")

        # Cutoff line (dashed red), kept in sync with threshold control
        self.spam_plot.addItem(self._cut_line)
        self._cut_line.setPos(self._cut_line_x())

        self.spam_plot.setLabel("bottom", "Counts in measurement window")
        self.spam_plot.setLabel("left", "Occurrences")

    def _update_fidelity_box(self):
        thr = int(self.threshold.value())
        dh = self._last_dark_hist or {}
        bh = self._last_bright_hist or {}

        if not dh or not bh:
            self.fidelity_out.setText("nan")
            self.p_dark_to_bright_out.setText("nan")
            self.p_bright_to_dark_out.setText("nan")
            return

        d_tot = float(sum(dh.values()) or 0.0)
        b_tot = float(sum(bh.values()) or 0.0)
        if d_tot <= 0 or b_tot <= 0:
            self.fidelity_out.setText("nan")
            self.p_dark_to_bright_out.setText("nan")
            self.p_bright_to_dark_out.setText("nan")
            return

        p_dark_to_bright = sum(v for k, v in dh.items() if int(k) >= thr) / d_tot
        se_dark_to_bright = np.sqrt(p_dark_to_bright * (1.0 - p_dark_to_bright) / d_tot) if d_tot > 0 else float("nan")
        p_bright_to_dark = sum(v for k, v in bh.items() if int(k) < thr) / b_tot
        se_bright_to_dark = np.sqrt(p_bright_to_dark * (1.0 - p_bright_to_dark) / b_tot) if b_tot > 0 else float("nan")
        fidelity = 1.0 - 0.5 * (p_dark_to_bright + p_bright_to_dark)

        self.p_dark_to_bright_val = p_dark_to_bright
        self.se_dark_to_bright_val = se_dark_to_bright
        self.p_bright_to_dark_val = p_bright_to_dark
        self.se_bright_to_dark_val = se_bright_to_dark

        self.fidelity_out.setText(f"{fidelity:.6f}")
        self.p_dark_to_bright_out.setText(f"{p_dark_to_bright:.6f}±{se_dark_to_bright:.6f}")
        self.p_bright_to_dark_out.setText(f"{p_bright_to_dark:.6f}±{se_bright_to_dark:.6f}")

    def _cut_line_x(self) -> float:
        # Place line BETWEEN integer-count bins: left (<thr) is dark, right (>=thr) is bright.
        return float(self.threshold.value()) - 0.5

    def _on_threshold_changed(self, _value: int):
        self._cut_line.setPos(self._cut_line_x())
        self._update_fidelity_box()

    def _measurement_preset_payload(self) -> dict:
        # Format matches your measurement.json schema.
        # The cooling preset determines the detunings used, so load it here.
        try:
            cooling_dict = _load_presets_dict(os.path.join(MEAS_PRESET_START_DIR, "presets.py"), "cooling")
            self._cooling_detunings = cooling_dict.get("detunings") or {}
        except Exception:
            self._cooling_detunings = {}

        det_397c = float(self._cooling_detunings.get("397c", COOLING_DETUNING_397C))
        det_397b = float(self._cooling_detunings.get("397b", COOLING_DETUNING_397B))

        return {
            "length": int(self.meas_length.value()),
            "threshold": int(self.threshold.value()),
            "pmt_gate_high": True,
            "amplitudes": {
                "397c": float(self.meas_amp_397c.value()),
                "397b": float(self.meas_amp_397b.value()),
                "866 RP": float(self.meas_amp_866rp.value()),
            },
            "detunings": {
                "397c": float(det_397c),
                "397b": float(det_397b),
                "866 RP": float(self.detuning_866rp.value()),
            },
            "errors": {
                "p_dark_to_bright": float(getattr(self, "p_dark_to_bright_val", None) or 0.0),
                "se_dark_to_bright": float(getattr(self, "se_dark_to_bright_val", None) or 0.0),
                "p_bright_to_dark": float(getattr(self, "p_bright_to_dark_val", None) or 0.0),
                "se_bright_to_dark": float(getattr(self, "se_bright_to_dark_val", None) or 0.0),
        }
    }


    def _measurement_preset_func_src(self) -> str:
        """
        Build a *single* top-level function definition as source code,
        suitable for adriq.experiment.upsert_autogen_function(...).
        """
        payload = self._measurement_preset_payload()
        payload_src = pprint.pformat(payload, sort_dicts=False, width=120)

        lines = payload_src.splitlines() or ["{}"]
        out = [
            "def measurement():",
            f"    # measurement preset (AUTOGENERATED by SPAM.py {datetime.now().isoformat()})",
            f"    return {lines[0]}",
        ]
        for ln in lines[1:]:
            out.append(f"    {ln}")
        out.append("")
        return "\n".join(out)

    def _save_measurement_preset(self):
        """
        Upsert measurement() into presets.py AUTOGEN region.
        Assumes presets.py already has BEGIN/END markers (otherwise fails).
        """
        presets_path = os.path.join(MEAS_PRESET_START_DIR, "presets.py")
        try:
            func_src = self._measurement_preset_func_src()
            upsert_autogen_function(presets_path, func_src)
            self.status.setText(f"Upserted measurement() into: {presets_path}")
        except Exception as e:
            self.status.setText(f"Preset upsert failed: {type(e).__name__}: {e}")


    def _write_spam_hist_csv(self, path_csv: str, hist: dict):
        x, y = _hist_dict_to_xy(hist or {})
        with open(path_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["counts_in_window", "occurrences"])
            for xi, yi in zip(x.tolist(), y.tolist()):
                w.writerow([int(xi), float(yi)])

    def _write_time_hist_csv(self, path_csv: str, time_diffs_by_channel: dict):
        edges, counts = time_diffs_to_hist_xy(
            time_diffs_by_channel or {},
            lower_us=TIME_HIST_LOWER_US,
            upper_us=TIME_HIST_UPPER_US,
            n_bins=TIME_HIST_N_BINS,
        )
        with open(path_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["bin_left_us", "bin_right_us", "occurrences"])
            if edges.size and counts.size:
                for i in range(int(counts.size)):
                    w.writerow([float(edges[i]), float(edges[i + 1]), float(counts[i])])

    def _save_data_csv(self):
        # Save what is currently plotted (last run)
        if not (self._last_dark_hist or self._last_bright_hist or self._last_dark_tdiffs or self._last_bright_tdiffs):
            self.status.setText("Nothing to save yet (run an experiment first).")
            return

        start = MEAS_PRESET_START_DIR if os.path.isdir(MEAS_PRESET_START_DIR) else ""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save SPAM data (CSV)",
            os.path.join(start, "spam_data.csv") if start else "spam_data.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        base = os.path.splitext(path)[0]
        try:
            self._write_spam_hist_csv(base + "_spam_dark.csv", self._last_dark_hist)
            self._write_spam_hist_csv(base + "_spam_bright.csv", self._last_bright_hist)
            self._write_time_hist_csv(base + "_time_dark.csv", self._last_dark_tdiffs)
            self._write_time_hist_csv(base + "_time_bright.csv", self._last_bright_tdiffs)
            self.status.setText(f"Saved CSVs: {os.path.basename(base)}_(spam/time)_(dark/bright).csv")
        except Exception as e:
            self.status.setText(f"Save failed: {type(e).__name__}: {e}")

    def _fit_and_overlay_time_models(self):
        """
        Fit only inside measurement window, with t=0 at window start:
          Bright: y = N_bg + N_s * exp(-gamma_b t)
          Dark:   y = N_bg + N_s * (1 - exp(-gamma_d t))
        and force shared N_s and N_bg.
        """
        # CHANGED: clear 4-box readout instead of single box
        for w in (self.time_fit_r_bright_out, self.time_fit_r_dark_out, self.time_fit_gb_out, self.time_fit_gd_out):
            w.setText("")

        if self._last_dark_time_edges.size < 2 or self._last_bright_time_edges.size < 2:
            self.time_fit_r_bright_out.setText("No time histogram to fit.")
            return

        t0, t1 = self._measurement_window_bounds_us()
        T = float(self.meas_length.value())

        tb, yb = _extract_window_xy_from_edges_counts(
            self._last_bright_time_edges, self._last_bright_time_counts, t0_us=t0, t1_us=t1
        )
        td, yd = _extract_window_xy_from_edges_counts(
            self._last_dark_time_edges, self._last_dark_time_counts, t0_us=t0, t1_us=t1
        )
        if tb.size < 3 or td.size < 3:
            self.time_fit_r_bright_out.setText("Not enough in-window bins to fit.")
            return

        Nv_b = float(self._last_bright_valid) if self._last_bright_valid > 0 else 1.0
        Nv_d = float(self._last_dark_valid) if self._last_dark_valid > 0 else 1.0
        yb_n = yb / Nv_b
        yd_n = yd / Nv_d
        # Initial guesses using PMT count data
        N_scatter0 = _avg_first_n(yb_n, 3)
        N_background0 = _avg_first_n(yd_n, 3)
        gamma_b0 = _initial_gamma_bright(
            N_scatter0=N_scatter0, N_background0=N_background0, y_end=float(yb_n[-1]), T_us=T
        )
        gamma_d0 = _initial_gamma_dark(
            N_scatter0=N_scatter0, N_background0=N_background0, y_end=float(yd_n[-1]), T_us=T
        )
    
        fit = fit_shared_scatter_background_models(
            tb, yb_n, td, yd_n,
            T_us=T,
            N_scatter0=N_scatter0,
            N_background0=N_background0,
            gamma_b0=gamma_b0,
            gamma_d0=gamma_d0,
        )
        if not fit.get("ok", False):
            self.time_fit_r_bright_out.setText(f"Fit failed: {fit.get('reason', 'unknown')}")
            return

        # Overlay on absolute time axis (convert fit back to raw "Occurrences")
        tb_abs = tb + t0
        td_abs = td + t0

        bright_fit_pen = pg.mkPen((200, 80, 255), width=2)  # purple/magenta
        dark_fit_pen = pg.mkPen((255, 140, 0), width=2)    # orange

        self.bright_time_plot.plot(tb_abs, fit["yfit_b"] * Nv_b, pen=bright_fit_pen)
        self.dark_time_plot.plot(td_abs, fit["yfit_d"] * Nv_d, pen=dark_fit_pen)
    
        dt_us = float(np.median(np.diff(self._last_bright_time_edges))) if self._last_bright_time_edges.size > 1 else float("nan")
        if not np.isfinite(dt_us) or dt_us <= 0:
            dt_us = 1.0

        R_dark_hz = (fit["N_background"] / dt_us) * 1e6
        R_bright_hz = ((fit["N_background"] + fit["N_scatter"]) / dt_us) * 1e6

        # CHANGED: fill 4 boxes (2 lines)
        self.time_fit_r_bright_out.setText(f"R_bright={R_bright_hz:.4g} CPS")
        self.time_fit_r_dark_out.setText(f"R_dark={R_dark_hz:.4g} CPS")
        self.time_fit_gb_out.setText(f"Γ_b/2π={fit['gamma_b']*1E6/(2*np.pi):.4g} Hz")
        self.time_fit_gd_out.setText(f"Γ_d/2π={fit['gamma_d']*1E6/(2*np.pi):.4g} Hz")

    def _apply_defaults_from_measurement_json(self, path: str):
        """Best-effort: set spinboxes from measurement.json; keep current defaults on failure."""
        try:
            if not (path and os.path.isfile(path)):
                return
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
        except Exception:
            return

        amps = (cfg.get("amplitudes") or {}) if isinstance(cfg, dict) else {}
        dets = (cfg.get("detunings") or {}) if isinstance(cfg, dict) else {}
        ui = (cfg.get("ui") or {}) if isinstance(cfg, dict) else {}

        def _set(spin, v):
            try:
                if v is None:
                    return
                spin.setValue(float(v) if hasattr(spin, "setDecimals") else int(v))
            except Exception:
                return

        _set(self.meas_length, cfg.get("length"))
        _set(self.threshold, cfg.get("threshold"))
        _set(self.meas_amp_397c, amps.get("397c"))
        _set(self.meas_amp_397b, amps.get("397b"))
        _set(self.meas_amp_866rp, amps.get("866 RP"))
        _set(self.detuning_866rp, dets.get("866 RP"))

        # Optional extras if you add them to measurement.json
        _set(self.n_runs, ui.get("n_runs"))
        _set(self.n_cycles, ui.get("n_cycles"))

        # Ensure visuals sync (in case signals were blocked by Qt during init)
        if hasattr(self, "_cut_line"):
            self._cut_line.setPos(self._cut_line_x())
        self._update_fidelity_box()

    def _apply_defaults_from_presets_py(self, presets_path: str, preset: str = "measurement"):
        """Best-effort: set spinboxes from presets.py:<preset>(); keep current defaults on failure."""
        cfg = _load_presets_dict(presets_path, preset) or {}
        if not isinstance(cfg, dict) or not cfg:
            return

        amps = cfg.get("amplitudes") or {}
        dets = cfg.get("detunings") or {}
        ui = cfg.get("ui") or {}

        def _set(spin, v):
            try:
                if v is None:
                    return
                # QDoubleSpinBox-like has setDecimals; QSpinBox-like doesn't.
                spin.setValue(float(v) if hasattr(spin, "setDecimals") else int(v))
            except Exception:
                return

        _set(self.meas_length, cfg.get("length"))
        _set(self.threshold, cfg.get("threshold"))
        _set(self.meas_amp_397c, amps.get("397c"))
        _set(self.meas_amp_397b, amps.get("397b"))
        _set(self.meas_amp_866rp, amps.get("866 RP"))
        _set(self.detuning_866rp, dets.get("866 RP"))

        # Optional extras if you add them to presets.py measurement() return dict
        _set(self.n_runs, ui.get("n_runs"))
        _set(self.n_cycles, ui.get("n_cycles"))

        # Ensure visuals sync
        if hasattr(self, "_cut_line"):
            self._cut_line.setPos(self._cut_line_x())
        self._update_fidelity_box()


def main():
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    w = SpamApp()
    w.resize(1200, 800)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
