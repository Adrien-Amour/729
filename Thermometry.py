import os  # moved up so env var is set before importing pyqtgraph
# Force pyqtgraph to use PyQt5 (prevents PySide/PyQt mismatches)
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtWidgets
from pyqtgraph.Qt import QtCore  # NEW
from adriq.experiment import *
from adriq.Optomechanics import *
from adriq.tdc_functions import filter_trailing_zeros, compute_time_diffs, filter_runs
from adriq.Resonance_Log import read_resonance
import csv
from tqdm import tqdm
import time
import webbrowser
import itertools
import re
from scipy.optimize import curve_fit  # added

# NEW: thermometry helpers
HBAR = 6.626070e-34 / (2 * np.pi)
K_B = 1.380649e-23

def r_from_nbar(nbar: float) -> float:
    return float(nbar) / (float(nbar) + 1.0)

def nbar_from_r(r: float) -> float:
    r = float(r)
    return r / (1.0 - r)

def T_from_nbar(nbar: float, f_sec_hz: float) -> float:
    if nbar <= 0:
        raise ValueError("nbar must be positive for a thermal state.")
    omega = 2.0 * np.pi * f_sec_hz
    return HBAR * omega / (K_B * np.log(1 + 1.0/nbar))

def zeemanFS(J, S, L, B):
    muB = 9.274078e-24   # Bohr magneton in J/T
    hbar = 1.054572669125e-34  # Planck constant / 2pi in J·s
    gJ = 1 + (J*(J+1) + S*(S+1) - L*(L+1)) / (2*J*(J+1))
    w = gJ * muB * B / hbar      # angular frequency in rad/s
    return w/1E6 #convert to MHz
 
B = 1e-4  # Tesla to gauss
probe_amplitude = 0.1
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

def fit_sinc2(x, y, x0_guess=None, y0_guess=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x0_guess is None:
        x0_guess = x[np.argmax(y)]
    A0 = (np.max(y) - np.min(y)) or 1e-3
    w0 = max((x.max() - x.min()) / 10.0, 1e-3)
    y00 = float(np.min(y)) if y0_guess is None else float(y0_guess)
    try:
        popt, pcov = curve_fit(
            sinc2, x, y,
            p0=[A0, float(x0_guess), w0, y00],
            bounds=([0.0, -np.inf, 1e-12, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
            maxfev=20000
        )
    except Exception:
        popt, pcov = [max(0.0, A0), float(x0_guess), max(1e-12, w0), y00], np.full((4, 4), np.nan)
    return popt, pcov

def fit_sinc2_fixed_y0(x, y, *, y0_fixed: float, x0_guess=None):
    """
    Fit sinc^2 with y0 held fixed (e.g. measured background).
    Returns popt compatible with sinc2(x, A, x0, w, y0): [A, x0, w, y0_fixed]
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y0_fixed = float(y0_fixed)

    if x0_guess is None:
        x0_guess = x[np.argmax(y)]

    # work in background-subtracted space for stable A guess
    y_sub = y - y0_fixed
    A0 = float(np.max(y_sub)) if np.isfinite(np.max(y_sub)) else 1e-3
    A0 = max(A0, 1e-6)
    w0 = max((x.max() - x.min()) / 10.0, 1e-3)

    def model(x_, A, x0, w):
        return sinc2(x_, A, x0, w, y0_fixed)

    try:
        popt3, pcov3 = curve_fit(
            model, x, y,
            p0=[A0, float(x0_guess), w0],
            bounds=([0.0, -np.inf, 1e-12], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
        popt = [float(popt3[0]), float(popt3[1]), float(popt3[2]), y0_fixed]
        pcov = np.full((4, 4), np.nan)
        pcov[:3, :3] = pcov3
    except Exception:
        popt = [A0, float(x0_guess), max(1e-12, w0), y0_fixed]
        pcov = np.full((4, 4), np.nan)

    return popt, pcov

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

def save_spectrum(detunings, values, errs=None, n_valid=None, directory=".", prefix="spectrum"):
    sideband_dir = os.path.join(str(directory), "Sideband_Spectra")
    os.makedirs(sideband_dir, exist_ok=True)
    out_csv = next_spectrum_filename(sideband_dir, prefix=prefix)

    detunings = np.asarray(detunings, dtype=float)
    values = np.asarray(values, dtype=float)
    errs = None if errs is None else np.asarray(errs, dtype=float)
    n_valid = None if n_valid is None else np.asarray(n_valid, dtype=int)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        if errs is not None and n_valid is not None:
            writer.writerow(["detuning_MHz", "P_dark", "P_dark_stderr", "N_valid"])
            for d, p, e, nv in zip(detunings, values, errs, n_valid):
                writer.writerow([float(d), float(p), float(e), int(nv)])
        else:
            writer.writerow(["detuning_MHz", "P_dark"])
            for d, p in zip(detunings, values):
                writer.writerow([float(d), float(p)])
    print(f"Saved: {out_csv}")
    return out_csv

def init_experiment(probe_amplitude, probe_time_us: int, *, ram_step=0.1, N_Cycles=500, op_mode: str = "m_minus", measure_fluorescence: bool = True):
    """
    (Re)initialize hardware/sequence for a run so probe amplitude is a clean parameter.

    op_mode:
      - "m_minus": Pump to S1/2 m=-1/2 (stretched pumping)
      - "both":    Pump to both S1/2 m=±1/2 (no stretched pumping)
    """
    dds_dict = load_dds_dict(
        "ram",
        r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg",
    )
    pulse_sequencer = Pulse_Sequencer()
    exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=ram_step, N_Cycles=N_Cycles)

    exp_sequence.set_detunings(detuning_dict={
        "854 SP1": 0, "854 SP2": 0, "397b": 0, "397c": -20, "866": 0, "866 OP": 30,
        "850 RP": 0, "866 RP": 40, "729 Temp1": 0, "729 Temp2": 0
    })

    exp_sequence.load_cooling(length=2000)
    exp_sequence.load_trapping()

    # --- Optical pumping mode selection ---
    if op_mode == "m_minus":
        # Pump to S1/2 m = -1/2  (your original "top" settings)
        # exp_sequence.create_section(name="Pump To Stretch", duration=18, dds_functions={
        #     "397c": lambda t: 0.35,
        #     "866 OP": lambda t: 0.5,
        #     "854 SP1": lambda t: 0.9,
        # }, pmt_gate_high=True)
        exp_sequence.load_section("pump_to_stretch")  # use preset for this
        exp_sequence.load_section("pump_to_ground") # use preset for this]
        exp_sequence.load_section("quench_metastables") # use preset for this

    elif op_mode == "both":
        # Pump to both S1/2 m = ±1/2  (your original "bottom" settings)
        exp_sequence.load_section("quench_metastables") # use preset for this
    else:
        raise ValueError(f"Unknown op_mode: {op_mode!r}")


    exp_sequence.create_section(name="wait before probe", duration=2, dds_functions={}, pmt_gate_high=False)

    # NOTE: freeze amplitude for this run (simple + predictable)
    exp_sequence.create_section(name="Probe", duration=int(probe_time_us), dds_functions={
        "729 Temp1": (lambda t, amp=float(probe_amplitude): amp),
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
        pulse_expected_fluorescence=5000,
        sp_threshold=None,
        load_timeout=100,
        trigger_mode="ram",
        cavity_lock=False,
    )
    if measure_fluorescence:
        exp_runner.measure_expected_fluorescence()
    return exp_sequence, exp_runner

def dark_probability(exp_sequence, exp_runner, detuning, *, trap_depth=0.55, n_runs=3, cancel_cb=None):
    exp_sequence.edit_detunings(detuning_dict={"729 Temp1": detuning})
    exp_sequence.build_ram_arrays()
    exp_sequence.flash()

    exp_runner.experiment_trap_depth = trap_depth

    total_bright = 0
    total_valid = 0


    exp_runner.clear_channels()
    exp_runner.start_experiment(N=n_runs)
    total_bright = int(getattr(exp_runner, "bright_counts", 0) or 0)
    total_valid = int(getattr(exp_runner, "N_Valid_Pulses", 0) or 0)

    P_bright = total_bright / (total_valid or 1)
    P_dark = 1.0 - float(P_bright)
    stderr = float(np.sqrt(P_dark * (1.0 - P_dark) / total_valid)) if total_valid > 0 else float("nan")
    P_dark_corrected, stderr_corrected, _, _ = correct_probabilities_with_spam(P_dark, stderr, P_bright, stderr)
    print(f"\033[91mUncorrected: P_dark={P_dark:.6f} ± {stderr:.6f} | Corrected: P_dark={P_dark_corrected:.6f} ± {stderr_corrected:.6f}\033[0m")
    return P_dark_corrected, stderr_corrected, int(total_valid)

# --- Qt binding compatibility (PyQt* vs PySide*) ---
Signal = getattr(QtCore, "pyqtSignal", None) or getattr(QtCore, "Signal", None)
Slot = getattr(QtCore, "pyqtSlot", None) or getattr(QtCore, "Slot", None)
if Signal is None or Slot is None:
    raise RuntimeError("Could not find Qt Signal/Slot (unsupported Qt binding via pyqtgraph.Qt).")

class ScanWorker(QtCore.QObject):
    point = Signal(str, float, float, float, int)     # (which, detuning, P_dark, stderr, N_valid)
    scan_done = Signal(str, object, object, object)   # (which, dets, vals, popt)
    background_measured = Signal(float, int)          # (bg_y0, bg_runs)
    status = Signal(str)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._stop = False

    @Slot()
    def stop(self):
        self._stop = True

    def _should_stop(self) -> bool:
        try:
            if QtCore.QThread.currentThread().isInterruptionRequested():
                return True
        except Exception:
            pass
        return bool(self._stop)

    @Slot()
    def run(self):
        try:
            p = self.params
            self.status.emit("Initializing experiment...")
            exp_sequence, exp_runner = init_experiment(
                p["probe_amplitude"],
                p["probe_time_us"],
                ram_step=p["ram_step"],
                N_Cycles=p["n_cycles"],
                op_mode=p["op_mode"],
            )

            # NEW: background (729 off) before scanning
            if self._should_stop():
                self.status.emit("Stopped.")
                self.finished.emit()
                return

            self.status.emit("Measuring background (729 off)...")
            exp_sequence_bg, exp_runner_bg = init_experiment(
                0.0,
                p["probe_time_us"],
                ram_step=p["ram_step"],
                N_Cycles=p["n_cycles"],
                op_mode=p["op_mode"],
                measure_fluorescence=False,
            )
            bg_runs = max(1, int(p["n_runs"]) * 2)
            bg_detuning = float(p["resonance"])
            bg_y0, _bg_e, _bg_nv = dark_probability(
                exp_sequence_bg, exp_runner_bg, bg_detuning,
                trap_depth=p["trap_depth"],
                n_runs=bg_runs,
                cancel_cb=self._should_stop,
            )
            if bg_y0 is None:
                self.status.emit("Stopped.")
                self.finished.emit()
                return
            self.background_measured.emit(float(bg_y0), int(bg_runs))
            self.status.emit(f"Background: P_dark={float(bg_y0):.6f} (N={bg_runs})")

            def do_scan(which, detunings, x0_guess):
                vals, errs, nvs = [], [], []
                for d in detunings:
                    if self._should_stop():
                        self.status.emit("Stopped.")
                        return None, None, None

                    y, e, nv = dark_probability(
                        exp_sequence, exp_runner, float(d),
                        trap_depth=p["trap_depth"],
                        n_runs=p["n_runs"],
                        cancel_cb=self._should_stop,
                    )
                    if y is None:
                        self.status.emit("Stopped.")
                        return None, None, None

                    vals.append(float(y)); errs.append(float(e)); nvs.append(int(nv))
                    self.point.emit(which, float(d), float(y), float(e), int(nv))

                dets = np.asarray(detunings, dtype=float)
                vals = np.asarray(vals, dtype=float)

                if p["save_csv"]:
                    save_spectrum(
                        dets, vals,
                        errs=np.asarray(errs, float),
                        n_valid=np.asarray(nvs, int),
                        directory=p["output_dir"],
                        prefix=f"{which}_spectrum",
                    )

                popt, _ = fit_sinc2_fixed_y0(dets, vals, y0_fixed=bg_y0, x0_guess=x0_guess)
                self.scan_done.emit(which, dets, vals, popt)
                return dets, vals, popt

            # NEW: scan at (resonance - secular) and (resonance + secular)
            span = float(p["span"])
            step = float(p["step"])
            res = float(p["resonance"])
            nu = float(p["secular_freq"])

            red_c = res - nu
            blue_c = res + nu

            red_dets = np.arange(red_c - span, red_c + span + 1e-12, step)
            blue_dets = np.arange(blue_c - span, blue_c + span + 1e-12, step)

            self.status.emit("Scanning red sideband (res - ν)...")
            r_red = do_scan("red", red_dets, red_c)
            if r_red[0] is None:
                self.finished.emit()
                return

            self.status.emit("Scanning blue sideband (res + ν)...")
            r_blue = do_scan("blue", blue_dets, blue_c)
            if r_blue[0] is None:
                self.finished.emit()
                return

            red_center = float(r_red[2][1])
            blue_center = float(r_blue[2][1])
            nu_fit = 0.5 * (blue_center - red_center)
            res_fit = 0.5 * (blue_center + red_center)

            self.status.emit(
                f"Done. red={red_center:.6f} MHz, blue={blue_center:.6f} MHz, "
                f"ν={(nu_fit):.6f} MHz, res={(res_fit):.6f} MHz"
            )
            self.finished.emit()

        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            self.finished.emit()

class LiveScanApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("B-field scan (pyqtgraph)")

        # --- controls
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QGridLayout(w)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form, 0, 0, 1, 1)

        # REPLACED: single center -> resonance + secular frequency
        self.resonance = QtWidgets.QDoubleSpinBox()
        self.resonance.setDecimals(6)
        self.resonance.setRange(-1e6, 1e6)
        self.resonance.setValue(-23.35)

        # NEW: load resonance button (sets resonance used for sideband centers)
        self.load_res_btn = QtWidgets.QPushButton("Load resonance…")
        self.load_res_btn.setToolTip("Select a resonance log CSV and load its most recent 'Resonant Detuning' value")

        self.secular_freq = QtWidgets.QDoubleSpinBox()
        self.secular_freq.setDecimals(6)
        self.secular_freq.setRange(0.0, 1e6)
        self.secular_freq.setValue(2.33)
        self.secular_freq.setToolTip("Rough secular frequency in MHz (scan at res±ν)")

        self.span = QtWidgets.QDoubleSpinBox(); self.span.setDecimals(6); self.span.setRange(0.0, 1e6); self.span.setValue(0.04)
        self.step = QtWidgets.QDoubleSpinBox(); self.step.setDecimals(6); self.step.setRange(1e-6, 1e3); self.step.setValue(0.01)  # "resolution"

        self.probe_amp = QtWidgets.QDoubleSpinBox()
        self.probe_amp.setDecimals(4)
        self.probe_amp.setRange(0.0, 1.0)
        self.probe_amp.setSingleStep(0.01)
        self.probe_amp.setValue(0.1)

        # NEW: probe time (µs)
        self.probe_time_us = QtWidgets.QSpinBox(); self.probe_time_us.setRange(1, 10_000_000); self.probe_time_us.setValue(15)

        self.n_runs = QtWidgets.QSpinBox(); self.n_runs.setRange(1, 999); self.n_runs.setValue(3)
        self.n_cycles = QtWidgets.QSpinBox(); self.n_cycles.setRange(1, 10_000_000); self.n_cycles.setValue(500)

        self.output_dir = QtWidgets.QLineEdit(os.getcwd())
        self.output_dir.setToolTip("Directory to save spectrum_###.csv files")

        self.save_csv = QtWidgets.QCheckBox("Save CSV"); self.save_csv.setChecked(True)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False)
        self.status = QtWidgets.QLabel("Idle.")

        # NEW: optical pumping selector
        self.op_mode = QtWidgets.QComboBox()
        self.op_mode.addItem("Pump to S1/2 m = -1/2", "m_minus")
        self.op_mode.addItem("Pump to both S1/2 m = ±1/2", "both")
        self.op_mode.setCurrentIndex(0)
        self.op_mode.setToolTip("Select optical pumping sequence")

        # NEW: experiment trap depth input
        self.trap_depth = QtWidgets.QDoubleSpinBox()
        self.trap_depth.setDecimals(3)
        self.trap_depth.setRange(0.0, 2.0)
        self.trap_depth.setSingleStep(0.01)
        self.trap_depth.setValue(0.8)
        self.trap_depth.setToolTip("Experiment_Runner.experiment_trap_depth")

        form.addRow("Resonance [MHz]", self.resonance)
        form.addRow("", self.load_res_btn)  # NEW
        form.addRow("Secular freq ν [MHz]", self.secular_freq)
        form.addRow("Span [MHz]", self.span)
        form.addRow("Resolution/step [MHz]", self.step)
        form.addRow("Optical pumping", self.op_mode)  # NEW
        form.addRow("Probe amplitude [0..1]", self.probe_amp)
        form.addRow("Probe time [µs]", self.probe_time_us)   # NEW
        form.addRow("N runs (per detuning point)", self.n_runs)
        form.addRow("N cycles (builder)", self.n_cycles)
        form.addRow("Output dir", self.output_dir)  # NEW (was missing from UI)
        form.addRow("", self.save_csv)
        form.addRow("", self.start_btn)
        form.addRow("", self.stop_btn)
        form.addRow("Status", self.status)
        form.addRow("Trap depth", self.trap_depth)  # NEW

        # --- plots
        pg.setConfigOptions(antialias=True)
        self.plots = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plots, 0, 1, 1, 1)

        # REPLACED: single plot -> two plots (red/blue)
        self.p_red = self.plots.addPlot(title="Red sideband scan (res - ν)")
        self.p_red.setLabel("bottom", "Detuning (MHz)")
        self.p_red.setLabel("left", "P_dark")
        self.p_red.showGrid(x=True, y=True)
        self.c_red = self.p_red.plot([], [], pen=None, symbol="o", symbolSize=5)
        self.err_red = pg.ErrorBarItem(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=0.6)
        self.p_red.addItem(self.err_red)
        self.fit_red = self.p_red.plot([], [], pen=pg.mkPen(width=2))
        self.vline_red = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(style=QtCore.Qt.PenStyle.DashLine))
        self.p_red.addItem(self.vline_red); self.vline_red.hide()

        # NEW: red dashed horizontal background line (red plot)
        self.bgline_red = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen("r", width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.p_red.addItem(self.bgline_red); self.bgline_red.hide()

        self.plots.nextRow()

        self.p_blue = self.plots.addPlot(title="Blue sideband scan (res + ν)")
        self.p_blue.setLabel("bottom", "Detuning (MHz)")
        self.p_blue.setLabel("left", "P_dark")
        self.p_blue.showGrid(x=True, y=True)
        self.c_blue = self.p_blue.plot([], [], pen=None, symbol="o", symbolSize=5)
        self.err_blue = pg.ErrorBarItem(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=0.6)
        self.p_blue.addItem(self.err_blue)
        self.fit_blue = self.p_blue.plot([], [], pen=pg.mkPen(width=2))
        self.vline_blue = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(style=QtCore.Qt.PenStyle.DashLine))
        self.p_blue.addItem(self.vline_blue); self.vline_blue.hide()

        # NEW: red dashed horizontal background line (blue plot)
        self.bgline_blue = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen("r", width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.p_blue.addItem(self.bgline_blue); self.bgline_blue.hide()

        # NEW: always define result widgets early so start() can clear them safely
        self._bg_y0 = None
        self._bg_runs = None

        self.sec_freq_mhz_out = None
        self.A_red_out = None
        self.A_blue_out = None
        self.nbar_out = None
        self.T_mK_out = None

        results = QtWidgets.QGroupBox("Results")
        results_form = QtWidgets.QFormLayout(results)

        def _ro():
            le = QtWidgets.QLineEdit()
            le.setReadOnly(True)
            return le

        self.sec_freq_mhz_out = _ro()
        self.A_red_out = _ro()
        self.A_blue_out = _ro()
        self.nbar_out = _ro()
        self.T_mK_out = _ro()

        results_form.addRow("Secular freq |(blue-red)/2| [MHz]", self.sec_freq_mhz_out)
        results_form.addRow("A_red over bg", self.A_red_out)
        results_form.addRow("A_blue over bg", self.A_blue_out)
        results_form.addRow("n̄", self.nbar_out)
        results_form.addRow("T [mK]", self.T_mK_out)

        # NOTE: use the same 'layout' you already created in __init__
        layout.addWidget(results, 1, 0, 1, 1)

        self._thread = None
        self._worker = None
        self._data = {"red": ([], []), "blue": ([], [])}
        self._popt = {}
        self._current_params = None

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.load_res_btn.clicked.connect(self.load_resonance_dialog)  # NEW

    def _params(self):
        return dict(
            resonance=float(self.resonance.value()),
            secular_freq=float(self.secular_freq.value()),
            span=float(self.span.value()),
            step=float(self.step.value()),
            op_mode=str(self.op_mode.currentData() or "m_minus"),
            probe_amplitude=float(self.probe_amp.value()),
            probe_time_us=int(self.probe_time_us.value()),
            n_runs=int(self.n_runs.value()),
            n_cycles=int(self.n_cycles.value()),
            output_dir=str(self.output_dir.text()).strip() or os.getcwd(),
            save_csv=bool(self.save_csv.isChecked()),
            # advanced defaults:
            ram_step=0.1,
            trap_depth=float(self.trap_depth.value()),  # CHANGED (was hardcoded 0.55)
        )

    def _update_x_range(self, which: str):
        xs, ys, es = self._data[which]
        if not xs:
            return
        xmin = float(min(xs)); xmax = float(max(xs))
        if xmin == xmax:
            step = float((self._current_params or {}).get("step", 0.02))
            dx = max(step * 5.0, 1e-6)
            xmin -= dx; xmax += dx
        else:
            pad = 0.05 * (xmax - xmin)
            xmin -= pad; xmax += pad

        # NEW: apply to the relevant plot
        (self.p_red if which == "red" else self.p_blue).setXRange(xmin, xmax, padding=0.0)

    @Slot()
    def start(self):
        # KEEP THIS start(): it is the one that starts the thread/worker.
        # Add the clears/reset at the top (so you don't need a second start()).
        self._bg_y0 = None
        self._bg_runs = None
        for w in (self.sec_freq_mhz_out, self.A_red_out, self.A_blue_out, self.nbar_out, self.T_mK_out):
            if w is not None:
                w.clear()

        self._data = {"red": ([], [], []), "blue": ([], [], [])}
        self._popt = {}

        self.c_red.setData([], []); self.fit_red.setData([], []); self.vline_red.hide()
        self.c_blue.setData([], []); self.fit_blue.setData([], []); self.vline_blue.hide()

        # NEW: hide background lines until a new background is measured
        self.bgline_red.hide()
        self.bgline_blue.hide()

        self.status.setText("Starting...")
        params = self._params()
        self._current_params = params
        self._err_beam = 0.6 * float(params.get("step", 0.02))

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._thread = QtCore.QThread(self)
        self._worker = ScanWorker(params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.point.connect(self.on_point)
        self._worker.scan_done.connect(self.on_scan_done)
        self._worker.background_measured.connect(self.on_background_measured)
        self._worker.status.connect(self.status.setText)
        self._worker.failed.connect(self.on_failed)
        self._worker.finished.connect(self.on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    @Slot(str, float, float, float, int)
    def on_point(self, which, detuning, p_dark, stderr, n_valid):
        xs, ys, es = self._data[which]
        xs.append(detuning); ys.append(p_dark); es.append(stderr)

        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        e = np.asarray(es, dtype=float)
        top = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)

        if which == "red":
            self.c_red.setData(x, y)
            self.err_red.setData(x=x, y=y, top=top, bottom=top, beam=self._err_beam)
            self.p_red.enableAutoRange(axis="y", enable=True)
        else:
            self.c_blue.setData(x, y)
            self.err_blue.setData(x=x, y=y, top=top, bottom=top, beam=self._err_beam)
            self.p_blue.enableAutoRange(axis="y", enable=True)

        self._update_x_range(which)

    @Slot(str, object, object, object)
    def on_scan_done(self, which, dets, vals, popt):
        self._popt[which] = popt
        xfit = np.linspace(float(np.min(dets)), float(np.max(dets)), 800)
        yfit = sinc2(xfit, *popt)
        center = float(popt[1])

        if which == "red":
            self.fit_red.setData(xfit, yfit)
            self.vline_red.setPos(center); self.vline_red.show()
        else:
            self.fit_blue.setData(xfit, yfit)
            self.vline_blue.setPos(center); self.vline_blue.show()

        # keep worker's final status as the authoritative "done" message
        self.status.setText(f"{which} done. Center={center:.6f} MHz")
        self._update_results()  # NEW

    @Slot(str)
    def on_failed(self, msg):
        self.status.setText(f"Error: {msg}")

    @Slot()
    def stop(self):
        if getattr(self, "_worker", None) is not None:
            try:
                self._worker.stop()
            except Exception:
                pass
        if getattr(self, "_thread", None) is not None:
            try:
                self._thread.requestInterruption()
            except Exception:
                pass
        self.stop_btn.setEnabled(False)
        self.status.setText("Stopping...")

    @Slot()
    def on_finished(self):
        # FIX: on_finished should only restore UI state; do NOT create widgets here.
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._worker = None
        self._thread = None

    @Slot(float, int)
    def on_background_measured(self, bg_y0: float, bg_runs: int):
        self._bg_y0 = float(bg_y0)
        self._bg_runs = int(bg_runs)

        # NEW: show background on both plots
        self.bgline_red.setPos(self._bg_y0); self.bgline_red.show()
        self.bgline_blue.setPos(self._bg_y0); self.bgline_blue.show()

        self._update_results()

    # NEW/RESTORED: results computation + UI update (was missing, causing AttributeError)
    def _update_results(self):
        bg = getattr(self, "_bg_y0", None)
        if bg is None:
            return
        popt = getattr(self, "_popt", {}) or {}
        if "red" not in popt or "blue" not in popt:
            return

        A_r, x0_r, _w_r, y0_r = map(float, popt["red"])
        A_b, x0_b, _w_b, y0_b = map(float, popt["blue"])

        # amplitude over measured background, using fitted peak value (y0 + A) minus bg
        A_red_over_bg = (y0_r + A_r) - float(bg)
        A_blue_over_bg = (y0_b + A_b) - float(bg)

        f_sec_mhz = abs(0.5 * (x0_b - x0_r))
        f_sec_hz = f_sec_mhz * 1e6

        r = (A_red_over_bg / A_blue_over_bg) if (A_blue_over_bg > 0.0) else float("nan")
        nbar = nbar_from_r(r) if (np.isfinite(r) and 0.0 < r < 1.0) else float("nan")
        T_mK = (T_from_nbar(nbar, f_sec_hz) * 1e3) if (np.isfinite(nbar) and f_sec_hz > 0.0) else float("nan")

        # write to UI if widgets exist
        if getattr(self, "sec_freq_mhz_out", None) is not None:
            self.sec_freq_mhz_out.setText(f"{f_sec_mhz:.6f}")
        if getattr(self, "A_red_out", None) is not None:
            self.A_red_out.setText(f"{A_red_over_bg:.6g}")
        if getattr(self, "A_blue_out", None) is not None:
            self.A_blue_out.setText(f"{A_blue_over_bg:.6g}")
        if getattr(self, "nbar_out", None) is not None:
            self.nbar_out.setText(f"{nbar:.6g}" if np.isfinite(nbar) else "")
        if getattr(self, "T_mK_out", None) is not None:
            self.T_mK_out.setText(f"{T_mK:.6g}" if np.isfinite(T_mK) else "")

    @Slot()
    def load_resonance_dialog(self):
        try:
            # Bring window forward (helps dialogs appear on top)
            try:
                self.raise_()
                self.activateWindow()
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass

            default_dir = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\Resonances"
            start_dir = default_dir if os.path.isdir(default_dir) else os.getcwd()

            # Prefer non-native dialog (more reliable on some setups)
            try:
                options = QtWidgets.QFileDialog.Options()
                options |= QtWidgets.QFileDialog.DontUseNativeDialog
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self,
                    "Select resonance log CSV",
                    start_dir,
                    "CSV files (*.csv);;All files (*.*)",
                    options=options,
                )
            except Exception:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self,
                    "Select resonance log CSV",
                    start_dir,
                    "CSV files (*.csv);;All files (*.*)",
                    options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
                )

            if not path:
                return

            res_mhz = float(read_resonance(path))
            self.resonance.setValue(res_mhz)
            self.status.setText(f"Loaded resonance: {res_mhz:.6f} MHz")
        except Exception as e:
            self.status.setText(f"Load resonance failed: {type(e).__name__}: {e}")

def main():
    app = QtWidgets.QApplication([])
    win = LiveScanApp()
    win.resize(1200, 700)
    win.show()
    app.exec()

if __name__ == "__main__":
    main()

# --- removed: Matplotlib live plotting + blocking scan code (was crashing AnyDesk)