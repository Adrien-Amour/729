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
from adriq.Resonance_Log import read_resonance, write_most_recent_resonance
import csv
from tqdm import tqdm
import time
import webbrowser
import itertools
import re
from scipy.optimize import curve_fit  # added

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

def save_spectrum(detunings, values, errs=None, n_valid=None, directory="."):
    spectra_dir = os.path.join(str(directory), "Spectra")
    os.makedirs(spectra_dir, exist_ok=True)
    out_csv = next_spectrum_filename(spectra_dir)

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

def init_experiment(probe_amplitude, probe_time_us: int, *, ram_step=0.08, N_Cycles=500, op_mode: str = "m_minus", measure_fluorescence: bool = True):
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
        pulse_expected_fluorescence=4200,
        sp_threshold=None,
        load_timeout=100,
        trigger_mode="ram",
        cavity_lock=False,
    )
    if measure_fluorescence:
        exp_runner.measure_expected_fluorescence()
    return exp_sequence, exp_runner

def dark_probability(exp_sequence, exp_runner, detuning, *, trap_depth=0.55, n_runs=3, cancel_cb=None, use_spam_correction=True):
    # CHANGED: run in cancelable chunks (N=1) so Stop becomes responsive.
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
    if use_spam_correction:
        _, _, P_dark_corrected, stderr_corrected = correct_probabilities_with_spam(P_bright, stderr, P_dark, stderr)
        print(f"\033[91mUncorrected: P_dark={P_dark:.6f} ± {stderr:.6f} | Corrected: P_dark={P_dark_corrected:.6f} ± {stderr_corrected:.6f}\033[0m")
        return P_dark_corrected, stderr_corrected, int(total_valid)
    else:
        print(f"\033[93mUncorrected: P_dark={P_dark:.6f} ± {stderr:.6f}\033[0m")
        return P_dark, stderr, int(total_valid)

# --- Qt binding compatibility (PyQt* vs PySide*) ---
Signal = getattr(QtCore, "pyqtSignal", None) or getattr(QtCore, "Signal", None)
Slot = getattr(QtCore, "pyqtSlot", None) or getattr(QtCore, "Slot", None)
if Signal is None or Slot is None:
    raise RuntimeError("Could not find Qt Signal/Slot (unsupported Qt binding via pyqtgraph.Qt).")

class ScanWorker(QtCore.QObject):
    point = Signal(str, float, float, float, int)     # (which, detuning, P_dark, stderr, N_valid)
    scan_done = Signal(str, object, object, object)   # (which, dets, vals, popt)
    status = Signal(str)
    finished = Signal()
    failed = Signal(str)
    background = Signal(float)                        # NEW: background P_dark level

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

            use_spam = self.params.get("use_spam_correction", True)

            bg_y0 = None
            if bool(p.get("measure_background", True)):
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
                    measure_fluorescence=True,
                )
                bg_runs = max(1, int(p["n_runs"]) * 2)
                bg_detuning = float(p["center"])
                bg_y0, _bg_e, _bg_nv = dark_probability(
                    exp_sequence_bg, exp_runner_bg, bg_detuning,
                    trap_depth=p["trap_depth"],
                    n_runs=bg_runs,
                    cancel_cb=self._should_stop,
                    use_spam_correction=use_spam
                )
                if bg_y0 is None:
                    self.status.emit("Stopped.")
                    self.finished.emit()
                    return

                self.background.emit(float(bg_y0))
                self.status.emit(f"Background: P_dark={float(bg_y0):.6f} (N={bg_runs})")
            else:
                self.status.emit("Background measurement: OFF")

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
                        use_spam_correction=use_spam
                    )
                    if y is None:
                        self.status.emit("Stopped.")
                        return None, None, None

                    vals.append(float(y)); errs.append(float(e)); nvs.append(int(nv))
                    self.point.emit(which, float(d), float(y), float(e), int(nv))

                dets = np.asarray(detunings, dtype=float)
                vals = np.asarray(vals, dtype=float)

                if p["save_csv"]:
                    save_spectrum(dets, vals, errs=np.asarray(errs, float), n_valid=np.asarray(nvs, int), directory=p["output_dir"])
                x0_guess = float(dets[np.nanargmax(vals)]) 
                popt, _ = fit_sinc2(dets, vals, x0_guess=x0_guess, y0_guess=bg_y0)
                self.scan_done.emit(which, dets, vals, popt)
                return dets, vals, popt

            # CHANGED: single detuning array around one center
            span = float(p["span"])
            step = float(p["step"])
            c = float(p["center"])
            dets = np.arange(c - span, c + span + 1e-12, step)

            self.status.emit("Scanning...")
            r = do_scan("single", dets, c)
            if r[0] is None:
                self.finished.emit()
                return

            # CHANGED: single-line scan => report fitted center (no splitting/B computed here)
            popt = r[2]
            self.status.emit(f"Done. Center={float(popt[1]):.6f} MHz")
            self.finished.emit()

        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            self.finished.emit()

class LiveScanApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("B-field scan (pyqtgraph)")
        self._last_fitted_center_mhz = None

        # --- controls
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QGridLayout(w)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form, 0, 0, 1, 1)

        # CHANGED: single center
        self.center = QtWidgets.QDoubleSpinBox(); self.center.setDecimals(6); self.center.setRange(-1e6, 1e6); self.center.setValue(-23.35)

        # NEW: load resonance button (sets scan center)
        self.load_res_btn = QtWidgets.QPushButton("Load resonance…")
        self.load_res_btn.setToolTip("Select a resonance log CSV and load its most recent 'Resonant Detuning' value")

        self.span = QtWidgets.QDoubleSpinBox(); self.span.setDecimals(6); self.span.setRange(0.0, 1e6); self.span.setValue(0.2)
        self.step = QtWidgets.QDoubleSpinBox(); self.step.setDecimals(6); self.step.setRange(1e-6, 1e3); self.step.setValue(0.01)  # "resolution"

        self.probe_amp = QtWidgets.QDoubleSpinBox()
        self.probe_amp.setDecimals(4)
        self.probe_amp.setRange(0.0, 1.0)
        self.probe_amp.setSingleStep(0.01)
        self.probe_amp.setValue(1)

        # NEW: probe time (µs)
        self.probe_time_us = QtWidgets.QSpinBox(); self.probe_time_us.setRange(1, 10_000_000); self.probe_time_us.setValue(6)

        self.n_runs = QtWidgets.QSpinBox(); self.n_runs.setRange(1, 999); self.n_runs.setValue(3)
        self.n_cycles = QtWidgets.QSpinBox(); self.n_cycles.setRange(1, 10_000_000); self.n_cycles.setValue(500)

        # NEW: trap depth (like in Rabi_Oscillations)
        self.trap_depth = QtWidgets.QDoubleSpinBox()
        self.trap_depth.setDecimals(3)
        self.trap_depth.setRange(0.0, 1.0)
        self.trap_depth.setSingleStep(0.01)
        self.trap_depth.setValue(0.8)

        self.output_dir = QtWidgets.QLineEdit(os.getcwd())
        self.output_dir.setToolTip("Directory to save spectrum_###.csv files")

        self.save_csv = QtWidgets.QCheckBox("Save CSV"); self.save_csv.setChecked(True)
        self.use_spam_correction = QtWidgets.QCheckBox("Apply SPAM correction")
        self.use_spam_correction.setChecked(True)


        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False)
        self.status = QtWidgets.QLabel("Idle.")

        self.write_res_btn = QtWidgets.QPushButton("Write resonance…")
        self.write_res_btn.setToolTip("Choose fitted/manual value, then select a resonance log CSV to append to.")


        # NEW: optical pumping selector
        self.op_mode = QtWidgets.QComboBox()
        self.op_mode.addItem("Pump to S1/2 m = -1/2", "m_minus")
        self.op_mode.addItem("Pump to both S1/2 m = ±1/2", "both")
        self.op_mode.setCurrentIndex(0)
        self.op_mode.setToolTip("Select optical pumping sequence")

        # NEW: background measurement toggle
        self.measure_background = QtWidgets.QCheckBox("Measure background (729 off)")
        self.measure_background.setChecked(True)
        self.measure_background.setToolTip("If enabled, measures P_dark once with 729 power=0 and uses it as y0 guess for the fit")

        form.addRow("Center [MHz]", self.center)
        form.addRow("", self.load_res_btn)  # NEW
        form.addRow("Span [MHz]", self.span)
        form.addRow("Resolution/step [MHz]", self.step)
        form.addRow("Optical pumping", self.op_mode)  # NEW
        form.addRow("Probe amplitude [0..1]", self.probe_amp)
        form.addRow("Probe time [µs]", self.probe_time_us)   # NEW
        form.addRow("N runs (per detuning point)", self.n_runs)
        form.addRow("N cycles (builder)", self.n_cycles)
        form.addRow("Trap depth", self.trap_depth)  # NEW
        form.addRow("Output dir", self.output_dir)  # NEW (was missing from UI)
        form.addRow("", self.save_csv)
        form.addRow("", self.start_btn)
        form.addRow("", self.stop_btn)
        form.addRow("", self.write_res_btn)
        form.addRow("Status", self.status)
        form.addRow("", self.measure_background)  # NEW
        form.addRow("", self.use_spam_correction)


        # --- plots
        pg.setConfigOptions(antialias=True)
        self.plots = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plots, 0, 1, 1, 1)

        # CHANGED: single plot
        self.p = self.plots.addPlot(title="Detuning scan")
        self.p.setLabel("bottom", "Detuning (MHz)")
        self.p.setLabel("left", "P_dark")
        self.p.showGrid(x=True, y=True)
        self.c = self.p.plot([], [], pen=None, symbol="o", symbolSize=5)

        self.err_item = pg.ErrorBarItem(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=0.6)
        self.p.addItem(self.err_item)

        self._err_beam = 0.0  # NEW: x-width of error bars in MHz

        self.fit = self.p.plot([], [], pen=pg.mkPen(width=2))
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(style=QtCore.Qt.PenStyle.DashLine))
        self.p.addItem(self.vline); self.vline.hide()

        # NEW: dashed horizontal red line for background level
        self.bgline = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen("r", width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.p.addItem(self.bgline); self.bgline.hide()

        self._thread = None
        self._worker = None
        self._data = {"single": ([], [], [])}   # (xs, ys, es)
        self._popt = {}
        self._current_params = None

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.write_res_btn.clicked.connect(self.write_resonance_dialog)
        self.load_res_btn.clicked.connect(self.load_resonance_dialog)  # NEW

    def _params(self):
        return dict(
            center=float(self.center.value()),               # CHANGED
            span=float(self.span.value()),
            step=float(self.step.value()),
            op_mode=str(self.op_mode.currentData() or "m_minus"),  # NEW
            probe_amplitude=float(self.probe_amp.value()),
            probe_time_us=int(self.probe_time_us.value()),   # NEW
            n_runs=int(self.n_runs.value()),
            n_cycles=int(self.n_cycles.value()),
            trap_depth=float(self.trap_depth.value()),       # NEW
            output_dir=str(self.output_dir.text()).strip() or os.getcwd(),
            save_csv=bool(self.save_csv.isChecked()),
            measure_background=bool(self.measure_background.isChecked()),  # NEW
            # advanced defaults:
            ram_step=0.1,
            use_spam_correction=bool(self.use_spam_correction.isChecked()),
        )

    def _update_x_range(self, which: str):
        xs, _ys, _es = self._data[which]
        if not xs:
            return
        xmin = float(min(xs))
        xmax = float(max(xs))
        if xmin == xmax:
            step = float((self._current_params or {}).get("step", 0.02))
            dx = max(step * 5.0, 1e-6)
            xmin -= dx
            xmax += dx
        else:
            pad = 0.05 * (xmax - xmin)
            xmin -= pad
            xmax += pad
        self.p.setXRange(xmin, xmax, padding=0.0)  # CHANGED

    @Slot()
    def start(self):
        self._data = {"single": ([], [], [])}
        self._popt = {}
        self.c.setData([], [])
        self.fit.setData([], [])
        self.vline.hide()
        self.bgline.hide()  # NEW
        self.status.setText("Starting...")

        params = self._params()
        self._current_params = params
        self._err_beam = 0.6 * float(params.get("step", 0.02))  # NEW: scale by resolution

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._thread = QtCore.QThread(self)
        self._worker = ScanWorker(params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.point.connect(self.on_point)
        self._worker.scan_done.connect(self.on_scan_done)
        self._worker.background.connect(self.on_background)  # NEW
        self._worker.status.connect(self.status.setText)
        self._worker.failed.connect(self.on_failed)
        self._worker.finished.connect(self.on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._last_fitted_center_mhz = None

        self._thread.start()

        self.err_item.setData(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=self._err_beam)

    @Slot(str, float, float, float, int)
    def on_point(self, which, detuning, p_dark, stderr, n_valid):
        xs, ys, es = self._data[which]
        xs.append(detuning); ys.append(p_dark); es.append(stderr)

        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        e = np.asarray(es, dtype=float)

        self.c.setData(x, y)
        top = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
        self.err_item.setData(x=x, y=y, top=top, bottom=top, beam=self._err_beam)

        self.p.enableAutoRange(axis="y", enable=True)
        self._update_x_range(which)

    @Slot(str, object, object, object)
    def on_scan_done(self, which, dets, vals, popt):
        self._popt[which] = popt
        xfit = np.linspace(float(np.min(dets)), float(np.max(dets)), 800)
        yfit = sinc2(xfit, *popt)

        self.fit.setData(xfit, yfit)
        self.vline.setPos(float(popt[1])); self.vline.show()
        self._last_fitted_center_mhz = float(popt[1]) 
        self.status.setText(f"Done. Center={float(popt[1]):.6f} MHz")

    @Slot(float)
    def on_background(self, level: float):
        self.bgline.setPos(float(level))
        self.bgline.show()

    @Slot(str)
    def on_failed(self, msg):
        self.status.setText(f"Error: {msg}")

    @Slot()
    def stop(self):
        # Match the Rabi pattern: set flag directly + request interruption.
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
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)  # ensure consistent UI state
        self._worker = None
        self._thread = None

    @Slot()
    def write_resonance_dialog(self):
        # --- Step 1: choose source (fitted vs manual) ---
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Write resonance")
        msg.setText("Write which resonance value to the log?")

        # Qt5/Qt6 compatibility for roles
        AcceptRole = getattr(QtWidgets.QMessageBox, "AcceptRole", None)
        ActionRole = getattr(QtWidgets.QMessageBox, "ActionRole", None)
        RejectRole = getattr(QtWidgets.QMessageBox, "RejectRole", None)
        if AcceptRole is None:  # Qt6-style
            AcceptRole = QtWidgets.QMessageBox.ButtonRole.AcceptRole
            ActionRole = QtWidgets.QMessageBox.ButtonRole.ActionRole
            RejectRole = QtWidgets.QMessageBox.ButtonRole.RejectRole

        fitted_btn = msg.addButton("Use fitted", AcceptRole)
        manual_btn = msg.addButton("Manual input", ActionRole)
        cancel_btn = msg.addButton("Cancel", RejectRole)

        # Don’t allow fitted if no fitted value exists yet
        fitted_btn.setEnabled(self._last_fitted_center_mhz is not None)
        msg.setDefaultButton(manual_btn if self._last_fitted_center_mhz is None else fitted_btn)

        msg.exec()
        clicked = msg.clickedButton()
        if clicked in (None, cancel_btn):
            return

        if clicked == fitted_btn:
            value_mhz = float(self._last_fitted_center_mhz)
        else:
            default = float(self.center.value())
            value_mhz, ok = QtWidgets.QInputDialog.getDouble(
                self,
                "Manual resonance",
                "Resonance value (MHz):",
                value=default,
                min=-1e9,
                max=1e9,
                decimals=6,
            )
            if not ok:
                return
            value_mhz = float(value_mhz)

        # --- Step 2: choose file to write to ---

        # Make sure the main window is active (helps dialogs appear in front)
        try:
            self.raise_()
            self.activateWindow()
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

        default_dir = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\Resonances"
        start_dir = default_dir if os.path.isdir(default_dir) else os.getcwd()
        # Force non-native dialog (fixes “dialog not showing” in some setups)
        try:
            # Qt5-style
            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.DontUseNativeDialog
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select resonance log CSV (append/create)",
                start_dir,
                "CSV files (*.csv);;All files (*.*)",
                options=options,
            )
        except Exception:
            # Qt6-style fallback
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select resonance log CSV (append/create)",
                start_dir,
                "CSV files (*.csv);;All files (*.*)",
                options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
            )

        print("dialog returned:", repr(path))
        if not path:
            return


        # --- Step 3: append row ---
        try:
            write_most_recent_resonance(path, value_mhz)
            self.status.setText(f"Wrote {value_mhz:.6f} MHz -> {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Write failed", f"{type(e).__name__}: {e}")

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
            self.center.setValue(res_mhz)
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