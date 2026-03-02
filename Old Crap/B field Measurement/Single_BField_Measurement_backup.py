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

def fit_sinc2(x, y, x0_guess=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x0_guess is None:
        x0_guess = x[np.argmax(y)]
    A0 = (np.max(y) - np.min(y)) or 1e-3
    # width guess ~ span/10; avoid zero
    w0 = max((x.max() - x.min()) / 10.0, 1e-3)
    y00 = float(np.min(y))
    try:
        popt, pcov = curve_fit(
            sinc2, x, y,
            p0=[A0, float(x0_guess), w0, y00],
            maxfev=20000
        )
    except Exception:
        popt, pcov = [A0, float(x0_guess), w0, y00], np.full((4, 4), np.nan)
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

def save_spectrum(detunings, values, directory="."):
    out_csv = next_spectrum_filename(directory)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["detuning_MHz", "P_dark"])
        for d, p in zip(detunings, values):
            writer.writerow([float(d), float(p)])
    print(f"Saved: {out_csv}")
    return out_csv

def init_experiment(probe_amplitude, probe_time_us: int, *, ram_step=0.08, N_Cycles=500):
    """
    (Re)initialize hardware/sequence for a run so probe amplitude is a clean parameter.
    """
    dds_dict = load_dds_dict(
        "ram",
        r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg",
    )
    pulse_sequencer = Pulse_Sequencer()
    exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=ram_step, N_Cycles=N_Cycles)

    exp_sequence.set_detunings(detuning_dict={
        "854 SP1": 0, "854 SP2": 0, "397a": 0, "397c": -20, "866": 0, "866 OP": 30,
        "850 RP": 0, "866 RP": 40, "729 Temp1": 0, "729 Temp2": 0
    })

    exp_sequence.load_cooling(length=2000)
    exp_sequence.load_trapping()

    exp_sequence.create_section(name="Pump To Stretch", duration=18, dds_functions={
        "397c": lambda t: 0.35,
        "866 OP": lambda t: 0.5,
        "854 SP1": lambda t: 0.2,
    }, pmt_gate_high=True)

    exp_sequence.create_section(name="Pump To Ground", duration=4, dds_functions={
        "850 RP":  lambda t: 0.35 if t >= 1.0 else 0.0,
        "854 SP1": lambda t: 0.20 if t >= 1.0 else 0.0,
    }, pmt_gate_high=True)

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
        pulse_expected_fluorescence=7000,
        sp_threshold=None,
        load_timeout=100,
        trigger_mode="ram",
        cavity_lock=False,
    )
    exp_runner.measure_expected_fluorescence()
    return exp_sequence, exp_runner

def dark_probability(exp_sequence, exp_runner, detuning, *, trap_depth=0.55, n_runs=3):
    # CHANGED: n_runs == number of repeated Experiment_Runner runs at this detuning point
    exp_sequence.edit_detunings(detuning_dict={"729 Temp1": detuning})
    exp_sequence.build_ram_arrays()
    exp_sequence.flash()

    exp_runner.clear_channels()
    exp_runner.experiment_trap_depth = trap_depth
    exp_runner.clear_channels()
    exp_runner.start_experiment(N=int(n_runs))

    N_bright = exp_runner.bright_counts
    P_bright = N_bright / (exp_runner.N_Valid_Pulses or 1)
    return 1.0 - P_bright

# --- Qt binding compatibility (PyQt* vs PySide*) ---
Signal = getattr(QtCore, "pyqtSignal", None) or getattr(QtCore, "Signal", None)
Slot = getattr(QtCore, "pyqtSlot", None) or getattr(QtCore, "Slot", None)
if Signal is None or Slot is None:
    raise RuntimeError("Could not find Qt Signal/Slot (unsupported Qt binding via pyqtgraph.Qt).")

class ScanWorker(QtCore.QObject):
    point = Signal(str, float, float)                 # (which, detuning, P_dark)
    scan_done = Signal(str, object, object, object)   # (which, dets, vals, popt)
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

    @Slot()
    def run(self):
        try:
            p = self.params
            self.status.emit("Initializing experiment...")
            exp_sequence, exp_runner = init_experiment(
                p["probe_amplitude"],
                p["probe_time_us"],          # NEW
                ram_step=p["ram_step"],
                N_Cycles=p["n_cycles"],
            )

            def do_scan(which, detunings, x0_guess):
                vals = []
                for d in detunings:
                    if self._stop:
                        self.status.emit("Stopped.")
                        return None, None, None

                    y = dark_probability(
                        exp_sequence, exp_runner, float(d),
                        trap_depth=p["trap_depth"],
                        n_runs=p["n_runs"],
                    )
                    vals.append(float(y))
                    self.point.emit(which, float(d), float(y))

                dets = np.asarray(detunings, dtype=float)
                vals = np.asarray(vals, dtype=float)

                if p["save_csv"]:
                    save_spectrum(dets, vals, directory=p["output_dir"])

                popt, _ = fit_sinc2(dets, vals, x0_guess=x0_guess)
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

        # --- controls
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QGridLayout(w)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form, 0, 0, 1, 1)

        # CHANGED: single center
        self.center = QtWidgets.QDoubleSpinBox(); self.center.setDecimals(6); self.center.setRange(-1e6, 1e6); self.center.setValue(-23.35)

        self.span = QtWidgets.QDoubleSpinBox(); self.span.setDecimals(6); self.span.setRange(0.0, 1e6); self.span.setValue(2.0)
        self.step = QtWidgets.QDoubleSpinBox(); self.step.setDecimals(6); self.step.setRange(1e-6, 1e3); self.step.setValue(0.02)  # "resolution"
        self.probe_amp = QtWidgets.QDoubleSpinBox()

        # NEW: probe time (µs)
        self.probe_time_us = QtWidgets.QSpinBox(); self.probe_time_us.setRange(1, 10_000_000); self.probe_time_us.setValue(15)

        self.n_runs = QtWidgets.QSpinBox(); self.n_runs.setRange(1, 999); self.n_runs.setValue(3)
        self.n_cycles = QtWidgets.QSpinBox(); self.n_cycles.setRange(1, 10_000_000); self.n_cycles.setValue(500)

        self.output_dir = QtWidgets.QLineEdit(os.getcwd())
        self.save_csv = QtWidgets.QCheckBox("Save CSV"); self.save_csv.setChecked(True)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False)
        self.status = QtWidgets.QLabel("Idle.")

        form.addRow("Center [MHz]", self.center)
        form.addRow("Span [MHz]", self.span)
        form.addRow("Resolution/step [MHz]", self.step)
        form.addRow("Probe amplitude [0..1]", self.probe_amp)
        form.addRow("Probe time [µs]", self.probe_time_us)   # NEW
        form.addRow("N runs (per detuning point)", self.n_runs)
        form.addRow("N cycles (builder)", self.n_cycles)
        form.addRow("", self.save_csv)
        form.addRow("", self.start_btn)
        form.addRow("", self.stop_btn)
        form.addRow("Status", self.status)

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
        self.p.getViewBox().setAutoVisible(y=True)
        self.p.enableAutoRange(axis="y", enable=True)

        self.fit = self.p.plot([], [], pen=pg.mkPen(width=2))
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(style=QtCore.Qt.PenStyle.DashLine))
        self.p.addItem(self.vline); self.vline.hide()

        self._thread = None
        self._worker = None
        self._data = {"single": ([], [])}   # CHANGED
        self._popt = {}
        self._current_params = None

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

    def _params(self):
        return dict(
            center=float(self.center.value()),               # CHANGED
            span=float(self.span.value()),
            step=float(self.step.value()),
            probe_amplitude=float(self.probe_amp.value()),
            probe_time_us=int(self.probe_time_us.value()),   # NEW
            n_runs=int(self.n_runs.value()),
            n_cycles=int(self.n_cycles.value()),
            output_dir=str(self.output_dir.text()).strip() or os.getcwd(),
            save_csv=bool(self.save_csv.isChecked()),
            # advanced defaults:
            ram_step=0.08,
            trap_depth=0.55,
        )

    def _update_x_range(self, which: str):
        xs, _ys = self._data[which]
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
        self._data = {"single": ([], [])}
        self._popt = {}
        self.c.setData([], [])
        self.fit.setData([], [])
        self.vline.hide()
        self.status.setText("Starting...")

        params = self._params()
        self._current_params = params

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._thread = QtCore.QThread(self)
        self._worker = ScanWorker(params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.point.connect(self.on_point)
        self._worker.scan_done.connect(self.on_scan_done)
        self._worker.status.connect(self.status.setText)
        self._worker.failed.connect(self.on_failed)
        self._worker.finished.connect(self.on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    @Slot(str, float, float)
    def on_point(self, which, detuning, p_dark):
        xs, ys = self._data[which]
        xs.append(detuning); ys.append(p_dark)
        self.c.setData(xs, ys)
        self.p.enableAutoRange(axis="y", enable=True)
        self._update_x_range(which)

    @Slot(str, object, object, object)
    def on_scan_done(self, which, dets, vals, popt):
        self._popt[which] = popt
        xfit = np.linspace(float(np.min(dets)), float(np.max(dets)), 800)
        yfit = sinc2(xfit, *popt)

        self.fit.setData(xfit, yfit)
        self.vline.setPos(float(popt[1])); self.vline.show()
        self.status.setText(f"Done. Center={float(popt[1]):.6f} MHz")

    @Slot(str)
    def on_failed(self, msg):
        self.status.setText(f"Error: {msg}")

    @Slot()
    def on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._worker = None
        self._thread = None

def main():
    app = QtWidgets.QApplication([])
    win = LiveScanApp()
    win.resize(1200, 700)
    win.show()
    app.exec()

if __name__ == "__main__":
    main()

# --- removed: Matplotlib live plotting + blocking scan code (was crashing AnyDesk)