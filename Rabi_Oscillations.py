import os
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import csv
import re
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.Qt import QtCore
import math  # <-- add

from scipy.optimize import curve_fit

from adriq.experiment import *
from adriq.Optomechanics import *
from adriq.Resonance_Log import read_resonance

# --- Qt binding compatibility (PyQt* vs PySide*) ---
Signal = getattr(QtCore, "pyqtSignal", None) or getattr(QtCore, "Signal", None)
Slot = getattr(QtCore, "pyqtSlot", None) or getattr(QtCore, "Slot", None)
if Signal is None or Slot is None:
    raise RuntimeError("Could not find Qt Signal/Slot (unsupported Qt binding via pyqtgraph.Qt).")


def rabi_model(t_us, A0, gamma_per_us, omega_rad_per_us):
    """y(t) = A0 * exp(-gamma t) * sin(omega t), with t in microseconds."""
    t = np.asarray(t_us, dtype=float)
    return float(A0) * np.exp(-float(gamma_per_us) * t) * np.sin(float(omega_rad_per_us) * t)


def _omega_guess_fft(t_us, y):
    t_us = np.asarray(t_us, dtype=float)
    y = np.asarray(y, dtype=float)
    if t_us.size < 5:
        return 0.2  # rad/us fallback
    dt = float(np.median(np.diff(t_us))) if t_us.size > 1 else 1.0
    if dt <= 0:
        return 0.2

    # FIX: this block was previously unreachable due to indentation under the dt<=0 return
    yc = y - np.mean(y)
    spec = np.abs(np.fft.rfft(yc))
    f = np.fft.rfftfreq(yc.size, d=dt)  # cycles per us
    if spec.size <= 2:
        return 0.2
    k = int(np.argmax(spec[1:]) + 1)  # ignore DC
    return float(2.0 * np.pi * f[k])  # rad/us


def fit_rabi_fixed_A0(t_us, p_dark, p_err, *, A0: float):
    t_us = np.asarray(t_us, dtype=float)
    p_dark = np.asarray(p_dark, dtype=float)
    p_err = np.asarray(p_err, dtype=float)

    # avoid zero sigma (curve_fit uses sigma as weights)
    sigma = np.where(p_err > 0, p_err, np.nanmedian(p_err[p_err > 0]) if np.any(p_err > 0) else 1.0)

    gamma0 = 0.001
    omega0 = _omega_guess_fft(t_us, p_dark)

    popt, pcov = curve_fit(
        lambda t, gamma_per_us, omega_rad_per_us: rabi_model(t, A0, gamma_per_us, omega_rad_per_us),
        t_us,
        p_dark,
        p0=[gamma0, omega0],
        sigma=sigma,
        absolute_sigma=True,
        bounds=([0.0, 0.0], [np.inf, np.inf]),
        maxfev=20000,
    )
    # popt = [gamma, omega]
    return popt, pcov


def fit_rabi_free_A0(t_us, p_dark, p_err):
    """Fit A0, gamma, omega in y(t)=A0*exp(-gamma t)*sin(omega t)."""
    t_us = np.asarray(t_us, dtype=float)
    p_dark = np.asarray(p_dark, dtype=float)
    p_err = np.asarray(p_err, dtype=float)

    sigma = np.where(p_err > 0, p_err, np.nanmedian(p_err[p_err > 0]) if np.any(p_err > 0) else 1.0)

    A0_0 = float(np.clip(np.nanmax(np.abs(p_dark)), 1e-6, 1.0))
    gamma0 = 0.001
    omega0 = _omega_guess_fft(t_us, p_dark)

    popt, pcov = curve_fit(
        lambda t, A0, gamma_per_us, omega_rad_per_us: rabi_model(t, A0, gamma_per_us, omega_rad_per_us),
        t_us,
        p_dark,
        p0=[A0_0, gamma0, omega0],
        sigma=sigma,
        absolute_sigma=True,
        bounds=([0.0, 0.0, 0.0], [1.0, np.inf, np.inf]),
        maxfev=20000,
    )
    # popt = [A0, gamma, omega]
    return popt, pcov


def next_rabi_filename(directory=None, pad=3):
    # Save into ./Rabi_Data by default
    if directory is None:
        directory = os.path.join(os.getcwd(), "Rabi_Data")

    os.makedirs(directory, exist_ok=True)

    pattern = re.compile(r"^rabi_(\d+)\.csv$")
    max_n = 0
    try:
        for fname in os.listdir(directory):
            m = pattern.match(fname)
            if m:
                max_n = max(max_n, int(m.group(1)))
    except FileNotFoundError:
        pass

    n = max_n + 1
    return os.path.join(directory, f"rabi_{n:0{pad}d}.csv")


def save_rabi_csv(path, meta: dict, t_us, p_dark, p_err, n_valid):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# meta", meta])
        w.writerow(["probe_time_us", "P_dark", "P_dark_stderr", "N_valid"])
        for t, p, e, nv in zip(t_us, p_dark, p_err, n_valid):
            w.writerow([float(t), float(p), float(e), int(nv)])

# ...existing code...

def make_probe_fn(*, amp: float, ton_us: float, shape: str = "square"):
    ton_us = float(max(0.0, ton_us))
    amp = float(amp)

    if shape == "square":
        return lambda t: amp if float(t) < ton_us else 0.0

    if shape == "gaussian":
        if ton_us <= 0.0:
            return lambda t: 0.0
        sigma = ton_us / (2.0 * math.sqrt(math.log(100.0)))
        center = ton_us / 2.0
        return lambda t: amp * np.exp(-((float(t) - center) ** 2) / (sigma ** 2))

    if shape == "adiabatic":
        if ton_us <= 0.0:
            return lambda t: 0.0
        edge_us = min(4.0, ton_us / 2.0)

        def _fn(t):
            x = float(t)
            if x < 0.0 or x > ton_us:
                return 0.0
            if x < edge_us:
                return amp * 0.5 * (1.0 - np.cos(np.pi * x / edge_us))
            if x > (ton_us - edge_us):
                y = ton_us - x
                return amp * 0.5 * (1.0 - np.cos(np.pi * y / edge_us))
            return amp

        return _fn

    raise ValueError(f"Unknown pulse shape: {shape!r}")


def prepare_rabi_context(*, params: dict):
    """
    Build/flash a single fixed-length sequence once.
    Probe section duration is fixed to max scan time (>=4 us).
    """
    pulse_sequencer = Pulse_Sequencer()
    dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

    RAM_BASE = 0.004
    ram_step = RAM_BASE * 50  # == 0.1 as 25 base steps

    exp_sequence = Experiment_Builder(
        dds_dict,
        pulse_sequencer,
        ram_step=ram_step,
        N_Cycles=int(params["n_cycles"]),
        external_trigger=bool(params["external_trigger"]),
    )

    exp_sequence.set_detunings(detuning_dict={
        "854 SP1": 0, "854 SP2": 0, "397b": 0, "397c": -18, "866 OP": 0,
        "850 RP": 0, "866 RP": 40, "729 t1": float(params["detuning_mhz"]), "729 t2": 0
    })

    exp_sequence.load_cooling(length=2000)
    exp_sequence.load_trapping()

    # --- Optical pumping mode selection (same as before) ---
    op_mode = str(params["op_mode"])
    if op_mode == "m_minus":
        exp_sequence.load_section("pump_to_stretch")  
        exp_sequence.load_section("pump_to_ground")
        exp_sequence.load_section("quench_metastables")  

    elif op_mode == "both":
        exp_sequence.load_section("quench_metastables")
    else:
        raise ValueError(f"Unknown op_mode: {op_mode!r}")

    exp_sequence.create_section(name="wait before probe", duration=2, dds_functions={}, pmt_gate_high=False)
    pulse_shape = str(params.get("pulse_shape", "square"))

    # FIXED-LENGTH probe section (duration = max scan time, but >=4 us)
    # Use ceil so decimal t_end_us is fully covered.
    probe_section_dur = int(max(4, math.ceil(float(params["t_end_us"]))))
    probe_section_dur = int(max(4, math.ceil(float(params["t_end_us"]))))
    if params.get("check_spam", True):
        exp_sequence.create_section(
            name="Probe",
            duration=probe_section_dur,
            dds_functions={
                "397b": make_probe_fn(amp=1, ton_us=0.0, shape=pulse_shape),
                "850 RP": make_probe_fn(amp=1, ton_us=0.0, shape=pulse_shape),
            },
            pmt_gate_high=True,
        )
    else:
        exp_sequence.create_section(
            name="Probe",
            duration=probe_section_dur,
            dds_functions={"729 t1": make_probe_fn(amp=float(params["probe_amplitude"]), ton_us=0.0, shape=pulse_shape)},
            pmt_gate_high=True,
        )

    exp_sequence.load_measurement()
    exp_sequence.build_ram_arrays()
    exp_sequence.flash()

    exp_runner = Experiment_Runner(
        dds_dict,
        pulse_sequencer,
        timeout=100,
        pmt_threshold=2000,
        expected_fluorescence=8000,
        pulse_expected_fluorescence=3200,
        sp_threshold=None,
        load_timeout=100,
        trigger_mode="ram",
        cavity_lock=False,
    )
    exp_runner.experiment_trap_depth = float(params["trap_depth"])
    exp_runner.measure_expected_fluorescence()  # Ensure expected fluorescence is set for correct probability calculation

    return dds_dict, pulse_sequencer, exp_sequence, exp_runner


def run_rabi_point_reuse(
    t_us: float,  # <-- was int
    *,
    exp_sequence: Experiment_Builder,
    exp_runner: Experiment_Runner,
    probe_amplitude: float,
    n_runs: int,
    pulse_shape: str = "square",
    check_spam: bool = False,
):
    # Only change the probe function (DDS '729 t1' becomes edited=True)
    if check_spam:
        exp_sequence.edit_section("Probe", {
            "397b": make_probe_fn(amp=1, ton_us=float(t_us), shape=pulse_shape),
            "850 RP": make_probe_fn(amp=1, ton_us=float(t_us), shape=pulse_shape),
        })
    else:
        exp_sequence.edit_section("Probe", {"729 t1": make_probe_fn(amp=float(probe_amplitude), ton_us=float(t_us), shape=pulse_shape)})

    # Rebuild RAM only for edited DDS and reflash only edited DDS
    exp_sequence.build_ram_arrays()
    exp_sequence.flash()

    exp_runner.clear_channels()
    exp_runner.start_experiment(N=int(n_runs))

    N_valid = int(exp_runner.N_Valid_Pulses or 0)
    N_bright = int(exp_runner.bright_counts or 0)
    P_bright = (N_bright / N_valid) if N_valid > 0 else 0.0
    P_dark = 1.0 - float(P_bright)
    err = float(np.sqrt(P_dark * (1.0 - P_dark) / N_valid)) if N_valid > 0 else float("nan")
    _, _, P_dark_corrected, stderr_corrected = correct_probabilities_with_spam(P_bright, err, P_dark, err)
    print(f"\033[91mUncorrected: P_dark={P_dark:.6f} ± {err:.6f} | Corrected: P_dark={P_dark_corrected:.6f} ± {stderr_corrected:.6f}\033[0m")
    return P_dark_corrected, stderr_corrected, int(N_valid)


class RabiWorker(QtCore.QObject):
    point = Signal(float, float, float, int)     # (t_us, P_dark, stderr, N_valid)
    scan_done = Signal(object, object, object, object)  # (t_us, P_dark, stderr, N_valid)
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
            t0 = float(p["t_start_us"])   # <-- was int
            t1 = float(p["t_end_us"])     # <-- was int
            dt = float(p["t_step_us"])    # <-- was int
            if dt <= 0:
                raise ValueError("Resolution/step must be > 0.")
            if t1 < t0:
                raise ValueError("End time must be >= start time.")

            # Build once
            _, _, exp_sequence, exp_runner = prepare_rabi_context(params=p)

            # Robust float stepping without accumulation drift
            n_steps = int(np.floor((t1 - t0) / dt)) + 1
            times = (t0 + dt * np.arange(max(n_steps, 0), dtype=float)).tolist()
            if not times:
                times = [t0]
            if t1 > times[-1] and not np.isclose(times[-1], t1, rtol=0, atol=1e-12):
                times.append(t1)

            out_t, out_p, out_e, out_n = [], [], [], []

            self.status.emit("Scanning...")
            for t_us in times:
                if self._should_stop():
                    self.status.emit("Stopped.")
                    self.finished.emit()
                    return

                P_dark, err, N_valid = run_rabi_point_reuse(
                    float(t_us),
                    exp_sequence=exp_sequence,
                    exp_runner=exp_runner,
                    probe_amplitude=p["probe_amplitude"],
                    n_runs=p["n_runs"],
                    pulse_shape=str(p.get("pulse_shape", "square")),
                    check_spam=p.get("check_spam", False)
                )

                out_t.append(float(t_us))
                out_p.append(float(P_dark))
                out_e.append(float(err))
                out_n.append(int(N_valid))
                self.point.emit(float(t_us), float(P_dark), float(err), int(N_valid))

            self.status.emit("Done.")
            self.scan_done.emit(np.asarray(out_t), np.asarray(out_p), np.asarray(out_e), np.asarray(out_n))
            self.finished.emit()

        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            self.finished.emit()

class RabiApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rabi oscillations (pyqtgraph)")

        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QGridLayout(w)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form, 0, 0, 1, 1)

        self.external_trigger = QtWidgets.QCheckBox("External trigger")
        self.external_trigger.setChecked(False)  # CHANGED: default off
        self.check_spam = QtWidgets.QCheckBox("Check SPAM")
        self.check_spam.setChecked(False)

        self.detuning = QtWidgets.QDoubleSpinBox()
        self.detuning.setDecimals(6)
        self.detuning.setRange(-1e6, 1e6)
        self.detuning.setValue(-23.305)

        self.load_res_btn = QtWidgets.QPushButton("Load resonance…")
        self.load_res_btn.setToolTip("Select a resonance log CSV and load its most recent 'Resonant Detuning' value")


        self.power = QtWidgets.QDoubleSpinBox()
        self.power.setDecimals(4)
        self.power.setRange(0.0, 1.0)
        self.power.setSingleStep(0.01)
        self.power.setValue(1.0)

        # Use doubles so probe times can be decimals (µs)
        self.t_start = QtWidgets.QDoubleSpinBox()
        self.t_start.setDecimals(3)
        self.t_start.setRange(0.0, 10_000_000.0)
        self.t_start.setSingleStep(0.1)
        self.t_start.setValue(0.0)

        self.t_end = QtWidgets.QDoubleSpinBox()
        self.t_end.setDecimals(3)
        self.t_end.setRange(0.0, 10_000_000.0)
        self.t_end.setSingleStep(0.1)
        self.t_end.setValue(50.0)

        self.t_step = QtWidgets.QDoubleSpinBox()
        self.t_step.setDecimals(3)
        self.t_step.setRange(0.001, 10_000_000.0)
        self.t_step.setSingleStep(0.1)
        self.t_step.setValue(2.0)

        self.trap_depth = QtWidgets.QDoubleSpinBox()
        self.trap_depth.setDecimals(3)
        self.trap_depth.setRange(0.0, 1.0)
        self.trap_depth.setSingleStep(0.01)
        self.trap_depth.setValue(0.8)

        self.n_runs = QtWidgets.QSpinBox(); self.n_runs.setRange(1, 999); self.n_runs.setValue(3)

        # NEW: N cycles (builder)
        self.n_cycles = QtWidgets.QSpinBox()
        self.n_cycles.setRange(1, 10_000_000)
        self.n_cycles.setValue(500)

        # NEW: optical pumping selector
        self.op_mode = QtWidgets.QComboBox()
        self.op_mode.addItem("Pump to S1/2 m = -1/2", "m_minus")
        self.op_mode.addItem("Pump to both S1/2 m = ±1/2", "both")
        self.op_mode.setCurrentIndex(0)

        self.pulse_shape = QtWidgets.QComboBox()
        self.pulse_shape.addItem("Square pulse", "square")
        self.pulse_shape.addItem("Gaussian pulse", "gaussian")
        self.pulse_shape.addItem("Adiabatic ramp", "adiabatic")
        self.pulse_shape.setCurrentIndex(0)

        self.output_dir = QtWidgets.QLineEdit(os.getcwd())
        self.output_dir = QtWidgets.QLineEdit(os.path.join(os.getcwd(), "Rabi_Data"))

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False)
        self.save_btn = QtWidgets.QPushButton("Save CSV"); self.save_btn.setEnabled(False)
        self.fit_btn = QtWidgets.QPushButton("Fit  A0·e^-γt sin(Ωt)"); self.fit_btn.setEnabled(False)
        self.status = QtWidgets.QLabel("Idle.")

        # NEW: A0 input (state prep efficiency / fixed amplitude for fit)
        self.A0 = QtWidgets.QDoubleSpinBox()
        self.A0.setDecimals(4)
        self.A0.setRange(0.0, 1.0)
        self.A0.setSingleStep(0.01)
        self.A0.setValue(1.0)

        form.addRow("", self.external_trigger)
        form.addRow("", self.check_spam)
        form.addRow("Detuning [MHz]", self.detuning)
        form.addRow("", self.load_res_btn)  # NEW
        form.addRow("Power/amplitude [0..1]", self.power)
        form.addRow("Start time [µs]", self.t_start)
        form.addRow("End time [µs]", self.t_end)
        form.addRow("Resolution/step [µs]", self.t_step)
        form.addRow("Trap depth", self.trap_depth)
        form.addRow("N runs (per point)", self.n_runs)
        form.addRow("N cycles (builder)", self.n_cycles)      # NEW
        form.addRow("Optical pumping", self.op_mode)          # NEW
        form.addRow("Probe pulse shape", self.pulse_shape)
        form.addRow("Output dir", self.output_dir)
        form.addRow("", self.start_btn)
        form.addRow("", self.stop_btn)
        form.addRow("", self.save_btn)

        # NEW: A0 next to fit button (same row)
        _fit_row = QtWidgets.QWidget()
        _fit_row_layout = QtWidgets.QHBoxLayout(_fit_row)
        _fit_row_layout.setContentsMargins(0, 0, 0, 0)
        _fit_row_layout.addWidget(QtWidgets.QLabel("A0"))
        _fit_row_layout.addWidget(self.A0)
        _fit_row_layout.addWidget(self.fit_btn)
        form.addRow("", _fit_row)

        form.addRow("Status", self.status)

        pg.setConfigOptions(antialias=True)
        self.plots = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plots, 0, 1, 1, 1)

        self.p = self.plots.addPlot(title="Rabi oscillations")
        self.p.setLabel("bottom", "Probe time (µs)")
        self.p.setLabel("left", "P_dark")
        self.p.showGrid(x=True, y=True)
        self.curve = self.p.plot(
            [],
            [],
            pen=pg.mkPen(color=(0, 150, 255), width=1),
            symbol="o",
            symbolSize=6,
        )

        self.err_item = pg.ErrorBarItem(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=0.6)
        self.p.addItem(self.err_item)

        self._err_beam = 0.0  # NEW: x-width of error bars in µs

        self.fit_curve = self.p.plot([], [], pen=pg.mkPen(width=2))

        self._thread = None
        self._worker = None

        self._t = []
        self._p_dark = []
        self._p_err = []
        self._n_valid = []
        self._last_params = None

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.save_btn.clicked.connect(self.save_csv)
        self.fit_btn.clicked.connect(self.fit)
        self.load_res_btn.clicked.connect(self.load_resonance_dialog)

    def _params(self):
        return dict(
            external_trigger=bool(self.external_trigger.isChecked()),
            detuning_mhz=float(self.detuning.value()),
            probe_amplitude=float(self.power.value()),
            t_start_us=float(self.t_start.value()),  # <-- was int
            t_end_us=float(self.t_end.value()),      # <-- was int
            t_step_us=float(self.t_step.value()),    # <-- was int
            trap_depth=float(self.trap_depth.value()),
            n_runs=int(self.n_runs.value()),
            n_cycles=int(self.n_cycles.value()),
            op_mode=str(self.op_mode.currentData() or "m_minus"),
            pulse_shape=str(self.pulse_shape.currentData() or "square"),
            output_dir=str(self.output_dir.text()).strip() or os.getcwd(),
            check_spam=bool(self.check_spam.isChecked())
        )

    @Slot()
    def start(self):
        self._t, self._p_dark, self._p_err, self._n_valid = [], [], [], []
        self.curve.setData([], [])
        self.err_item.setData(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=self._err_beam)
        self.fit_curve.setData([], [])
        self.status.setText("Starting...")
        self.save_btn.setEnabled(False)
        self.fit_btn.setEnabled(False)

        params = self._params()
        self._last_params = params
        self._err_beam = 0.6 * float(params.get("t_step_us", 2.0))  # NEW: scale by resolution

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._thread = QtCore.QThread(self)
        self._worker = RabiWorker(params)
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

    @Slot(float, float, float, int)
    def on_point(self, t_us, p_dark, stderr, n_valid):
        self._t.append(float(t_us))
        self._p_dark.append(float(p_dark))
        self._p_err.append(float(stderr))
        self._n_valid.append(int(n_valid))

        x = np.asarray(self._t, dtype=float)
        y = np.asarray(self._p_dark, dtype=float)
        e = np.asarray(self._p_err, dtype=float)

        self.curve.setData(x, y)

        # ErrorBarItem wants top/bottom lengths (not absolute y)
        top = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
        self.err_item.setData(x=x, y=y, top=top, bottom=top, beam=self._err_beam)

        self.p.enableAutoRange(axis="y", enable=True)

    @Slot(object, object, object, object)
    def on_scan_done(self, t_us, p_dark, p_err, n_valid):
        self.save_btn.setEnabled(True)
        self.fit_btn.setEnabled(True)

    @Slot(str)
    def on_failed(self, msg):
        self.status.setText(f"Error: {msg}")

    @Slot()
    def stop(self):
        # CHANGED: direct flag + interrupt (no queued invoke; worker may be busy).
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
        self.stop_btn.setEnabled(False)
        self._worker = None
        self._thread = None

    @Slot()
    def save_csv(self):
        if not self._t:
            return
        params = dict(self._last_params or {})
        out_dir = params.get("output_dir") or os.getcwd()
        path = next_rabi_filename(out_dir)
        meta = {
            "detuning_mhz": params.get("detuning_mhz"),
            "probe_amplitude": params.get("probe_amplitude"),
            "external_trigger": params.get("external_trigger"),
            "trap_depth": params.get("trap_depth"),
            "t_start_us": params.get("t_start_us"),
            "t_end_us": params.get("t_end_us"),
            "t_step_us": params.get("t_step_us"),
            "n_runs": params.get("n_runs"),
            "n_cycles": params.get("n_cycles"),         # NEW
            "op_mode": params.get("op_mode"),           # NEW
            "pulse_shape": params.get("pulse_shape"),
        }
        save_rabi_csv(path, meta, self._t, self._p_dark, self._p_err, self._n_valid)
        self.status.setText(f"Saved: {path}")

    @Slot()
    def fit(self):
        if len(self._t) < 5:
            self.status.setText("Not enough points to fit.")
            return
        try:
            A0_in = float(self.A0.value())

            t = np.asarray(self._t, dtype=float)
            y = np.asarray(self._p_dark, dtype=float)
            e = np.asarray(self._p_err, dtype=float)

            if A0_in == 0.0:
                (A0_fit, gamma, omega), _ = fit_rabi_free_A0(t, y, e)
                A0_used = float(A0_fit)
                label = f"Fit (A0 fitted={A0_used:.3f})"
            else:
                A0_used = float(A0_in)
                (gamma, omega), _ = fit_rabi_fixed_A0(t, y, e, A0=A0_used)
                label = f"Fit (fixed A0={A0_used:.3f})"

            tfit = np.linspace(float(np.min(t)), float(np.max(t)), 1200)
            yfit = rabi_model(tfit, A0_used, gamma, omega)
            self.fit_curve.setData(tfit, yfit)

            self.status.setText(f"{label}: γ={gamma:.4g}MHz, Ω={omega/(2*np.pi):.4g} (2π) MHz)")
        except Exception as ex:
            self.status.setText(f"Fit error: {type(ex).__name__}: {ex}")

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
                # Qt6 fallback (if needed)
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
            self.detuning.setValue(res_mhz)
            self.status.setText(f"Loaded resonance: {res_mhz:.6f} MHz")

        except Exception as e:
            self.status.setText(f"Load resonance failed: {type(e).__name__}: {e}")


def main():
    app = QtWidgets.QApplication([])
    win = RabiApp()
    win.resize(1200, 700)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
