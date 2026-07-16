import os  # moved up so env var is set before importing pyqtgraph

# Force pyqtgraph to use PyQt5 (prevents PySide/PyQt mismatches)
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import math
import csv
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.optimize import curve_fit

from adriq.experiment import (
    correct_probabilities_with_spam,
    Experiment_Builder,
    Experiment_Runner,
    Pulse_Sequencer,
    load_dds_dict,
)
from adriq.Resonance_Log import read_resonance, write_most_recent_resonance


# --- Qt binding compatibility (PyQt* vs PySide*) ---
Signal = getattr(QtCore, "pyqtSignal", None) or getattr(QtCore, "Signal", None)
Slot = getattr(QtCore, "pyqtSlot", None) or getattr(QtCore, "Slot", None)
if Signal is None or Slot is None:
    raise RuntimeError("Could not find Qt Signal/Slot (unsupported Qt binding via pyqtgraph.Qt).")


def parse_csv_list(text: str):
    """Parse comma/space separated list of numbers (floats)"""
    parts = [p.strip() for p in text.replace(";", ",").replace(" ", ",").split(",")]
    vals = []
    for p in parts:
        if not p:
            continue
        vals.append(float(p))
    return vals


def ramsey_detuning_cos_model(detuning_mhz, y0, amp, phi_rad, *, t_precess_us: float):
    """Simple Ramsey fringes vs detuning for fixed free-precession time.

    phase = 2*pi*detuning(MHz)*t_precess(us) + phi.

    This is intentionally minimal (good enough to estimate the nearest fringe maximum).
    """
    detuning_mhz = np.asarray(detuning_mhz, dtype=float)
    tp = float(t_precess_us)
    return float(y0) + float(amp) * np.cos(2.0 * np.pi * detuning_mhz * tp + float(phi_rad))


def fit_ramsey_detuning_cos(detuning_mhz, p_dark, p_err, *, t_precess_us: float):
    detuning_mhz = np.asarray(detuning_mhz, dtype=float)
    p_dark = np.asarray(p_dark, dtype=float)
    p_err = np.asarray(p_err, dtype=float)

    # Avoid zero/NaN sigma
    if np.any(np.isfinite(p_err) & (p_err > 0)):
        fallback = float(np.nanmedian(p_err[(p_err > 0) & np.isfinite(p_err)]))
    else:
        fallback = 1.0
    sigma = np.where((p_err > 0) & np.isfinite(p_err), p_err, fallback)

    y0_0 = float(np.nanmean(p_dark)) if np.isfinite(np.nanmean(p_dark)) else 0.5
    amp0 = float(0.5 * (np.nanmax(p_dark) - np.nanmin(p_dark))) if np.isfinite(np.nanmax(p_dark)) else 0.1
    phi0 = 0.0

    def f(x, y0, amp, phi_rad):
        return ramsey_detuning_cos_model(x, y0, amp, phi_rad, t_precess_us=t_precess_us)

    popt, pcov = curve_fit(
        f,
        detuning_mhz,
        p_dark,
        sigma=sigma,
        p0=[y0_0, amp0, phi0],
        absolute_sigma=True,
        maxfev=20000,
    )

    perr = np.sqrt(np.diag(pcov)) if pcov.size else np.array([np.nan, np.nan, np.nan])
    return popt, perr, pcov


def nearest_fringe_maximum(center_guess_mhz: float, *, amp: float, phi_rad: float, t_precess_us: float):
    """Compute the detuning (MHz) of the cosine maximum closest to center_guess_mhz."""
    tp = float(t_precess_us)
    if not (tp > 0):
        return float("nan")

    # Maxima when 2*pi*tp*x + phi = 2*pi*n
    x0 = (-float(phi_rad)) / (2.0 * np.pi * tp)
    period = 1.0 / tp

    if not np.isfinite(x0) or not np.isfinite(period) or period <= 0:
        return float("nan")

    n = int(np.round((float(center_guess_mhz) - x0) / period))
    return float(x0 + n * period)


def next_spectrum_filename(directory=".", pad=3):
    import re

    pattern = re.compile(r"^ramsey_detuning_(\d+)\.csv$")
    max_n = 0
    try:
        for fname in os.listdir(directory):
            m = pattern.match(fname)
            if m:
                max_n = max(max_n, int(m.group(1)))
    except FileNotFoundError:
        pass
    n = max_n + 1
    return os.path.join(directory, f"ramsey_detuning_{n:0{pad}d}.csv")


def save_spectrum(detunings, values, errs=None, n_valid=None, directory="."):
    spectra_dir = os.path.join(str(directory), "Ramsey_Spectra")
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


def build_ramsey_sequence(
    *,
    dds_dict,
    pulse_sequencer,
    detuning_mhz: float,
    power: float,
    pulse_time_us: float,
    t_precess_us: float,
    phase2_deg: float,
    n_cycles: int,
    external_trigger: bool,
    op_mode: str,
):
    """Build/flash a Ramsey block sequence (two pulses separated by free precession)."""

    RAM_STEP = 0.2
    COOLING_US = 2000

    exp_sequence = Experiment_Builder(
        dds_dict,
        pulse_sequencer,
        ram_step=float(RAM_STEP),
        N_Cycles=int(n_cycles),
        external_trigger=bool(external_trigger),
    )

    exp_sequence.set_detunings(detuning_dict={
        "729 Temp1": float(detuning_mhz)
    })

    exp_sequence.load_cooling(length=int(COOLING_US))
    exp_sequence.load_trapping()

    # Optical pumping selection (match Spectroscopy_App pattern)
    if op_mode == "m_minus":
        exp_sequence.load_section("pump_to_stretch")
        exp_sequence.load_section("pump_to_ground")
        exp_sequence.load_section("quench_metastables")
    elif op_mode == "both":
        exp_sequence.load_section("quench_metastables")
    else:
        raise ValueError(f"Unknown op_mode: {op_mode!r}")

    ton = float(pulse_time_us)
    tp = float(t_precess_us)
    a = float(power)
    phase2 = float(phase2_deg)

    duration_us = int(math.ceil(2.0 * ton + tp))

    def dds_fn(t, a=a, ton=ton, tp=tp):
        tt = float(t)
        if tt < ton:
            return a
        if tt < ton + tp:
            return 0.0
        if tt < ton + tp + ton:
            return a
        return 0.0

    def phase_fn(t, phase2=phase2, ton=ton, tp=tp):
        tt = float(t)
        # Apply phase only during 2nd pulse
        return float(phase2) if (ton + tp) <= tt < (ton + tp + ton) else 0.0

    exp_sequence.create_section(
        name="RamseyBlock",
        duration=duration_us,
        dds_functions={"729 Temp1": dds_fn},
        phase_functions={"729 Temp1": phase_fn},
        pmt_gate_high=True,
    )

    exp_sequence.load_measurement()
    exp_sequence.build_ram_arrays()
    exp_sequence.flash()

    return exp_sequence


def dark_probability(exp_sequence, exp_runner, *, detuning_mhz: float, trap_depth: float, n_runs: int, use_spam_correction: bool):
    exp_sequence.edit_detunings(detuning_dict={"729 Temp1": float(detuning_mhz)})
    exp_sequence.build_ram_arrays()
    exp_sequence.flash()

    exp_runner.experiment_trap_depth = float(trap_depth)

    exp_runner.clear_channels()
    exp_runner.start_experiment(N=int(n_runs))

    n_bright = int(getattr(exp_runner, "bright_counts", 0) or 0)
    n_valid = int(getattr(exp_runner, "N_Valid_Pulses", 0) or 0)

    if n_valid > 0:
        p_bright = n_bright / n_valid
        p_dark = 1.0 - float(p_bright)
        err = float(np.sqrt(p_dark * (1.0 - p_dark) / n_valid))
    else:
        p_bright = 0.0
        p_dark = 0.0
        err = float("nan")

    if use_spam_correction:
        _, _, p_dark_corr, err_corr = correct_probabilities_with_spam(p_bright, err, p_dark, err)
        return float(p_dark_corr), float(err_corr), int(n_valid)

    return float(p_dark), float(err), int(n_valid)


class ScanWorker(QtCore.QObject):
    point = Signal(float, float, float, int)  # (detuning_MHz, P_dark, stderr, N_valid)
    scan_done = Signal(object, object, object, object, object)  # (dets, vals, errs, popt, meta)
    status = Signal(str)
    finished = Signal()
    failed = Signal(str)
    background = Signal(float)  # background P_dark

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

            # Hardware context
            pulse_sequencer = Pulse_Sequencer()
            dds_dict = load_dds_dict(
                "ram",
                r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg",
            )

            TIMEOUT_S = 100.0
            PMT_THRESHOLD = 2000
            EXPECTED_FLUOR = 8000
            PULSE_EXPECTED_FLUOR = 4200

            exp_runner = Experiment_Runner(
                dds_dict,
                pulse_sequencer,
                timeout=float(TIMEOUT_S),
                pmt_threshold=int(PMT_THRESHOLD),
                expected_fluorescence=int(EXPECTED_FLUOR),
                pulse_expected_fluorescence=int(PULSE_EXPECTED_FLUOR),
                trigger_mode="ram",
            )

            self.status.emit("Measuring expected fluorescence...")
            try:
                exp_runner.measure_expected_fluorescence()
            except Exception:
                pass

            center = float(p["center_mhz"])
            span = float(p["span_mhz"])
            step = float(p["step_mhz"])
            dets = np.arange(center - span, center + span + 1e-12, step)

            # Build a single Ramsey sequence (detuning edited per point)
            self.status.emit("Building Ramsey sequence...")
            exp_sequence = build_ramsey_sequence(
                dds_dict=dds_dict,
                pulse_sequencer=pulse_sequencer,
                detuning_mhz=center,
                power=float(p["power"]),
                pulse_time_us=float(p["pulse_time_us"]),
                t_precess_us=float(p["t_precess_us"]),
                phase2_deg=float(p["phase2_deg"]),
                n_cycles=int(p["n_cycles"]),
                external_trigger=bool(p["external_trigger"]),
                op_mode=str(p["op_mode"]),
            )

            use_spam = bool(p.get("use_spam_correction", True))

            bg_y0 = None
            if bool(p.get("measure_background", True)):
                if self._should_stop():
                    self.status.emit("Stopped.")
                    self.finished.emit()
                    return

                # Background measurement: power=0, same timing
                self.status.emit("Measuring background (729 off)...")
                exp_sequence_bg = build_ramsey_sequence(
                    dds_dict=dds_dict,
                    pulse_sequencer=pulse_sequencer,
                    detuning_mhz=center,
                    power=0.0,
                    pulse_time_us=float(p["pulse_time_us"]),
                    t_precess_us=float(p["t_precess_us"]),
                    phase2_deg=float(p["phase2_deg"]),
                    n_cycles=int(p["n_cycles"]),
                    external_trigger=bool(p["external_trigger"]),
                    op_mode=str(p["op_mode"]),
                )

                bg_runs = max(1, int(p["n_runs"]) * 2)
                bg_y0, _bg_e, _bg_nv = dark_probability(
                    exp_sequence_bg,
                    exp_runner,
                    detuning_mhz=center,
                    trap_depth=float(p["trap_depth"]),
                    n_runs=bg_runs,
                    use_spam_correction=use_spam,
                )
                self.background.emit(float(bg_y0))
                self.status.emit(f"Background: P_dark={float(bg_y0):.6f} (N={bg_runs})")
            else:
                self.status.emit("Background measurement: OFF")

            vals, errs, nvs = [], [], []
            self.status.emit("Scanning detuning...")

            for d in dets:
                if self._should_stop():
                    self.status.emit("Stopped.")
                    self.finished.emit()
                    return

                y, e, nv = dark_probability(
                    exp_sequence,
                    exp_runner,
                    detuning_mhz=float(d),
                    trap_depth=float(p["trap_depth"]),
                    n_runs=int(p["n_runs"]),
                    use_spam_correction=use_spam,
                )

                vals.append(float(y))
                errs.append(float(e))
                nvs.append(int(nv))
                self.point.emit(float(d), float(y), float(e), int(nv))

            dets = np.asarray(dets, dtype=float)
            vals = np.asarray(vals, dtype=float)
            errs_arr = np.asarray(errs, dtype=float)

            if bool(p.get("save_csv", True)):
                save_spectrum(dets, vals, errs=errs_arr, n_valid=np.asarray(nvs, dtype=int), directory=str(p["output_dir"]))

            popt = None
            try:
                popt, perr, _pcov = fit_ramsey_detuning_cos(dets, vals, errs_arr, t_precess_us=float(p["t_precess_us"]))
            except Exception:
                popt = None

            meta = {
                "center_guess_mhz": center,
                "t_precess_us": float(p["t_precess_us"]),
                "phase2_deg": float(p["phase2_deg"]),
            }

            self.scan_done.emit(dets, vals, errs_arr, popt, meta)
            self.finished.emit()

        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            self.finished.emit()


class RamseyDetuningScanApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ramsey detuning scan (pyqtgraph)")
        self._last_fitted_center_mhz = None

        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QGridLayout(w)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form, 0, 0, 1, 1)

        # --- scan params (match Spectroscopy_App)
        self.center = QtWidgets.QDoubleSpinBox(); self.center.setDecimals(6); self.center.setRange(-1e6, 1e6); self.center.setValue(0.0)
        self.load_res_btn = QtWidgets.QPushButton("Load resonance…")
        self.load_res_btn.setToolTip("Select a resonance log CSV and load its most recent 'Resonant Detuning' value")

        self.span = QtWidgets.QDoubleSpinBox(); self.span.setDecimals(6); self.span.setRange(0.0, 1e6); self.span.setValue(0.2)
        self.step = QtWidgets.QDoubleSpinBox(); self.step.setDecimals(6); self.step.setRange(1e-6, 1e3); self.step.setValue(0.01)

        self.op_mode = QtWidgets.QComboBox()
        self.op_mode.addItem("Pump to S1/2 m = -1/2", "m_minus")
        self.op_mode.addItem("Pump to both S1/2 m = ±1/2", "both")
        self.op_mode.setCurrentIndex(0)

        # --- Ramsey params
        self.power = QtWidgets.QDoubleSpinBox(); self.power.setDecimals(4); self.power.setRange(0.0, 1.0); self.power.setSingleStep(0.01); self.power.setValue(1.0)
        self.pulse_time = QtWidgets.QDoubleSpinBox(); self.pulse_time.setDecimals(3); self.pulse_time.setRange(0.0, 10_000_000.0); self.pulse_time.setSingleStep(0.1); self.pulse_time.setValue(3.2)
        self.t_precess = QtWidgets.QDoubleSpinBox(); self.t_precess.setDecimals(3); self.t_precess.setRange(0.0, 10_000_000.0); self.t_precess.setSingleStep(1.0); self.t_precess.setValue(20.0)
        self.phase2 = QtWidgets.QDoubleSpinBox(); self.phase2.setDecimals(2); self.phase2.setRange(-3600.0, 3600.0); self.phase2.setSingleStep(5.0); self.phase2.setValue(0.0)

        self.external_trigger = QtWidgets.QCheckBox("External trigger")
        self.external_trigger.setChecked(False)

        self.n_runs = QtWidgets.QSpinBox(); self.n_runs.setRange(1, 999); self.n_runs.setValue(3)
        self.n_cycles = QtWidgets.QSpinBox(); self.n_cycles.setRange(1, 10_000_000); self.n_cycles.setValue(500)

        self.trap_depth = QtWidgets.QDoubleSpinBox(); self.trap_depth.setDecimals(3); self.trap_depth.setRange(0.0, 1.0); self.trap_depth.setSingleStep(0.01); self.trap_depth.setValue(0.8)

        self.output_dir = QtWidgets.QLineEdit(os.getcwd())
        self.save_csv = QtWidgets.QCheckBox("Save CSV"); self.save_csv.setChecked(True)

        self.measure_background = QtWidgets.QCheckBox("Measure background (729 off)")
        self.measure_background.setChecked(True)

        self.use_spam_correction = QtWidgets.QCheckBox("Apply SPAM correction")
        self.use_spam_correction.setChecked(True)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False)
        self.write_res_btn = QtWidgets.QPushButton("Write resonance…")

        self.status = QtWidgets.QLabel("Idle.")

        form.addRow("Center [MHz]", self.center)
        form.addRow("", self.load_res_btn)
        form.addRow("Span [MHz]", self.span)
        form.addRow("Resolution/step [MHz]", self.step)
        form.addRow("Optical pumping", self.op_mode)
        form.addRow("Power/amplitude [0..1]", self.power)
        form.addRow("Pulse time [µs]", self.pulse_time)
        form.addRow("t_precess [µs]", self.t_precess)
        form.addRow("Fixed phase2 [deg]", self.phase2)
        form.addRow("", self.external_trigger)
        form.addRow("N runs (per point)", self.n_runs)
        form.addRow("N cycles (builder)", self.n_cycles)
        form.addRow("Trap depth", self.trap_depth)
        form.addRow("Output dir", self.output_dir)
        form.addRow("", self.save_csv)
        form.addRow("", self.measure_background)
        form.addRow("", self.use_spam_correction)
        form.addRow("", self.start_btn)
        form.addRow("", self.stop_btn)
        form.addRow("", self.write_res_btn)
        form.addRow("Status", self.status)

        # --- plot
        pg.setConfigOptions(antialias=True)
        self.plots = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plots, 0, 1, 1, 1)

        self.p = self.plots.addPlot(title="Ramsey detuning scan")
        self.p.setLabel("bottom", "Detuning (MHz)")
        self.p.setLabel("left", "P_dark")
        self.p.showGrid(x=True, y=True)

        # Show markers and connect them with a line
        self.curve = self.p.plot([], [], pen=pg.mkPen(width=1), symbol="o", symbolSize=5)
        self.err_item = pg.ErrorBarItem(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=0.0)
        self.p.addItem(self.err_item)

        self.fit_curve = self.p.plot([], [], pen=pg.mkPen(width=2))
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(style=QtCore.Qt.PenStyle.DashLine))
        self.p.addItem(self.vline); self.vline.hide()

        self.bgline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("r", width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.p.addItem(self.bgline); self.bgline.hide()

        # --- runtime
        self._thread = None
        self._worker = None
        self._xs = []
        self._ys = []
        self._es = []
        self._err_beam = 0.0

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.load_res_btn.clicked.connect(self.load_resonance_dialog)
        self.write_res_btn.clicked.connect(self.write_resonance_dialog)

    def _params(self):
        step = float(self.step.value())
        return dict(
            center_mhz=float(self.center.value()),
            span_mhz=float(self.span.value()),
            step_mhz=step,
            op_mode=str(self.op_mode.currentData() or "m_minus"),
            power=float(self.power.value()),
            pulse_time_us=float(self.pulse_time.value()),
            t_precess_us=float(self.t_precess.value()),
            phase2_deg=float(self.phase2.value()),
            external_trigger=bool(self.external_trigger.isChecked()),
            n_runs=int(self.n_runs.value()),
            n_cycles=int(self.n_cycles.value()),
            trap_depth=float(self.trap_depth.value()),
            output_dir=str(self.output_dir.text()).strip() or os.getcwd(),
            save_csv=bool(self.save_csv.isChecked()),
            measure_background=bool(self.measure_background.isChecked()),
            use_spam_correction=bool(self.use_spam_correction.isChecked()),
        )

    def _update_plot(self):
        x = np.asarray(self._xs, dtype=float)
        y = np.asarray(self._ys, dtype=float)
        e = np.asarray(self._es, dtype=float)

        self.curve.setData(x, y)
        top = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
        self.err_item.setData(x=x, y=y, top=top, bottom=top, beam=self._err_beam)

        if x.size:
            self.p.enableAutoRange(axis="y", enable=True)
            xmin = float(np.min(x)); xmax = float(np.max(x))
            if xmin == xmax:
                dx = max(float(self.step.value()) * 5.0, 1e-6)
                xmin -= dx; xmax += dx
            else:
                pad = 0.05 * (xmax - xmin)
                xmin -= pad; xmax += pad
            self.p.setXRange(xmin, xmax, padding=0.0)

    @Slot()
    def start(self):
        try:
            params = self._params()
            if params["step_mhz"] <= 0:
                raise ValueError("step must be > 0")
            if params["t_precess_us"] < 0:
                raise ValueError("t_precess must be >= 0")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        self._xs, self._ys, self._es = [], [], []
        self._last_fitted_center_mhz = None

        self.curve.setData([], [])
        self.fit_curve.setData([], [])
        self.vline.hide()
        self.bgline.hide()

        self._err_beam = 0.6 * float(params.get("step_mhz", 0.02))
        self.err_item.setData(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=self._err_beam)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("Starting...")

        self._thread = QtCore.QThread(self)
        self._worker = ScanWorker(params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.point.connect(self.on_point)
        self._worker.scan_done.connect(self.on_scan_done)
        self._worker.background.connect(self.on_background)
        self._worker.status.connect(self.status.setText)
        self._worker.failed.connect(self.on_failed)
        self._worker.finished.connect(self.on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    @Slot(float, float, float, int)
    def on_point(self, detuning_mhz, p_dark, stderr, n_valid):
        self._xs.append(float(detuning_mhz))
        self._ys.append(float(p_dark))
        self._es.append(float(stderr))
        self._update_plot()

    @Slot(object, object, object, object, object)
    def on_scan_done(self, dets, vals, errs, popt, meta):
        dets = np.asarray(dets, dtype=float)
        vals = np.asarray(vals, dtype=float)
        errs = np.asarray(errs, dtype=float)

        if popt is None:
            # Fall back to argmax (nearest peak)
            if dets.size:
                i = int(np.nanargmax(vals))
                self._last_fitted_center_mhz = float(dets[i])
                self.vline.setPos(self._last_fitted_center_mhz)
                self.vline.show()
                self.status.setText(f"Done. Peak≈{self._last_fitted_center_mhz:.6f} MHz (argmax)")
            return

        y0, amp, phi = map(float, popt)
        tp = float(meta.get("t_precess_us", float(self.t_precess.value())))
        center_guess = float(meta.get("center_guess_mhz", float(self.center.value())))

        xfit = np.linspace(float(np.min(dets)), float(np.max(dets)), 1200)
        yfit = ramsey_detuning_cos_model(xfit, y0, amp, phi, t_precess_us=tp)
        self.fit_curve.setData(xfit, yfit)

        x_peak = nearest_fringe_maximum(center_guess, amp=amp, phi_rad=phi, t_precess_us=tp)
        if np.isfinite(x_peak):
            self._last_fitted_center_mhz = float(x_peak)
            self.vline.setPos(float(x_peak))
            self.vline.show()
            self.status.setText(f"Done. Peak≈{float(x_peak):.6f} MHz")
        else:
            self.status.setText("Done.")

    @Slot(float)
    def on_background(self, level: float):
        self.bgline.setPos(float(level))
        self.bgline.show()

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
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._worker = None
        self._thread = None
        if self.status.text().startswith("Starting") or self.status.text().startswith("Scanning"):
            self.status.setText("Idle.")

    @Slot()
    def write_resonance_dialog(self):
        # --- Step 1: choose source (fitted vs manual) ---
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Write resonance")
        msg.setText("Write which resonance value to the log?")

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
        try:
            self.raise_()
            self.activateWindow()
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

        default_dir = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\Resonances"
        start_dir = default_dir if os.path.isdir(default_dir) else os.getcwd()

        try:
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
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select resonance log CSV (append/create)",
                start_dir,
                "CSV files (*.csv);;All files (*.*)",
                options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
            )

        if not path:
            return

        try:
            write_most_recent_resonance(path, value_mhz)
            self.status.setText(f"Wrote {value_mhz:.6f} MHz -> {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Write failed", f"{type(e).__name__}: {e}")

    @Slot()
    def load_resonance_dialog(self):
        try:
            try:
                self.raise_()
                self.activateWindow()
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass

            default_dir = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\Resonances"
            start_dir = default_dir if os.path.isdir(default_dir) else os.getcwd()

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
    win = RamseyDetuningScanApp()
    win.resize(1200, 700)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
