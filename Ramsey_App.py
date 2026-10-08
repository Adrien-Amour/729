import os

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

from adriq.experiment import correct_probabilities_with_spam, Experiment_Builder, Experiment_Runner, Pulse_Sequencer, load_dds_dict
from adriq.Resonance_Log import read_resonance


# --- Qt binding compatibility (PyQt* vs PySide*) ---
Signal = getattr(QtCore, "pyqtSignal", None) or getattr(QtCore, "Signal", None)
Slot = getattr(QtCore, "pyqtSlot", None) or getattr(QtCore, "Slot", None)
if Signal is None or Slot is None:
    raise RuntimeError("Could not find Qt Signal/Slot (unsupported Qt binding via pyqtgraph.Qt).")


def ramsey_fringe_model(phase_deg, A, phi0_rad, C):
    """0.5*A*(1 + C*cos(2*phase + phi0)), phase in degrees."""
    phase_deg = np.asarray(phase_deg, dtype=float)
    return 0.5 * float(A) * (1.0 + float(C) * np.cos(2.0 * np.deg2rad(phase_deg) + float(phi0_rad)))


def fit_ramsey_fringe(phases_deg, p_dark, p_err):
    phases_deg = np.asarray(phases_deg, dtype=float)
    p_dark = np.asarray(p_dark, dtype=float)
    p_err = np.asarray(p_err, dtype=float)

    # Avoid zero/NaN sigma
    if np.any(np.isfinite(p_err) & (p_err > 0)):
        fallback = float(np.nanmedian(p_err[(p_err > 0) & np.isfinite(p_err)]))
    else:
        fallback = 1.0
    sigma = np.where((p_err > 0) & np.isfinite(p_err), p_err, fallback)

    A0 = float(np.clip(np.nanmax(p_dark) * 2.0 if np.isfinite(np.nanmax(p_dark)) else 1.0, 1e-6, 2.0))
    phi0_0 = 0.0
    C0 = 1.0

    popt, pcov = curve_fit(
        ramsey_fringe_model,
        phases_deg,
        p_dark,
        sigma=sigma,
        p0=[A0, phi0_0, C0],
        absolute_sigma=True,
        bounds=([0.0, -np.inf, -1.0], [np.inf, np.inf, 1.0]),
        maxfev=20000,
    )

    perr = np.sqrt(np.diag(pcov)) if pcov.size else np.array([np.nan, np.nan, np.nan])
    return popt, perr, pcov


def parse_csv_list(text: str):
    """Parse comma/space separated list of numbers (floats)"""
    parts = [p.strip() for p in text.replace(";", ",").replace(" ", ",").split(",")]
    vals = []
    for p in parts:
        if not p:
            continue
        vals.append(float(p))
    return vals


@dataclass
class RamseyPoint:
    t_precess_us: float
    phase_deg: float
    p_dark: float
    p_err: float
    n_valid: int


class RamseyWorker(QtCore.QObject):
    point = Signal(float, float, float, float, int)  # (t_precess_us, phase_deg, p_dark, p_err, n_valid)
    t_done = Signal(float, object, object, object, object)  # (t_precess_us, phases, p_dark, p_err, fit_dict)
    scan_done = Signal(object)  # summary dict
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
        return bool(self._stop)

    def _build_sequence_for_t(self, t_precess_us: float, *, dds_dict, pulse_sequencer):
        """Build/flash sequence for a given precession time."""
        pulse_time = float(self.params["pulse_time_us"])
        power = float(self.params["power"])

        # Match Rabi_Oscillations fixed hardware/sequence settings
        RAM_STEP = 0.2
        COOLING_US = 2000
        TIMEOUT_S = 100.0
        PMT_THRESHOLD = 2000
        EXPECTED_FLUOR = 8000
        PULSE_EXPECTED_FLUOR = 4200

        exp_sequence = Experiment_Builder(
            dds_dict,
            pulse_sequencer,
            ram_step=float(RAM_STEP),
            N_Cycles=int(self.params["n_cycles"]),
            external_trigger=bool(self.params["external_trigger"]),
        )

        exp_sequence.set_detunings(detuning_dict={
            "729 t1": float(self.params["detuning_mhz"])  # MHz
        })

        exp_sequence.load_cooling(length=int(COOLING_US))
        exp_sequence.load_trapping()
        exp_sequence.load_section("pump_to_stretch")
        exp_sequence.load_section("pump_to_ground")
        exp_sequence.load_section("quench_metastables")

        duration_us = int(math.ceil(2.0 * pulse_time + float(t_precess_us)))

        def dds_fn(t, a=power, ton=pulse_time, tp=float(t_precess_us)):
            tt = float(t)
            if tt < ton:
                return a
            if tt < ton + tp:
                return 0.0
            if tt < ton + tp + ton:
                return a
            return 0.0

        def phase_fn(t, phase2=0.0, ton=pulse_time, tp=float(t_precess_us)):
            tt = float(t)
            # Apply phase only during 2nd pulse
            return float(phase2) if (ton + tp/2) <= tt < (ton + tp + ton) else 0.0

        exp_sequence.create_section(
            name="RamseyBlock",
            duration=duration_us,
            dds_functions={"729 t1": dds_fn},
            phase_functions={"729 t1": phase_fn},
            pmt_gate_high=True,
        )

        exp_sequence.load_measurement()
        exp_sequence.build_ram_arrays()
        exp_sequence.flash()

        return exp_sequence

    @Slot()
    def run(self):
        try:
            # Build shared hardware context once
            pulse_sequencer = Pulse_Sequencer()
            dds_dict = load_dds_dict(
                "ram",
                r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg",
            )

            # Match Rabi_Oscillations fixed hardware/sequence settings
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

            t_list = list(self.params["t_precess_list_us"])
            phases = np.arange(
                float(self.params["phase_start_deg"]),
                float(self.params["phase_end_deg"]) + 0.5 * float(self.params["phase_step_deg"]),
                float(self.params["phase_step_deg"]),
            )

            summary = {
                "t_precess_us": [],
                "A": [],
                "phi0_rad": [],
                "C": [],
                "C_err": [],
            }

            for ti, t_precess_us in enumerate(t_list):
                if self._should_stop():
                    break

                self.status.emit(f"Building sequence for t_precess={t_precess_us} µs")
                exp_sequence = self._build_sequence_for_t(
                    t_precess_us,
                    dds_dict=dds_dict,
                    pulse_sequencer=pulse_sequencer,
                )

                p_dark_vals = []
                p_err_vals = []
                n_valid_vals = []

                for phase in phases:
                    if self._should_stop():
                        break

                    self.status.emit(f"t_precess={t_precess_us} µs — phase {phase:.1f}°")

                    # Apply phase only during 2nd pulse
                    pulse_time = float(self.params["pulse_time_us"])
                    tp = float(t_precess_us)
                    exp_sequence.edit_section(
                        "RamseyBlock",
                        phase_functions={
                            "729 t1": (lambda t, p=float(phase), ton=pulse_time, tpre=tp: (p if (ton + tpre) <= float(t) < (ton + tpre + ton) else 0.0))
                        },
                    )
                    exp_sequence.build_ram_arrays()
                    exp_sequence.flash()

                    exp_runner.clear_channels()
                    exp_runner.start_experiment(N=int(self.params["n_runs_per_phase"]))

                    n_valid = int(getattr(exp_runner, "N_Valid_Pulses", 0) or 0)
                    n_bright = int(getattr(exp_runner, "bright_counts", 0) or 0)

                    if n_valid > 0:
                        p_bright = n_bright / n_valid
                        p_dark = 1.0 - float(p_bright)
                        err = float(np.sqrt(p_dark * (1.0 - p_dark) / n_valid))
                    else:
                        p_bright = 0.0
                        p_dark = 0.0
                        err = float("nan")

                    if bool(self.params["apply_spam_correction"]):
                        _, _, p_dark_corr, err_corr = correct_probabilities_with_spam(p_bright, err, p_dark, err)
                    else:
                        p_dark_corr, err_corr = p_dark, err

                    p_dark_vals.append(float(p_dark_corr))
                    p_err_vals.append(float(err_corr))
                    n_valid_vals.append(int(n_valid))

                    self.point.emit(float(t_precess_us), float(phase), float(p_dark_corr), float(err_corr), int(n_valid))

                phases_used = np.asarray(phases[: len(p_dark_vals)], dtype=float)
                p_dark_arr = np.asarray(p_dark_vals, dtype=float)
                p_err_arr = np.asarray(p_err_vals, dtype=float)

                fit_dict = None
                if p_dark_arr.size >= 3 and np.all(np.isfinite(p_dark_arr)):
                    try:
                        popt, perr, _pcov = fit_ramsey_fringe(phases_used, p_dark_arr, p_err_arr)
                        A, phi0, C = map(float, popt)
                        A_err, phi0_err, C_err = map(float, perr)
                        fit_dict = {
                            "A": A,
                            "A_err": A_err,
                            "phi0_rad": phi0,
                            "phi0_err": phi0_err,
                            "C": C,
                            "C_err": C_err,
                        }

                        summary["t_precess_us"].append(float(t_precess_us))
                        summary["A"].append(A)
                        summary["phi0_rad"].append(phi0)
                        summary["C"].append(C)
                        summary["C_err"].append(C_err)
                    except Exception:
                        fit_dict = None

                self.t_done.emit(float(t_precess_us), phases_used, p_dark_arr, p_err_arr, fit_dict)

                if bool(self.params["save_csv"]):
                    out_dir = str(self.params["output_dir"])
                    os.makedirs(out_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = os.path.join(out_dir, f"plus_ramsey_tprecess_{t_precess_us:g}us_{ts}.csv")
                    with open(out_path, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["phase_deg", "p_dark", "stderr"])
                        for ph, pd, er in zip(phases_used, p_dark_arr, p_err_arr):
                            w.writerow([float(ph), float(pd), float(er)])

            self.scan_done.emit(summary)
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            self.finished.emit()


class RamseyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ramsey scan (pyqtgraph)")

        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QGridLayout(w)

        # --- controls
        form = QtWidgets.QFormLayout()
        layout.addLayout(form, 0, 0, 1, 1)

        self.external_trigger = QtWidgets.QCheckBox("External trigger")
        self.external_trigger.setChecked(False)

        self.apply_spam = QtWidgets.QCheckBox("Apply SPAM correction")
        self.apply_spam.setChecked(True)

        self.detuning = QtWidgets.QDoubleSpinBox(); self.detuning.setDecimals(6); self.detuning.setRange(-1e6, 1e6); self.detuning.setValue(0.0)
        self.load_res_btn = QtWidgets.QPushButton("Load resonance…")
        self.load_res_btn.setToolTip("Select a resonance log CSV and load its most recent 'Resonant Detuning' value")

        self.power = QtWidgets.QDoubleSpinBox(); self.power.setDecimals(4); self.power.setRange(0.0, 1.0); self.power.setSingleStep(0.01); self.power.setValue(1.0)
        self.pulse_time = QtWidgets.QDoubleSpinBox(); self.pulse_time.setDecimals(3); self.pulse_time.setRange(0.0, 10_000_000.0); self.pulse_time.setSingleStep(0.1); self.pulse_time.setValue(3.2)

        self.t_precess_list = QtWidgets.QLineEdit("10,20,40,60,90,120,160")
        self.t_precess_list.setToolTip("Comma/space separated list of precession times in µs")

        self.phase_start = QtWidgets.QDoubleSpinBox(); self.phase_start.setDecimals(1); self.phase_start.setRange(-1e6, 1e6); self.phase_start.setValue(0.0)
        self.phase_end = QtWidgets.QDoubleSpinBox(); self.phase_end.setDecimals(1); self.phase_end.setRange(-1e6, 1e6); self.phase_end.setValue(180.0)
        self.phase_step = QtWidgets.QDoubleSpinBox(); self.phase_step.setDecimals(1); self.phase_step.setRange(0.1, 1e6); self.phase_step.setValue(10.0)

        self.n_runs = QtWidgets.QSpinBox(); self.n_runs.setRange(1, 999); self.n_runs.setValue(5)
        self.n_cycles = QtWidgets.QSpinBox(); self.n_cycles.setRange(1, 10_000_000); self.n_cycles.setValue(250)

        self.output_dir = QtWidgets.QLineEdit(os.path.join(os.getcwd(), "Ramsey_Data"))
        self.save_csv = QtWidgets.QCheckBox("Save CSV")
        self.save_csv.setChecked(True)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False)
        self.status = QtWidgets.QLabel("Idle.")

        form.addRow("", self.external_trigger)
        form.addRow("", self.apply_spam)
        form.addRow("Detuning [MHz]", self.detuning)
        form.addRow("", self.load_res_btn)
        form.addRow("Power/amplitude [0..1]", self.power)
        form.addRow("Pulse time [µs]", self.pulse_time)
        form.addRow("t_precess list [µs]", self.t_precess_list)
        form.addRow("Phase start [deg]", self.phase_start)
        form.addRow("Phase end [deg]", self.phase_end)
        form.addRow("Phase step [deg]", self.phase_step)
        form.addRow("N runs (per phase)", self.n_runs)
        form.addRow("N cycles (builder)", self.n_cycles)
        form.addRow("Output dir", self.output_dir)
        form.addRow("", self.save_csv)
        form.addRow("", self.start_btn)
        form.addRow("", self.stop_btn)
        form.addRow("Status", self.status)

        # --- plots: two panels
        pg.setConfigOptions(antialias=True)
        self.plots = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plots, 0, 1, 1, 1)

        self.p_phase = self.plots.addPlot(title="P_dark vs phase (colors = t_precess)")
        self.p_phase.setLabel("bottom", "Phase (deg)")
        self.p_phase.setLabel("left", "P_dark")
        self.p_phase.showGrid(x=True, y=True)
        self.p_phase.addLegend()

        self.plots.nextRow()

        self.p_contrast = self.plots.addPlot(title="Contrast |C| vs precession time")
        self.p_contrast.setLabel("bottom", "t_precess (µs)")
        self.p_contrast.setLabel("left", "|C|")
        self.p_contrast.showGrid(x=True, y=True)
        self.contrast_curve = self.p_contrast.plot([], [], pen=None, symbol="o", symbolSize=7)
        self.contrast_err = pg.ErrorBarItem(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=0.8)
        self.p_contrast.addItem(self.contrast_err)

        # --- runtime state
        self._thread = None
        self._worker = None
        self._t_values = []
        self._color_by_t = {}
        self._phase_data = {}  # t -> dict(phases, p_dark, p_err)
        self._phase_curve = {}  # t -> PlotDataItem
        self._fit_curve = {}  # t -> PlotDataItem
        self._phase_errbar = {}  # t -> ErrorBarItem

        self._contrast_t = []
        self._contrast_c = []
        self._contrast_e = []

        # connections
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.load_res_btn.clicked.connect(self.load_resonance_dialog)

    def _params(self) -> dict:
        t_list = parse_csv_list(self.t_precess_list.text())
        if len(t_list) == 0:
            raise ValueError("t_precess list is empty")

        return {
            "external_trigger": self.external_trigger.isChecked(),
            "apply_spam_correction": self.apply_spam.isChecked(),
            "detuning_mhz": float(self.detuning.value()),
            "power": float(self.power.value()),
            "pulse_time_us": float(self.pulse_time.value()),
            "t_precess_list_us": t_list,
            "phase_start_deg": float(self.phase_start.value()),
            "phase_end_deg": float(self.phase_end.value()),
            "phase_step_deg": float(self.phase_step.value()),
            "n_runs_per_phase": int(self.n_runs.value()),
            "n_cycles": int(self.n_cycles.value()),
            "output_dir": str(self.output_dir.text()).strip() or os.path.join(os.getcwd(), "Ramsey_Data"),
            "save_csv": self.save_csv.isChecked(),
        }

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
            QtWidgets.QMessageBox.critical(self, "Resonance load failed", f"{type(e).__name__}: {e}")

    @Slot()
    def start(self):
        try:
            params = self._params()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Invalid parameters", str(e))
            return

        # reset plots/state
        self.p_phase.clear()
        self.p_phase.addLegend()
        self._phase_data.clear()
        self._phase_curve.clear()
        self._fit_curve.clear()
        self._phase_errbar.clear()

        self._contrast_t.clear()
        self._contrast_c.clear()
        self._contrast_e.clear()
        self._update_contrast_plot()

        self._t_values = list(params["t_precess_list_us"])
        for i, t in enumerate(self._t_values):
            self._color_by_t[float(t)] = pg.intColor(i, hues=max(3, len(self._t_values)))

        # worker thread
        self._thread = QtCore.QThread()
        self._worker = RamseyWorker(params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.point.connect(self.on_point)
        self._worker.t_done.connect(self.on_t_done)
        self._worker.scan_done.connect(self.on_scan_done)
        self._worker.status.connect(self.on_status)
        self._worker.failed.connect(self.on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._cleanup)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("Running…")
        self._thread.start()

    @Slot()
    def stop(self):
        if self._worker is not None:
            self._worker.stop()
        self.status.setText("Stopping…")

    @Slot(float, float, float, float, int)
    def on_point(self, t_precess_us, phase_deg, p_dark, p_err, n_valid):
        t = float(t_precess_us)
        d = self._phase_data.setdefault(t, {"phase": [], "p_dark": [], "p_err": []})
        d["phase"].append(float(phase_deg))
        d["p_dark"].append(float(p_dark))
        d["p_err"].append(float(p_err))

        if t not in self._phase_curve:
            color = self._color_by_t.get(t, pg.intColor(len(self._phase_curve) + 1))
            self._phase_curve[t] = self.p_phase.plot(
                [],
                [],
                pen=None,
                symbol="o",
                symbolSize=6,
                symbolBrush=color,
                symbolPen=color,
                name=f"t={t:g} µs",
            )
            self._fit_curve[t] = self.p_phase.plot([], [], pen=pg.mkPen(color=color, width=2, style=QtCore.Qt.PenStyle.DashLine))

            # Error bars for this t_precess
            self._phase_errbar[t] = pg.ErrorBarItem(
                x=np.array([], dtype=float),
                y=np.array([], dtype=float),
                top=np.array([], dtype=float),
                bottom=np.array([], dtype=float),
                beam=1.5,
                pen=pg.mkPen(color=color, width=1),
            )
            self.p_phase.addItem(self._phase_errbar[t])

        self._phase_curve[t].setData(d["phase"], d["p_dark"])  # live update

        # Live-update error bars (ignore NaNs/infs)
        if t in self._phase_errbar:
            x = np.asarray(d["phase"], dtype=float)
            y = np.asarray(d["p_dark"], dtype=float)
            e = np.asarray(d["p_err"], dtype=float)
            e = np.where(np.isfinite(e) & (e >= 0), e, 0.0)
            self._phase_errbar[t].setData(x=x, y=y, top=e, bottom=e)

    @Slot(float, object, object, object, object)
    def on_t_done(self, t_precess_us, phases, p_dark, p_err, fit_dict):
        t = float(t_precess_us)
        if fit_dict is not None:
            phases = np.asarray(phases, dtype=float)
            ph_grid = np.linspace(float(np.min(phases)), float(np.max(phases)), 361)
            y_fit = ramsey_fringe_model(ph_grid, fit_dict["A"], fit_dict["phi0_rad"], fit_dict["C"])
            if t in self._fit_curve:
                self._fit_curve[t].setData(ph_grid, y_fit)

            self._contrast_t.append(t)
            self._contrast_c.append(abs(float(fit_dict["C"])))
            self._contrast_e.append(float(fit_dict.get("C_err", float("nan"))))
            self._update_contrast_plot()

    def _update_contrast_plot(self):
        if len(self._contrast_t) == 0:
            self.contrast_curve.setData([], [])
            self.contrast_err.setData(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]))
            return

        # sort by t
        order = np.argsort(np.asarray(self._contrast_t, dtype=float))
        t = np.asarray(self._contrast_t, dtype=float)[order]
        c = np.asarray(self._contrast_c, dtype=float)[order]
        e = np.asarray(self._contrast_e, dtype=float)[order]
        e = np.where(np.isfinite(e) & (e >= 0), e, 0.0)

        self.contrast_curve.setData(t, c)
        self.contrast_err.setData(x=t, y=c, top=e, bottom=e)

    @Slot(object)
    def on_scan_done(self, summary):
        # Optionally write a summary CSV alongside per-t CSVs
        try:
            out_dir = str(self.output_dir.text()).strip() or os.path.join(os.getcwd(), "Ramsey_Data")
            if self.save_csv.isChecked():
                os.makedirs(out_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(out_dir, f"ramsey_summary_{ts}.csv")
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["t_precess_us", "A", "phi0_rad", "C", "C_err"])
                    for t, A, phi0, C, Ce in zip(
                        summary.get("t_precess_us", []),
                        summary.get("A", []),
                        summary.get("phi0_rad", []),
                        summary.get("C", []),
                        summary.get("C_err", []),
                    ):
                        w.writerow([t, A, phi0, C, Ce])
        except Exception:
            pass

    @Slot(str)
    def on_status(self, text):
        self.status.setText(text)

    @Slot(str)
    def on_failed(self, msg):
        QtWidgets.QMessageBox.critical(self, "Ramsey scan failed", msg)

    @Slot()
    def _cleanup(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("Idle.")

    def closeEvent(self, event):
        # Try to stop worker cleanly
        try:
            if self._worker is not None:
                self._worker.stop()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication([])
    win = RamseyApp()
    win.resize(1200, 800)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
