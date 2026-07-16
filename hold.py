import os

# Force pyqtgraph to use PyQt5 (prevents PySide/PyQt mismatches)
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import csv
import re
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from adriq.experiment import *


def gaussian(amplitude, mu, sigma):
    # Sigma is half-width at 1/e height (multiply by 2*sqrt(ln(2)) to get FWHM ~1.665)
    return lambda t: amplitude * np.exp(-((t - mu) ** 2) / (sigma ** 2))


def binomial_stderr_percent(k: int, n: int) -> float:
    """1-sigma standard error (%) for binomial proportion k/n."""
    n = int(n)
    k = int(k)
    if n <= 0:
        return float("nan")
    p = k / n
    if p < 0.0:
        p = 0.0
    elif p > 1.0:
        p = 1.0
    return float(np.sqrt(p * (1.0 - p) / n) * 100.0)


def next_spectrum_filename(directory=".", pad=3):
    pattern = re.compile(r"^cavity_raman_(\d+)\.csv$")
    max_n = 0
    try:
        for fname in os.listdir(directory):
            m = pattern.match(fname)
            if m:
                max_n = max(max_n, int(m.group(1)))
    except FileNotFoundError:
        pass
    n = max_n + 1
    return os.path.join(directory, f"cavity_raman_{n:0{pad}d}.csv")


def save_scan_csv(detunings, eff3, eff4, n_valid, c3, c4, directory="."):
    out_dir = os.path.join(str(directory), "Spectra")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = next_spectrum_filename(out_dir)

    detunings = np.asarray(detunings, dtype=float)
    eff3 = np.asarray(eff3, dtype=float)
    eff4 = np.asarray(eff4, dtype=float)
    n_valid = np.asarray(n_valid, dtype=int)
    c3 = np.asarray(c3, dtype=int)
    c4 = np.asarray(c4, dtype=int)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "detuning_MHz",
            "efficiency_chan3_percent",
            "efficiency_chan4_percent",
            "counts_chan3",
            "counts_chan4",
            "N_valid_pulses",
        ])
        for d, e3, e4, cc3, cc4, nv in zip(detunings, eff3, eff4, c3, c4, n_valid):
            w.writerow([float(d), float(e3), float(e4), int(cc3), int(cc4), int(nv)])

    print(f"Saved: {out_csv}")
    return out_csv


def init_experiment(
    *,
    ram_step=0.08,
    n_cycles=50_000,
    cooling_length: int = 12,
    state_prep_mode: str = "stretch",
    single_photon_length: int = 12,
):
    # This is a direct port of the sequence setup in Find_Resonance.py
    dds_dict = load_dds_dict(
        "ram",
        r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg",
    )

    pulse_sequencer = Pulse_Sequencer()
    exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=float(ram_step), N_Cycles=float(n_cycles))

    exp_sequence.set_detunings(detuning_dict={
        "850 SP1": 0,
        "850 SP2": 0,
        "854 SP1": 5,
        "854 SP2": 0,
        "397a": -48,
        "854 Cav": 0,
        "397b": -18,
        "397c": -18,
        "866S": -18,
        "866 OP": 30,
        "850 RP": 0,
    })


    exp_sequence.load_cooling(length=int(cooling_length))
    exp_sequence.load_trapping()

    if state_prep_mode == "all":
        exp_sequence.load_section("shelve_all")

    elif state_prep_mode == "plus":
        exp_sequence.load_section("pump_to_stretch")
        exp_sequence.load_section("pump_to_ground")
        exp_sequence.load_section("pi_gaussian_plus")
        prep_section_names = ["pump_to_stretch", "pump_to_ground", "pi_gaussian_plus", "quench_ground"]

    elif state_prep_mode == "minus":
        exp_sequence.load_section("pump_to_stretch")
        exp_sequence.load_section("pump_to_ground")
        exp_sequence.load_section("pi_gaussian_minus")
        prep_section_names = ["pump_to_stretch", "pump_to_ground", "pi_gaussian_minus", "quench_ground"]

    else:
        raise ValueError(f"Unknown state_prep_mode: {state_prep_mode!r}")

    prep_len = sum(
    s["duration"]
    for s in exp_sequence.playback_sections
    if s["name"] in prep_section_names
)

    exp_sequence.create_section(
        name="Single Photon",
        duration=int(single_photon_length),
        dds_functions={
            "854 SP1": gaussian(amplitude=0.18, mu=6, sigma=1),
        },
        pmt_gate_high=False,
        coincidence_detector=True,
        coincidence_low=1,
        coincidence_high=100,
    )

    exp_sequence.build_ram_arrays()
    exp_sequence.flash()

    exp_runner = Experiment_Runner(
        dds_dict,
        pulse_sequencer,
        timeout=100,
        pmt_threshold=2800,
        expected_fluorescence=8000,
        pulse_expected_fluorescence=1200,
        sp_threshold=None,
        load_timeout=50,
        cavity_lock=False,
    )

    return exp_sequence, exp_runner, prep_len


# --- Qt binding compatibility (PyQt* vs PySide*) ---
Signal = getattr(QtCore, "pyqtSignal", None) or getattr(QtCore, "Signal", None)
Slot = getattr(QtCore, "pyqtSlot", None) or getattr(QtCore, "Slot", None)
if Signal is None or Slot is None:
    raise RuntimeError("Could not find Qt Signal/Slot (unsupported Qt binding via pyqtgraph.Qt).")


class ScanWorker(QtCore.QObject):
    point = Signal(float, float, float, int, int, int)  # detuning, eff3, eff4, N_valid, c3, c4
    scan_done = Signal(object, object, object, object, object, object)  # dets, eff3, eff4, nvs, c3, c4
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
            exp_sequence, exp_runner, prep_len = init_experiment(
                ram_step=0.08,
                n_cycles=p["n_cycles"],
                cooling_length=p.get("cooling_length", 12),
                state_prep_mode=p.get("state_prep_mode", "stretch"),
            )
            exp_runner.experiment_trap_depth = float(p["trap_depth"])

            center = float(p["center"])
            span = float(p["span"])
            step = float(p["step"])
            detunings = np.arange(center - span, center + span + 1e-12, step)

            amplitude = float(p["amplitude"])
            mu = float(p["gauss_mu"])
            sigma = float(p["gauss_sigma"])

            # Fixed settings (not user-editable in UI)
            window_name = "signal-sp"
            # Cutoff window based on the sequence timing:
            #   lower = cooling + state_prep + quench_S + 0.3 (safety)
            #   upper = lower + single_photon_length
            cooling_len = float(p.get("cooling_length", 12))
            single_photon_len = 12.0
            lower_cutoff = cooling_len + prep_len + 0.3
            upper_cutoff = lower_cutoff + single_photon_len
            print(f"Cutoff window: {lower_cutoff:.2f} to {upper_cutoff:.2f} us")
            n_runs = int(p["n_runs"])

            eff3_list, eff4_list, nv_list, c3_list, c4_list = [], [], [], [], []

            self.status.emit("Scanning...")
            for d in detunings:
                if self._should_stop():
                    self.status.emit("Stopped.")
                    self.finished.emit()
                    return

                exp_sequence.edit_detunings({"854 SP1": float(d)})
                exp_sequence.edit_section(
                    name="Single Photon",
                    dds_functions={
                        "854 SP1": gaussian(amplitude=amplitude, mu=mu, sigma=sigma),
                    },
                )

                exp_sequence.build_ram_arrays()
                exp_sequence.flash()

                # Keep behavior identical to Find_Resonance.py
                exp_runner.measure_expected_fluorescence()

                exp_runner.clear_channels()
                exp_runner.start_experiment(N=n_runs)

                total_counts = exp_runner.get_counts_in_window(window_name, lower_cutoff=lower_cutoff, upper_cutoff=upper_cutoff)

                counts_channel_4 = int(total_counts.get("single_photon_chan4", 0) or 0)
                counts_channel_3 = int(total_counts.get("single_photon_chan3", 0) or 0)
                n_valid = int(getattr(exp_runner, "N_Valid_Pulses", 0) or 0)

                if n_valid <= 0:
                    eff4 = 0.0
                    eff3 = 0.0
                else:
                    eff4 = (counts_channel_4 / n_valid) * 100.0
                    eff3 = (counts_channel_3 / n_valid) * 100.0

                eff3_list.append(float(eff3))
                eff4_list.append(float(eff4))
                nv_list.append(int(n_valid))
                c3_list.append(int(counts_channel_3))
                c4_list.append(int(counts_channel_4))

                self.point.emit(float(d), float(eff3), float(eff4), int(n_valid), int(counts_channel_3), int(counts_channel_4))

            dets = np.asarray(detunings, dtype=float)
            eff3 = np.asarray(eff3_list, dtype=float)
            eff4 = np.asarray(eff4_list, dtype=float)
            nvs = np.asarray(nv_list, dtype=int)
            c3 = np.asarray(c3_list, dtype=int)
            c4 = np.asarray(c4_list, dtype=int)

            if bool(p.get("save_csv", True)):
                save_scan_csv(dets, eff3, eff4, nvs, c3, c4, directory=p["output_dir"])

            self.scan_done.emit(dets, eff3, eff4, nvs, c3, c4)
            self.status.emit("Done.")
            self.finished.emit()

        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            self.finished.emit()


class CavityRamanSpectroscopyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cavity Raman spectroscopy (pyqtgraph)")

        # --- controls
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QGridLayout(w)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form, 0, 0, 1, 1)

        self.center = QtWidgets.QDoubleSpinBox(); self.center.setDecimals(6); self.center.setRange(-1e6, 1e6); self.center.setValue(0.0)
        self.span = QtWidgets.QDoubleSpinBox(); self.span.setDecimals(6); self.span.setRange(0.0, 1e6); self.span.setValue(12.0)
        self.step = QtWidgets.QDoubleSpinBox(); self.step.setDecimals(6); self.step.setRange(1e-6, 1e3); self.step.setValue(0.2)

        # NEW: cooling length (used both in sequence and cutoff timing)
        self.cooling_length = QtWidgets.QSpinBox(); self.cooling_length.setRange(1, 10_000_000); self.cooling_length.setValue(12)
        self.cooling_length.setToolTip("Cooling section length (affects TDC cutoff timing)")

        self.amplitude = QtWidgets.QDoubleSpinBox(); self.amplitude.setDecimals(4); self.amplitude.setRange(0.0, 1.0); self.amplitude.setSingleStep(0.01); self.amplitude.setValue(0.18)
        self.gauss_mu = QtWidgets.QDoubleSpinBox(); self.gauss_mu.setDecimals(3); self.gauss_mu.setRange(-1e6, 1e6); self.gauss_mu.setValue(6.0)
        self.gauss_sigma = QtWidgets.QDoubleSpinBox(); self.gauss_sigma.setDecimals(3); self.gauss_sigma.setRange(1e-6, 1e6); self.gauss_sigma.setValue(1.0)

        # NEW: state preparation selection (matches Spectroscopy_App style)
        self.state_prep_mode = QtWidgets.QComboBox()
        self.state_prep_mode.addItem("Pump to all D5/2 states", "all")
        self.state_prep_mode.addItem("|D5/2, m=-5/2>", "minus")
        self.state_prep_mode.addItem("|D5/2, m=+3/2>", "plus")

        self.state_prep_mode.setCurrentIndex(0)
        self.state_prep_mode.setToolTip("Select state preparation preset section")

        self.n_runs = QtWidgets.QSpinBox(); self.n_runs.setRange(1, 999); self.n_runs.setValue(3)
        self.n_cycles = QtWidgets.QSpinBox(); self.n_cycles.setRange(1, 10_000_000); self.n_cycles.setValue(50_000)

        self.trap_depth = QtWidgets.QDoubleSpinBox(); self.trap_depth.setDecimals(3); self.trap_depth.setRange(0.0, 1.0); self.trap_depth.setSingleStep(0.01); self.trap_depth.setValue(0.8)

        self.output_dir = QtWidgets.QLineEdit(os.getcwd())
        self.output_dir.setToolTip("Directory to save cavity_raman_###.csv files")

        self.save_csv = QtWidgets.QCheckBox("Save CSV")
        self.save_csv.setChecked(True)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.status = QtWidgets.QLabel("Idle.")

        form.addRow("Center detuning [MHz]", self.center)
        form.addRow("Span [MHz]", self.span)
        form.addRow("Step [MHz]", self.step)
        form.addRow("Cooling length", self.cooling_length)
        form.addRow("854 SP1 amplitude", self.amplitude)
        form.addRow("Gaussian mu", self.gauss_mu)
        form.addRow("Gaussian sigma", self.gauss_sigma)
        form.addRow("State prep", self.state_prep_mode)
        form.addRow("N runs (per detuning point)", self.n_runs)
        form.addRow("N cycles (builder)", self.n_cycles)
        form.addRow("Trap depth", self.trap_depth)
        form.addRow("Output dir", self.output_dir)
        form.addRow("", self.save_csv)
        form.addRow("", self.start_btn)
        form.addRow("", self.stop_btn)
        form.addRow("Status", self.status)

        # --- plot
        pg.setConfigOptions(antialias=True)
        self.plots = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plots, 0, 1, 1, 1)

        self.p = self.plots.addPlot(title="Efficiency vs detuning")
        self.p.setLabel("bottom", "Detuning (MHz)")
        self.p.setLabel("left", "Efficiency (%)")
        self.p.showGrid(x=True, y=True)

        self.p.addLegend()

        self.curve4 = self.p.plot(
            [], [],
            pen=pg.mkPen("b", width=2),
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush("b"),
            symbolPen=pg.mkPen("b"),
            name="Detector 4",
        )
        self.curve3 = self.p.plot(
            [], [],
            pen=pg.mkPen("r", width=2, style=QtCore.Qt.PenStyle.DashLine),
            symbol="t",
            symbolSize=8,
            symbolBrush=pg.mkBrush("r"),
            symbolPen=pg.mkPen("r"),
            name="Detector 3",
        )

        self._err_beam = 0.0
        self.err4 = pg.ErrorBarItem(
            x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=self._err_beam,
            pen=pg.mkPen("b"),
        )
        self.err3 = pg.ErrorBarItem(
            x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=self._err_beam,
            pen=pg.mkPen("r"),
        )
        self.p.addItem(self.err4)
        self.p.addItem(self.err3)

        self._thread = None
        self._worker = None
        self._xs = []
        self._eff3 = []
        self._eff4 = []
        self._err3 = []
        self._err4 = []

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

    def _params(self):
        return dict(
            center=float(self.center.value()),
            span=float(self.span.value()),
            step=float(self.step.value()),
            cooling_length=int(self.cooling_length.value()),
            amplitude=float(self.amplitude.value()),
            gauss_mu=float(self.gauss_mu.value()),
            gauss_sigma=float(self.gauss_sigma.value()),
            state_prep_mode=str(self.state_prep_mode.currentData() or "stretch"),
            n_runs=int(self.n_runs.value()),
            n_cycles=int(self.n_cycles.value()),
            trap_depth=float(self.trap_depth.value()),
            output_dir=str(self.output_dir.text()).strip() or os.getcwd(),
            save_csv=bool(self.save_csv.isChecked()),
        )

    @Slot()
    def start(self):
        self._xs, self._eff3, self._eff4, self._err3, self._err4 = [], [], [], [], []
        self.curve3.setData([], [])
        self.curve4.setData([], [])
        self.err3.setData(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=0.0)
        self.err4.setData(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=0.0)

        params = self._params()
        self._err_beam = 0.6 * float(params.get("step", 0.2))

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("Starting...")

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

    @Slot(float, float, float, int, int, int)
    def on_point(self, detuning, eff3, eff4, n_valid, c3, c4):
        self._xs.append(float(detuning))
        self._eff3.append(float(eff3))
        self._eff4.append(float(eff4))
        self._err3.append(binomial_stderr_percent(int(c3), int(n_valid)))
        self._err4.append(binomial_stderr_percent(int(c4), int(n_valid)))

        x = np.asarray(self._xs, dtype=float)
        y3 = np.asarray(self._eff3, dtype=float)
        y4 = np.asarray(self._eff4, dtype=float)

        self.curve3.setData(x, y3)
        self.curve4.setData(x, y4)

        e3 = np.asarray(self._err3, dtype=float)
        e4 = np.asarray(self._err4, dtype=float)

        top3 = np.nan_to_num(e3, nan=0.0, posinf=0.0, neginf=0.0)
        top4 = np.nan_to_num(e4, nan=0.0, posinf=0.0, neginf=0.0)
        self.err3.setData(x=x, y=y3, top=top3, bottom=top3, beam=self._err_beam)
        self.err4.setData(x=x, y=y4, top=top4, bottom=top4, beam=self._err_beam)

        self.p.enableAutoRange(axis="y", enable=True)

    @Slot(object, object, object, object, object, object)
    def on_scan_done(self, dets, eff3, eff4, nvs, c3, c4):
        # Final update for safety
        x = np.asarray(dets, float)
        y3 = np.asarray(eff3, float)
        y4 = np.asarray(eff4, float)
        self.curve3.setData(x, y3)
        self.curve4.setData(x, y4)

        nvs = np.asarray(nvs, int)
        c3 = np.asarray(c3, int)
        c4 = np.asarray(c4, int)
        e3 = np.asarray([binomial_stderr_percent(int(k), int(n)) for k, n in zip(c3, nvs)], float)
        e4 = np.asarray([binomial_stderr_percent(int(k), int(n)) for k, n in zip(c4, nvs)], float)
        top3 = np.nan_to_num(e3, nan=0.0, posinf=0.0, neginf=0.0)
        top4 = np.nan_to_num(e4, nan=0.0, posinf=0.0, neginf=0.0)
        self.err3.setData(x=x, y=y3, top=top3, bottom=top3, beam=self._err_beam)
        self.err4.setData(x=x, y=y4, top=top4, bottom=top4, beam=self._err_beam)

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


def main():
    app = QtWidgets.QApplication([])
    win = CavityRamanSpectroscopyApp()
    win.resize(1200, 700)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
