import os
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

from adriq.experiment import *
from adriq.Optomechanics import *

# --- Qt binding compatibility (PyQt* vs PySide*) ---
Signal = getattr(QtCore, "pyqtSignal", None) or getattr(QtCore, "Signal", None)
Slot = getattr(QtCore, "pyqtSlot", None) or getattr(QtCore, "Slot", None)
if Signal is None or Slot is None:
    raise RuntimeError("Could not find Qt Signal/Slot (unsupported Qt binding via pyqtgraph.Qt).")


def _probe_time_from_pi(pi_time_us: int, fraction: float = 0.60) -> int:
    pi_time_us = int(pi_time_us)
    if pi_time_us <= 0:
        return 0
    t = int(round(float(fraction) * pi_time_us))
    return max(1, t)


def _prepare_alignment_runner(
    probe_time_us: int,
    *,
    detuning_mhz: float,
    probe_amplitude: float,
    trap_depth: float,
    external_trigger: bool,
    n_cycles: int,
    op_mode: str,
):
    """
    One-time setup: build sequence, build RAM, flash. Returns a ready Experiment_Runner.
    """
    pulse_sequencer = Pulse_Sequencer()
    dds_dict = load_dds_dict(
        "ram",
        r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg",
    )

    exp_sequence = Experiment_Builder(
        dds_dict,
        pulse_sequencer,
        ram_step=0.1,
        N_Cycles=int(n_cycles),
        external_trigger=bool(external_trigger),
    )

    exp_sequence.set_detunings(detuning_dict={
        "854 SP1": 0, "854 SP2": 0, "397b": 0, "397c": -18, "866": 0, "866 OP": 0,
        "850 RP": 0, "866 RP": -10, "729 Temp1": float(detuning_mhz), "729 Temp2": 0
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

    probe_time_us_i = int(probe_time_us)
    if probe_time_us_i > 0:
        section_dur = max(4, probe_time_us_i)

        if probe_time_us_i < 4:
            def _probe_shape(t, amp=float(probe_amplitude), ton=float(probe_time_us_i)):
                return amp if t < ton else 0.0
            dds_fn = _probe_shape
        else:
            dds_fn = (lambda t, amp=float(probe_amplitude): amp)

        exp_sequence.create_section(
            name="Probe",
            duration=int(section_dur),
            dds_functions={"729 Temp1": dds_fn},
            pmt_gate_high=True,
        )

    exp_sequence.load_measurement()
    exp_sequence.build_ram_arrays()
    exp_sequence.flash()  # <-- flash ONCE here

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
    exp_runner.experiment_trap_depth = float(trap_depth)
    exp_runner.clear_channels()
    return exp_runner


def _dark_from_runner(exp_runner):
    N_valid = int(exp_runner.N_Valid_Pulses or 0)
    N_bright = int(exp_runner.bright_counts or 0)

    P_bright = (N_bright / N_valid) if N_valid > 0 else 0.0
    P_dark = 1.0 - float(P_bright)
    err = float(np.sqrt(P_dark * (1.0 - P_dark) / N_valid)) if N_valid > 0 else float("nan")
    return P_dark, err, N_valid


class AlignmentWorker(QtCore.QObject):
    point = Signal(float, float, float, int)  # (t_s, P_dark, stderr, N_valid)
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
            pi_time_us = int(p["pi_time_us"])
            probe_time_us = _probe_time_from_pi(pi_time_us, 0.60)
            if probe_time_us <= 0:
                raise ValueError("pi time must be > 0.")

            self.status.emit("Building + flashing once...")
            exp_runner = _prepare_alignment_runner(
                probe_time_us,
                detuning_mhz=p["detuning_mhz"],
                probe_amplitude=p["probe_amplitude"],
                trap_depth=p["trap_depth"],
                external_trigger=p["external_trigger"],
                n_cycles=p["n_cycles"],
                op_mode=p["op_mode"],
            )

            self.status.emit(f"Running. π={pi_time_us} µs → probe={probe_time_us} µs (60%).")
            t_start = time.time()

            while not self._should_stop():
                exp_runner.clear_channels()

                n_runs = int(p["n_runs"])
                for _ in range(max(1, n_runs)):
                    if self._should_stop():
                        break
                    exp_runner.start_experiment(N=1)  # cannot interrupt mid-call; stop is checked between calls

                if self._should_stop():
                    break

                P_dark, err, N_valid = _dark_from_runner(exp_runner)
                t_s = time.time() - t_start
                self.point.emit(float(t_s), float(P_dark), float(err), int(N_valid))

                delay_ms = int(p.get("delay_ms", 0))
                if delay_ms > 0:
                    for _ in range(max(1, delay_ms // 10)):
                        if self._should_stop():
                            break
                        QtCore.QThread.msleep(10)

            self.status.emit("Stopped.")
            self.finished.emit()

        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            self.finished.emit()


class AlignmentApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Alignment (live P_dark)")

        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QGridLayout(w)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form, 0, 0, 1, 1)

        self.external_trigger = QtWidgets.QCheckBox("External trigger")
        self.external_trigger.setChecked(False)

        self.detuning = QtWidgets.QDoubleSpinBox()
        self.detuning.setDecimals(6)
        self.detuning.setRange(-1e6, 1e6)
        self.detuning.setValue(-23.305)

        self.power = QtWidgets.QDoubleSpinBox()
        self.power.setDecimals(4)
        self.power.setRange(0.0, 1.0)
        self.power.setSingleStep(0.01)
        self.power.setValue(1.0)

        self.pi_time = QtWidgets.QSpinBox()
        self.pi_time.setRange(1, 10_000_000)
        self.pi_time.setValue(20)

        self.trap_depth = QtWidgets.QDoubleSpinBox()
        self.trap_depth.setDecimals(3)
        self.trap_depth.setRange(0.0, 1.0)
        self.trap_depth.setSingleStep(0.01)
        self.trap_depth.setValue(0.55)

        self.n_runs = QtWidgets.QSpinBox()
        self.n_runs.setRange(1, 999)
        self.n_runs.setValue(3)

        self.n_cycles = QtWidgets.QSpinBox()
        self.n_cycles.setRange(1, 10_000_000)
        self.n_cycles.setValue(500)

        self.op_mode = QtWidgets.QComboBox()
        self.op_mode.addItem("Pump to S1/2 m = -1/2", "m_minus")
        self.op_mode.addItem("Pump to both S1/2 m = ±1/2", "both")
        self.op_mode.setCurrentIndex(0)

        self.delay_ms = QtWidgets.QSpinBox()
        self.delay_ms.setRange(0, 60_000)
        self.delay_ms.setValue(0)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.status = QtWidgets.QLabel("Idle.")

        form.addRow("", self.external_trigger)
        form.addRow("Detuning [MHz]", self.detuning)
        form.addRow("Power/amplitude [0..1]", self.power)
        form.addRow("π time [µs]", self.pi_time)
        form.addRow("Probe time [µs]", QtWidgets.QLabel("0.6 × π time"))  # fixed by design
        form.addRow("Trap depth", self.trap_depth)
        form.addRow("N runs (per point)", self.n_runs)
        form.addRow("N cycles (builder)", self.n_cycles)
        form.addRow("Optical pumping", self.op_mode)
        form.addRow("Delay between points [ms]", self.delay_ms)

        _btn_row = QtWidgets.QWidget()
        _btn_row_l = QtWidgets.QHBoxLayout(_btn_row)
        _btn_row_l.setContentsMargins(0, 0, 0, 0)
        _btn_row_l.addWidget(self.start_btn)
        _btn_row_l.addWidget(self.stop_btn)
        _btn_row_l.addWidget(self.clear_btn)
        form.addRow("", _btn_row)

        form.addRow("Status", self.status)

        pg.setConfigOptions(antialias=True)
        self.plots = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plots, 0, 1, 1, 1)

        self.p = self.plots.addPlot(title="Alignment: dark probability vs time")
        self.p.setLabel("bottom", "Elapsed time (s)")
        self.p.setLabel("left", "P_dark")
        self.p.showGrid(x=True, y=True)

        self.curve = self.p.plot([], [], pen=None, symbol="o", symbolSize=6)
        self.err_item = pg.ErrorBarItem(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=0.6)
        self.p.addItem(self.err_item)

        self._thread = None
        self._worker = None
        self._t = []
        self._p_dark = []
        self._p_err = []
        self._n_valid = []

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.clear_btn.clicked.connect(self.clear)

    def _params(self):
        return dict(
            external_trigger=bool(self.external_trigger.isChecked()),
            detuning_mhz=float(self.detuning.value()),
            probe_amplitude=float(self.power.value()),
            pi_time_us=int(self.pi_time.value()),
            trap_depth=float(self.trap_depth.value()),
            n_runs=int(self.n_runs.value()),
            n_cycles=int(self.n_cycles.value()),
            op_mode=str(self.op_mode.currentData() or "m_minus"),
            delay_ms=int(self.delay_ms.value()),
        )

    @Slot()
    def clear(self):
        self._t, self._p_dark, self._p_err, self._n_valid = [], [], [], []
        self.curve.setData([], [])
        self.err_item.setData(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]))
        self.status.setText("Cleared.")

    @Slot()
    def start(self):
        self.status.setText("Starting...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        params = self._params()

        self._thread = QtCore.QThread(self)
        self._worker = AlignmentWorker(params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.point.connect(self.on_point)
        self._worker.status.connect(self.status.setText)
        self._worker.failed.connect(self.on_failed)
        self._worker.finished.connect(self.on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    @Slot()
    def stop(self):
        # CHANGED: don't queue a stop call; set flag directly + interrupt the thread.
        if getattr(self, "_worker", None) is not None:
            try:
                self._worker.stop()  # just sets a boolean; safe cross-thread in CPython
            except Exception:
                pass
        if getattr(self, "_thread", None) is not None:
            try:
                self._thread.requestInterruption()
            except Exception:
                pass

        self.stop_btn.setEnabled(False)
        self.status.setText("Stopping (will stop after current start_experiment call returns)...")

    @Slot()
    def on_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._worker = None
        self._thread = None

    @Slot(str)
    def on_failed(self, msg):
        self.status.setText(f"Error: {msg}")

    @Slot(float, float, float, int)
    def on_point(self, t_s, p_dark, stderr, n_valid):
        self._t.append(float(t_s))
        self._p_dark.append(float(p_dark))
        self._p_err.append(float(stderr))
        self._n_valid.append(int(n_valid))

        x = np.asarray(self._t, dtype=float)
        y = np.asarray(self._p_dark, dtype=float)
        e = np.asarray(self._p_err, dtype=float)

        self.curve.setData(x, y)
        top = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
        self.err_item.setData(x=x, y=y, top=top, bottom=top)

        self.p.enableAutoRange(axis="y", enable=True)


def main():
    app = QtWidgets.QApplication([])
    win = AlignmentApp()
    win.resize(1200, 700)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
