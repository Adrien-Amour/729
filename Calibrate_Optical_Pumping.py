import os
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import math
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import csv
import re
import pprint
from datetime import datetime
from adriq.experiment import _load_presets_dict, upsert_autogen_function, Pulse_Sequencer, Experiment_Builder, Experiment_Runner, load_dds_dict, correct_probabilities_with_spam
from adriq.Resonance_Log import *


PRESETS_PATH = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\ps_presets\presets.py"

stretch = _load_presets_dict(PRESETS_PATH, "pump_to_stretch")
ground  = _load_presets_dict(PRESETS_PATH, "pump_to_ground")
quench  = _load_presets_dict(PRESETS_PATH, "quench_metastables")  # optional

# --- Qt binding compatibility (PyQt* vs PySide*) ---
Signal = getattr(QtCore, "pyqtSignal", None) or getattr(QtCore, "Signal", None)
Slot = getattr(QtCore, "pyqtSlot", None) or getattr(QtCore, "Slot", None)
if Signal is None or Slot is None:
    raise RuntimeError("Could not find Qt Signal/Slot (unsupported Qt binding via pyqtgraph.Qt).")


def make_probe_fn(*, amp: float, ton_us: float):
    ton_us = float(max(0.0, ton_us))
    amp = float(amp)

    def _fn(t):
        return amp if float(t) < ton_us else 0.0

    return _fn
default_dir = r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\Resonances"

DEFAULT_RESONANCE_PLUS_MHZ = 23.0
DEFAULT_RESONANCE_MINUS_MHZ = -23.0  # change to -23.0 if you truly want +/-23 to both be 23-ish

try:
    resonance_plus = read_resonance(os.path.join(default_dir, "Quadrupole_S_1_2_m_+1_2_D_5_2_m+5_2.csv"))
except Exception as e:
    print(f"[WARN] Failed to read resonance_plus; using default {DEFAULT_RESONANCE_PLUS_MHZ} MHz. ({type(e).__name__}: {e})")
    resonance_plus = DEFAULT_RESONANCE_PLUS_MHZ

try:
    resonance_minus = read_resonance(os.path.join(default_dir, "Quadrupole_S_1_2_m_-1_2_D_5_2_m-5_2.csv"))
except Exception as e:
    print(f"[WARN] Failed to read resonance_minus; using default {DEFAULT_RESONANCE_MINUS_MHZ} MHz. ({type(e).__name__}: {e})")
    resonance_minus = DEFAULT_RESONANCE_MINUS_MHZ

def set_detuning(exp_sequence: Experiment_Builder, dds_name: str, detuning_mhz: float):
    """
    Per request: initialise all detunings once, then update only the scanned/changed ones via this method.
    Uses edit_detunings so the relevant DDS is marked edited=True and gets reflashed.
    """
    exp_sequence.edit_detunings({str(dds_name): float(detuning_mhz)})


def prepare_op_cal_context(*, params: dict, dds_dict=None, pulse_sequencer=None):
    """
    Build/flash a fixed sequence once.
    Then scan by only changing detunings with set_detuning(...), rebuilding/flashing as needed.
    """
    pulse_sequencer = pulse_sequencer or Pulse_Sequencer()
    dds_dict = dds_dict or load_dds_dict(
        "ram",
        r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg",
    )

    # Keep same RAM grid approach as your Rabi app
    RAM_BASE = 0.004
    ram_step = RAM_BASE * 25  # 0.1 us

    exp_sequence = Experiment_Builder(
        dds_dict,
        pulse_sequencer,
        ram_step=ram_step,
        N_Cycles=int(params["n_cycles"]),
        external_trigger=bool(params["external_trigger"]),
    )

    # ---- Initialise ALL detunings once (requested) ----
    op_det_init = float(params.get("op_detuning_mhz", params.get("op_detuning_start_mhz", 0.0)))
    exp_sequence.set_detunings(
        detuning_dict={
            "854 SP1": 0,
            "854 SP2": 0,
            "397b": -18,  # CHANGED: move OP detuning from 397c -> 397b
            "397c": 0,    # CHANGED: keep 397c at 0 (not used for OP now)
            "866 OP": op_det_init,  # initial value; may be scanned via edit_detunings
            "850 RP": 0,
            "866 RP": 40,
            "729 t1": float(params["res_m_plus_mhz"]),  # initial; will be swapped per scenario
            "729 t2": 0,
        }
    )

    # ---- Presets (cooling/trapping/measurement) ----
    exp_sequence.load_cooling(length=2000)
    exp_sequence.load_trapping()
    exp_sequence.load_measurement()

    # ---- Optical pumping sections (parameters provided by UI) ----
    pump_stretch_len = float(params["pump_to_stretch_len_us"])
    pump_ground_len = float(params["pump_to_ground_len_us"])

    exp_sequence.create_section(
        name="Pump To Stretch",
        duration=int(max(0, round(pump_stretch_len))),
        dds_functions={
            "397b": lambda t, a=float(params["amp_397b"]): a,  # CHANGED: use 397b during optical pumping
            "866 OP": lambda t, a=float(params["amp_866op"]): a,
            "854 SP1": lambda t, a=float(params["amp_854sp1"]): a,
        },
        pmt_gate_high=True,
    )

    exp_sequence.create_section(
        name="Pump To Ground",
        duration=int(max(0, round(pump_ground_len))),
        dds_functions={
            "850 RP": lambda t, a=float(params["amp_850rp"]): a if float(t) >= 1.0 else 0.0,
            "854 SP1": lambda t, a=float(params["amp_854sp1"]): a if float(t) >= 1.0 else 0.0,
        },
        pmt_gate_high=True,
    )

    # NEW: optional quench inserted before probe (requested)
    if bool(params.get("quench_before_probe", False)):
        exp_sequence.load_section("quench_metastables")  # use preset for this

    exp_sequence.create_section(
        name="Wait Before Probe",
        duration=2,
        dds_functions={},
        pmt_gate_high=False,
    )

    probe_len_us = float(params["probe_len_us"])
    probe_section_dur = int(max(4.0, math.ceil(probe_len_us)))  # ensure section fully covers probe pulse
    exp_sequence.create_section(
        name="Probe",
        duration=int(probe_section_dur),
        dds_functions={"729 t1": make_probe_fn(amp=float(params["probe_power"]), ton_us=probe_len_us)},
        pmt_gate_high=True,
    )

    # ---- Build/flash once ----
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

    return exp_sequence, exp_runner


def run_point(
    *,
    exp_sequence: Experiment_Builder,
    exp_runner: Experiment_Runner,
    op_detuning_mhz: float,
    probe_detuning_mhz: float,
    n_runs: int,
):
    # Change only what is needed using set_detuning
    set_detuning(exp_sequence, "866 OP", float(op_detuning_mhz))
    set_detuning(exp_sequence, "729 t1", float(probe_detuning_mhz))

    exp_sequence.build_ram_arrays()
    exp_sequence.flash()

    exp_runner.clear_channels()
    exp_runner.start_experiment(N=int(n_runs))

    n_valid = int(exp_runner.N_Valid_Pulses or 0)
    n_bright = int(exp_runner.bright_counts or 0)
    p_bright = (n_bright / n_valid) if n_valid > 0 else 0.0
    p_dark = 1.0 - float(p_bright)
    err = float(np.sqrt(p_dark * (1.0 - p_dark) / n_valid)) if n_valid > 0 else float("nan")
    _, _, P_dark_corrected, stderr_corrected = correct_probabilities_with_spam(p_bright, err, p_dark, err)
    print(f"\033[91mUncorrected: P_dark={p_dark:.6f} ± {err:.6f} | Corrected: P_dark={P_dark_corrected:.6f} ± {stderr_corrected:.6f}\033[0m")
    return P_dark_corrected, stderr_corrected, int(n_valid)



class OpticalPumpingWorker(QtCore.QObject):
    point = Signal(float, str, float, float, int)  # (x_value, "m_plus"/"m_minus", P_dark, stderr, N_valid)
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

            scan_param = str(p["scan_param"])  # "op_detuning" | "pump_stretch_len" | "pump_stretch_power"
            s0 = float(p["scan_start"])
            s1 = float(p["scan_end"])
            ss = float(p["scan_step"])
            if ss <= 0:
                raise ValueError("Scan step must be > 0.")
            if s1 < s0:
                raise ValueError("Scan end must be >= scan start.")

            res_plus = float(p["res_m_plus_mhz"])
            res_minus = float(p["res_m_minus_mhz"])
            n_runs = int(p["n_runs"])

            # Inclusive scan points
            n_steps = int(np.floor((s1 - s0) / ss)) + 1
            xs = (s0 + ss * np.arange(max(n_steps, 0), dtype=float)).tolist()
            if not xs:
                xs = [s0]
            if s1 > xs[-1] and not np.isclose(xs[-1], s1, rtol=0, atol=1e-12):
                xs.append(s1)

            # Reuse hardware handles across rebuilds (minimal overhead)
            pulse_sequencer = Pulse_Sequencer()
            dds_dict = load_dds_dict(
                "ram",
                r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg",
            )

            def _run_two_scenarios(*, exp_sequence, exp_runner, x_value: float, op_det_mhz: float):
                # Scenario: probe on m=+1/2 resonance
                p_dark, err, n_valid = run_point(
                    exp_sequence=exp_sequence,
                    exp_runner=exp_runner,
                    op_detuning_mhz=float(op_det_mhz),
                    probe_detuning_mhz=res_plus,
                    n_runs=n_runs,
                )
                self.point.emit(float(x_value), "m_plus", float(p_dark), float(err), int(n_valid))

                if self._should_stop():
                    return

                # Scenario: probe on m=-1/2 resonance
                p_dark, err, n_valid = run_point(
                    exp_sequence=exp_sequence,
                    exp_runner=exp_runner,
                    op_detuning_mhz=float(op_det_mhz),
                    probe_detuning_mhz=res_minus,
                    n_runs=n_runs,
                )
                self.point.emit(float(x_value), "m_minus", float(p_dark), float(err), int(n_valid))

            self.status.emit("Scanning...")

            if scan_param == "op_detuning":
                # Build once, then only edit detunings per point (fast path)
                p0 = dict(p)
                p0["op_detuning_mhz"] = float(xs[0])  # initial
                exp_sequence, exp_runner = prepare_op_cal_context(
                    params=p0, dds_dict=dds_dict, pulse_sequencer=pulse_sequencer
                )

                for x in xs:
                    if self._should_stop():
                        self.status.emit("Stopped.")
                        self.finished.emit()
                        return
                    _run_two_scenarios(exp_sequence=exp_sequence, exp_runner=exp_runner, x_value=float(x), op_det_mhz=float(x))

            elif scan_param == "pump_stretch_len":
                # Rebuild every point because the sequence duration changes.
                fixed_op_det = float(p["fixed_op_detuning_mhz"])

                for x in xs:
                    if self._should_stop():
                        self.status.emit("Stopped.")
                        self.finished.emit()
                        return

                    plocal = dict(p)
                    plocal["op_detuning_mhz"] = fixed_op_det
                    plocal["pump_to_stretch_len_us"] = float(x)

                    # Must initialise a new DDS dict because the sequence duration changes.
                    dds_dict = load_dds_dict(
                        "ram",
                        r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg",
                    )

                    exp_sequence, exp_runner = prepare_op_cal_context(
                        params=plocal,
                        dds_dict=dds_dict,
                        pulse_sequencer=pulse_sequencer,
                    )

                    _run_two_scenarios(
                        exp_sequence=exp_sequence,
                        exp_runner=exp_runner,
                        x_value=float(x),
                        op_det_mhz=fixed_op_det,
                    )

            elif scan_param == "pump_stretch_power":
                # Rebuild every point because DDS functions change.
                fixed_op_det = float(p["fixed_op_detuning_mhz"])

                for x in xs:
                    if self._should_stop():
                        self.status.emit("Stopped.")
                        self.finished.emit()
                        return

                    plocal = dict(p)
                    plocal["op_detuning_mhz"] = fixed_op_det
                    # Interpret "pump stretch power" as the 866 OP amplitude.
                    plocal["amp_866op"] = float(x)

                    exp_sequence, exp_runner = prepare_op_cal_context(
                        params=plocal,
                        dds_dict=dds_dict,
                        pulse_sequencer=pulse_sequencer,
                    )

                    _run_two_scenarios(
                        exp_sequence=exp_sequence,
                        exp_runner=exp_runner,
                        x_value=float(x),
                        op_det_mhz=fixed_op_det,
                    )

            else:
                raise ValueError(f"Unknown scan_param: {scan_param}")

            self.status.emit("Done.")
            self.finished.emit()

        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            self.finished.emit()


class OpticalPumpingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical pumping calibration")

        def _preset_amp(preset: dict, dds_name: str) -> float:
            fn = (preset.get("dds_functions") or {}).get(dds_name, None)
            if fn is None:
                return 0.0
            if not callable(fn):  # if you ever store a scalar instead of a lambda
                return float(fn)
            d = fn.__defaults__ or ()
            return float(d[0]) if len(d) >= 1 else 0.0  # convention: first default is amplitude 'a'


        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QGridLayout(w)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form, 0, 0, 1, 1)



        # --- Scan selector (minimal: one dropdown + generic scan boxes) ---
        self.scan_param = QtWidgets.QComboBox()
        self.scan_param.addItem("866 OP detuning", userData="op_detuning")
        self.scan_param.addItem("Pump to stretch length", userData="pump_stretch_len")
        self.scan_param.addItem("Pump to stretch power (866 OP amp)", userData="pump_stretch_power")

        # Generic scan settings (REUSE old op_det_* widgets; relabel dynamically)
        self.op_det_start = QtWidgets.QDoubleSpinBox(); self.op_det_start.setDecimals(6); self.op_det_start.setRange(-1e6, 1e6); self.op_det_start.setValue(-10.0)
        self.op_det_end = QtWidgets.QDoubleSpinBox();   self.op_det_end.setDecimals(6);   self.op_det_end.setRange(-1e6, 1e6);   self.op_det_end.setValue(10.0)
        self.op_det_step = QtWidgets.QDoubleSpinBox();  self.op_det_step.setDecimals(6);  self.op_det_step.setRange(1e-6, 1e6);   self.op_det_step.setValue(5.0)

        self._scan_start_lbl = QtWidgets.QLabel("Scan start [MHz]")
        self._scan_end_lbl = QtWidgets.QLabel("Scan end [MHz]")
        self._scan_step_lbl = QtWidgets.QLabel("Scan step [MHz]")

        # Fixed 866 OP detuning (only used when NOT scanning detuning)
        self.fixed_op_det = QtWidgets.QDoubleSpinBox()
        self.fixed_op_det.setDecimals(6); self.fixed_op_det.setRange(-1e6, 1e6); self.fixed_op_det.setValue(float((stretch.get("detunings") or {}).get("866 OP", 0.0)))
        self._fixed_op_det_lbl = QtWidgets.QLabel("866 OP detuning (fixed) [MHz]")

        # Probe settings
        self.res_plus = QtWidgets.QDoubleSpinBox();  self.res_plus.setDecimals(6);  self.res_plus.setRange(-1e6, 1e6); self.res_plus.setValue(resonance_plus)
        self.res_minus = QtWidgets.QDoubleSpinBox(); self.res_minus.setDecimals(6); self.res_minus.setRange(-1e6, 1e6); self.res_minus.setValue(resonance_minus)
        self.probe_power = QtWidgets.QDoubleSpinBox(); self.probe_power.setDecimals(4); self.probe_power.setRange(0.0, 1.0); self.probe_power.setSingleStep(0.01); self.probe_power.setValue(1.0)
        self.probe_len = QtWidgets.QDoubleSpinBox(); self.probe_len.setDecimals(3); self.probe_len.setRange(0.0, 1e7); self.probe_len.setSingleStep(0.5); self.probe_len.setValue(6.0)

        # Pump parameters
        self.amp_866op = QtWidgets.QDoubleSpinBox(); self.amp_866op.setDecimals(4); self.amp_866op.setRange(0.0, 1.0); self.amp_866op.setSingleStep(0.01); self.amp_866op.setValue(_preset_amp(stretch, "866 OP"))
        self.amp_397b = QtWidgets.QDoubleSpinBox();  self.amp_397b.setDecimals(4);  self.amp_397b.setRange(0.0, 1.0);  self.amp_397b.setSingleStep(0.01);  self.amp_397b.setValue(_preset_amp(stretch, "397b"))
        self.amp_854sp1 = QtWidgets.QDoubleSpinBox(); self.amp_854sp1.setDecimals(4); self.amp_854sp1.setRange(0.0, 1.0); self.amp_854sp1.setSingleStep(0.01); self.amp_854sp1.setValue(_preset_amp(stretch, "854 SP1"))
        self.amp_850rp = QtWidgets.QDoubleSpinBox();  self.amp_850rp.setDecimals(4);  self.amp_850rp.setRange(0.0, 1.0);  self.amp_850rp.setSingleStep(0.01);  self.amp_850rp.setValue(_preset_amp(ground, "850 RP"))
        self.pump_stretch_len = QtWidgets.QDoubleSpinBox(); self.pump_stretch_len.setDecimals(3); self.pump_stretch_len.setRange(0.0, 1e6); self.pump_stretch_len.setValue(float(stretch.get("duration", 0.0)))
        self.pump_ground_len = QtWidgets.QDoubleSpinBox();  self.pump_ground_len.setDecimals(3);  self.pump_ground_len.setRange(0.0, 1e6);  self.pump_ground_len.setValue(float(ground.get("duration", 0.0)))

        # Runner/builder settings
        self.external_trigger = QtWidgets.QCheckBox("External trigger"); self.external_trigger.setChecked(False)

        # NEW: checkbox was referenced later but never created
        self.quench_before_probe = QtWidgets.QCheckBox("Quench metastables before probe")
        self.quench_before_probe.setChecked(False)
        self.quench_before_probe.setToolTip(
            'If enabled, inserts exp_sequence.load_section("quench_metastables") between "Pump To Ground" and "Wait Before Probe".'
        )

        self.trap_depth = QtWidgets.QDoubleSpinBox(); self.trap_depth.setDecimals(3); self.trap_depth.setRange(0.0, 1.0); self.trap_depth.setSingleStep(0.01); self.trap_depth.setValue(0.8)
        self.n_runs = QtWidgets.QSpinBox(); self.n_runs.setRange(1, 999); self.n_runs.setValue(3)
        self.n_cycles = QtWidgets.QSpinBox(); self.n_cycles.setRange(1, 10_000_000); self.n_cycles.setValue(500)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False)
        self.save_btn = QtWidgets.QPushButton("Save scan CSV"); self.save_btn.setEnabled(False)
        self.save_preset_btn = QtWidgets.QPushButton("Save OP presets...")
        self.save_preset_btn.clicked.connect(self._save_OP_presets)


        self.status = QtWidgets.QLabel("Idle.")

        form.addRow("Scan", self.scan_param)
        form.addRow("", self.external_trigger)

        form.addRow(self._scan_start_lbl, self.op_det_start)
        form.addRow(self._scan_end_lbl, self.op_det_end)
        form.addRow(self._scan_step_lbl, self.op_det_step)
        form.addRow(self._fixed_op_det_lbl, self.fixed_op_det)

        form.addRow("Resonance m=+1/2 [MHz]", self.res_plus)
        form.addRow("Resonance m=-1/2 [MHz]", self.res_minus)
        form.addRow("Probe power [0..1]", self.probe_power)
        form.addRow("Probe length [µs]", self.probe_len)

        form.addRow("866 OP Amplitude [0..1]", self.amp_866op)
        form.addRow("397b Amplitude [0..1]", self.amp_397b)
        form.addRow("854 SP1 Power [0..1]", self.amp_854sp1)
        form.addRow("850 RP Power [0..1]", self.amp_850rp)
        form.addRow("Pump to stretch length [µs]", self.pump_stretch_len)
        form.addRow("Pump to ground length [µs]", self.pump_ground_len)
        form.addRow("", self.quench_before_probe)  # NEW

        form.addRow("Trap depth", self.trap_depth)
        form.addRow("N runs (per point, per scenario)", self.n_runs)
        form.addRow("N cycles (builder)", self.n_cycles)
        form.addRow("", self.start_btn)
        form.addRow("", self.stop_btn)
        form.addRow("", self.save_btn)
        form.addRow("", self.save_preset_btn)
        form.addRow("Status", self.status)

        # Plot
        pg.setConfigOptions(antialias=True)
        self.plots = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plots, 0, 1, 1, 1)

        self.p = self.plots.addPlot(title="P_dark vs scan variable")
        self.p.setLabel("bottom", "scan variable")
        self.p.setLabel("left", "P_dark")
        self.p.showGrid(x=True, y=True)
        self.p.addLegend()

        self.curve_plus = self.p.plot([], [], pen=pg.mkPen(color=(220, 50, 50), width=2), symbol="o", symbolSize=6, name="Probe @ m=+1/2")
        self.curve_minus = self.p.plot([], [], pen=pg.mkPen(color=(50, 120, 255), width=2), symbol="t", symbolSize=7, name="Probe @ m=-1/2")
        self.curve_eff = self.p.plot([], [], pen=pg.mkPen(color=(60, 180, 75), width=2), symbol="s", symbolSize=6, name="OP_Efficiency")

        self.err_plus = pg.ErrorBarItem(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=0.0)
        self.err_minus = pg.ErrorBarItem(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=0.0)
        self.err_eff = pg.ErrorBarItem(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=0.0)
        self.p.addItem(self.err_plus)
        self.p.addItem(self.err_minus)
        self.p.addItem(self.err_eff)

        # Data buffers
        self._x_plus, self._y_plus, self._e_plus = [], [], []
        self._x_minus, self._y_minus, self._e_minus = [], [], []
        self._x_eff, self._y_eff, self._e_eff = [], [], []
        self._by_det = {}  # reused as "by x_value"

        self._thread = None
        self._worker = None

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.save_btn.clicked.connect(self.save_scan_csv)
        self.scan_param.currentIndexChanged.connect(self._update_scan_ui)
        self._update_scan_ui()

    def _update_scan_ui(self):
        mode = self.scan_param.currentData()
        enable_fixed_det = (mode != "op_detuning")
        self._fixed_op_det_lbl.setEnabled(True)
        self.fixed_op_det.setEnabled(enable_fixed_det)

        # Enable/disable the fixed widgets that become scanned (to reduce ambiguity)
        self.pump_stretch_len.setEnabled(mode != "pump_stretch_len")
        self.amp_866op.setEnabled(mode != "pump_stretch_power")

        if mode == "op_detuning":
            self._scan_start_lbl.setText("Scan start [MHz]")
            self._scan_end_lbl.setText("Scan end [MHz]")
            self._scan_step_lbl.setText("Scan step [MHz]")
            for sb in (self.op_det_start, self.op_det_end, self.op_det_step):
                sb.setDecimals(6); sb.setRange(-1e6, 1e6)
            self.op_det_step.setRange(1e-6, 1e6)
            self.p.setTitle("P_dark vs 866 OP detuning")
            self.p.setLabel("bottom", "866 OP detuning (MHz)")

        elif mode == "pump_stretch_len":
            self._scan_start_lbl.setText("Scan start [µs]")
            self._scan_end_lbl.setText("Scan end [µs]")
            self._scan_step_lbl.setText("Scan step [µs]")
            for sb in (self.op_det_start, self.op_det_end, self.op_det_step):
                sb.setDecimals(3); sb.setRange(0.0, 1e6)
            self.op_det_step.setRange(1e-6, 1e6)
            self.p.setTitle("P_dark vs pump-to-stretch length")
            self.p.setLabel("bottom", "Pump to stretch length (µs)")

        elif mode == "pump_stretch_power":
            self._scan_start_lbl.setText("Scan start [0..1]")
            self._scan_end_lbl.setText("Scan end [0..1]")
            self._scan_step_lbl.setText("Scan step")
            for sb in (self.op_det_start, self.op_det_end, self.op_det_step):
                sb.setDecimals(4); sb.setRange(0.0, 1.0)
            self.op_det_step.setRange(1e-6, 1.0)
            self.p.setTitle("P_dark vs pump-to-stretch power (866 OP amp)")
            self.p.setLabel("bottom", "866 OP amplitude (0..1)")

    def _params(self):
        return dict(
            scan_param=str(self.scan_param.currentData()),
            scan_start=float(self.op_det_start.value()),
            scan_end=float(self.op_det_end.value()),
            scan_step=float(self.op_det_step.value()),
            fixed_op_detuning_mhz=float(self.fixed_op_det.value()),

            external_trigger=bool(self.external_trigger.isChecked()),
            res_m_plus_mhz=float(self.res_plus.value()),
            res_m_minus_mhz=float(self.res_minus.value()),
            probe_power=float(self.probe_power.value()),
            probe_len_us=float(self.probe_len.value()),
            amp_866op=float(self.amp_866op.value()),
            amp_397b=float(self.amp_397b.value()),
            amp_854sp1=float(self.amp_854sp1.value()),
            amp_850rp=float(self.amp_850rp.value()),
            pump_to_stretch_len_us=float(self.pump_stretch_len.value()),
            pump_to_ground_len_us=float(self.pump_ground_len.value()),
            quench_before_probe=bool(self.quench_before_probe.isChecked()),  # NEW
            trap_depth=float(self.trap_depth.value()),
            n_runs=int(self.n_runs.value()),
            n_cycles=int(self.n_cycles.value()),
        )

    @Slot()
    def start(self):
        self._x_plus, self._y_plus, self._e_plus = [], [], []
        self._x_minus, self._y_minus, self._e_minus = [], [], []
        self._x_eff, self._y_eff, self._e_eff = [], [], []
        self._by_det = {}

        self.curve_plus.setData([], [])
        self.curve_minus.setData([], [])
        self.curve_eff.setData([], [])
        self.err_plus.setData(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=0.0)
        self.err_minus.setData(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=0.0)
        self.err_eff.setData(x=np.array([]), y=np.array([]), top=np.array([]), bottom=np.array([]), beam=0.0)

        params = self._params()
        beam = 0.6 * float(params["scan_step"])
        self.err_plus.opts["beam"] = beam
        self.err_minus.opts["beam"] = beam
        self.err_eff.opts["beam"] = beam

        self.status.setText("Starting...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)

        self._thread = QtCore.QThread(self)
        self._worker = OpticalPumpingWorker(params)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.point.connect(self.on_point)
        self._worker.status.connect(self.status.setText)
        self._worker.failed.connect(self.on_failed)
        self._worker.finished.connect(self.on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    @Slot(float, str, float, float, int)
    def on_point(self, x_value, scenario, p_dark, stderr, n_valid):
        x_value = float(x_value)
        p_dark = float(p_dark)
        stderr = float(stderr)
        n_valid = int(n_valid)

        rec = self._by_det.setdefault(x_value, {})

        if scenario == "m_plus":
            self._x_plus.append(x_value)
            self._y_plus.append(p_dark)
            self._e_plus.append(stderr)
            rec["plus"] = (p_dark, stderr, n_valid)

            x = np.asarray(self._x_plus, dtype=float)
            y = np.asarray(self._y_plus, dtype=float)
            e = np.asarray(self._e_plus, dtype=float)
            self.curve_plus.setData(x, y)
            top = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
            self.err_plus.setData(x=x, y=y, top=top, bottom=top, beam=self.err_plus.opts.get("beam", 0.0))

        elif scenario == "m_minus":
            self._x_minus.append(x_value)
            self._y_minus.append(p_dark)
            self._e_minus.append(stderr)
            rec["minus"] = (p_dark, stderr, n_valid)

            x = np.asarray(self._x_minus, dtype=float)
            y = np.asarray(self._y_minus, dtype=float)
            e = np.asarray(self._e_minus, dtype=float)
            self.curve_minus.setData(x, y)
            top = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
            self.err_minus.setData(x=x, y=y, top=top, bottom=top, beam=self.err_minus.opts.get("beam", 0.0))

        # --- Update efficiency whenever we have both scenarios for any x_value ---
        xs, ys, es = [], [], []
        for det, v in self._by_det.items():
            if "plus" not in v or "minus" not in v:
                continue

            # CHANGED: tuples are now (p_dark, stderr, n_valid)
            B, sB, _nvB = v["plus"]   # P_dark(m=+1/2)
            A, sA, _nvA = v["minus"]  # P_dark(m=-1/2)

            denom = A + B
            if not np.isfinite(denom) or denom <= 0.0:
                continue

            E = A / denom
            # error propagation: dE/dA = B/(A+B)^2 ; dE/dB = -A/(A+B)^2
            sE2 = (B * B * sA * sA + A * A * sB * sB) / (denom ** 4)
            sE = float(np.sqrt(sE2)) if np.isfinite(sE2) and sE2 >= 0.0 else float("nan")

            xs.append(float(det))
            ys.append(float(E))
            es.append(float(sE))

        if xs:
            order = np.argsort(np.asarray(xs, dtype=float))
            x = np.asarray(xs, dtype=float)[order]
            y = np.asarray(ys, dtype=float)[order]
            e = np.asarray(es, dtype=float)[order]
            self.curve_eff.setData(x, y)
            top = np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0)
            self.err_eff.setData(x=x, y=y, top=top, bottom=top, beam=self.err_eff.opts.get("beam", 0.0))

        self.p.enableAutoRange(axis="y", enable=True)

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
        # Enable save if we ended with any data
        self.save_btn.setEnabled(bool(self._by_det))
        self._worker = None
        self._thread = None

    @staticmethod
    def next_opscan_filename(directory=".", pad=3):
        pattern = re.compile(r"^op_scan_(\d+)\.csv$")
        max_n = 0
        try:
            for fname in os.listdir(directory):
                m = pattern.match(fname)
                if m:
                    max_n = max(max_n, int(m.group(1)))
        except FileNotFoundError:
            pass
        n = max_n + 1
        return os.path.join(directory, f"op_scan_{n:0{pad}d}.csv")

    @staticmethod
    def save_op_scan(rows, *, directory="."):
        out_dir = os.path.join(str(directory), "OP_Data")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = OpticalPumpingApp.next_opscan_filename(out_dir)

        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            # CHANGED: do not save any standard errors (recompute from N_valid in post)
            w.writerow([
                "x_value",
                "P_dark_plus", "N_valid_plus",
                "P_dark_minus", "N_valid_minus",
                "OP_efficiency",
            ])
            for r in rows:
                w.writerow([
                    r.get("x_value"),
                    r.get("p_plus"), r.get("nv_plus"),
                    r.get("p_minus"), r.get("nv_minus"),
                    r.get("eff"),
                ])
        print(f"Saved: {out_csv}")
        return out_csv

    @Slot()
    def save_scan_csv(self):
        if not self._by_det:
            return

        rows = []
        xs = sorted(self._by_det.keys())
        for x in xs:
            v = self._by_det.get(x, {})
            p_plus = e_plus = nv_plus = None
            p_minus = e_minus = nv_minus = None
            if "plus" in v:
                p_plus, e_plus, nv_plus = v["plus"]
            if "minus" in v:
                p_minus, e_minus, nv_minus = v["minus"]

            # Compute/save efficiency only (no efficiency stderr)
            eff = None
            if (p_plus is not None) and (p_minus is not None):
                B = float(p_plus)   # P_dark(m=+1/2)
                A = float(p_minus)  # P_dark(m=-1/2)
                denom = A + B
                if np.isfinite(denom) and denom > 0.0:
                    eff = A / denom

            rows.append(dict(
                x_value=float(x),
                p_plus=(None if p_plus is None else float(p_plus)),
                nv_plus=(None if nv_plus is None else int(nv_plus)),
                p_minus=(None if p_minus is None else float(p_minus)),
                nv_minus=(None if nv_minus is None else int(nv_minus)),
                eff=(None if eff is None else float(eff)),
            ))

        try:
            out_csv = self.save_op_scan(rows, directory=os.getcwd())
            self.status.setText(f"Saved: {out_csv}")
        except Exception as e:
            self.status.setText(f"Error saving: {type(e).__name__}: {e}")


    def _pump_to_stretch_preset_func_src(self) -> str:
        try:
            cooling_dict = _load_presets_dict(PRESETS_PATH, "cooling")
            self._cooling_detunings = cooling_dict.get("detunings") or {}
            print("Loaded cooling detunings:", self._cooling_detunings)
        except Exception:
            self._cooling_detunings = {}
        det_866 = float(self.fixed_op_det.value())
        det_397b = float(self._cooling_detunings.get("397b", PRESETS_PATH))
        det_854sp1 = float(self._cooling_detunings.get("854 SP1", PRESETS_PATH))
        duration = float(self.pump_stretch_len.value())
        amp_397b = float(self.amp_397b.value())
        amp_866op = float(self.amp_866op.value())
        amp_854sp1 = float(self.amp_854sp1.value())

        return f"""def pump_to_stretch():
    # Pump to D3/2 m=-3/2 stretch state preset (AUTOGENERATED by Calibrate_Optical_Pumping.py {datetime.now().isoformat()} )  
    return {{
        "detunings": {{ "866 OP": {det_866}, "397b": {det_397b}, "854 SP1": {det_854sp1} }},
        "duration": {duration},
        "pmt_gate_high": True,
        "dds_functions": {{
            "397b": (lambda t, a={amp_397b}: a),
            "866 OP": (lambda t, a={amp_866op}: a),
            "854 SP1": (lambda t, a={amp_854sp1}: a),
        }},
        "phase_functions": {{}},
        "coincidence_detector": False,
    }}
"""
    
    
    def _pump_to_ground_preset_func_src(self) -> str:
        try:
            cooling_dict = _load_presets_dict(PRESETS_PATH, "cooling")
            self._cooling_detunings = cooling_dict.get("detunings") or {}
            print("Loaded cooling detunings:", self._cooling_detunings)
        except Exception:
            self._cooling_detunings = {}
        
        det_854sp1 = float(self._cooling_detunings.get("854 SP1", PRESETS_PATH))
        det_850rp = float(self._cooling_detunings.get("850 RP", PRESETS_PATH))

        det_854sp1 = float(self._cooling_detunings.get("854 SP1", PRESETS_PATH))
        det_850rp = float(self._cooling_detunings.get("850 RP", PRESETS_PATH))
        duration = float(self.pump_ground_len.value())
        amp_850rp = float(self.amp_850rp.value())
        amp_854sp1 = float(self.amp_854sp1.value())

        return f"""
def pump_to_ground():
    # Pump from D3/2 m=-3/2 to S1/2 m=-1/2 preset (AUTOGENERATED by Calibrate_Optical_Pumping.py {datetime.now().isoformat()} )
    return {{
        "detunings": {{ "854 SP1": {det_854sp1}, "850 RP": {det_850rp} }},
        "duration": {duration},
        "pmt_gate_high": True,
        "dds_functions": {{
            "850 RP": (lambda t, a={amp_850rp}, t0=1: a if float(t) >= t0 else 0.0),
            "854 SP1": (lambda t, a={amp_854sp1}, t0=1: a if float(t) >= t0 else 0.0),
        }},
        "phase_functions": {{}},
        "coincidence_detector": False,
    }}
"""


    def _save_OP_presets(self):
        """
        Upsert measurement() into presets.py AUTOGEN region.
        Assumes presets.py already has BEGIN/END markers (otherwise fails).
        """
        try:
            print("Saving OP presets...")
            func_src_stretch = self._pump_to_stretch_preset_func_src()
            func_src_ground = self._pump_to_ground_preset_func_src()
            print("Generated pump_to_stretch function source:\n", func_src_stretch)
            print("Generated pump_to_ground function source:\n", func_src_ground)
            upsert_autogen_function(PRESETS_PATH, func_src_stretch)
            upsert_autogen_function(PRESETS_PATH, func_src_ground)
            self.status.setText(f"Upserted measurement() and pump_to_ground() into: {PRESETS_PATH}")
        except Exception as e:
            self.status.setText(f"Preset upsert failed: {type(e).__name__}: {e}")



def main():
    app = QtWidgets.QApplication([])
    win = OpticalPumpingApp()
    win.resize(1250, 720)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()