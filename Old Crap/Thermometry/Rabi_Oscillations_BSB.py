import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from adriq.experiment import *
from adriq.Optomechanics import *
import csv
from scipy.optimize import curve_fit
import os
import re

RESONANCE_DETUNING = -23.278
secular_f = 1.714

def run_rabi_point(probe_time):
	# Build a fresh sequence for each probe_time
	pulse_sequencer = Pulse_Sequencer()
	dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")
	exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.08, N_Cycles=1E3)

	# Base settings (fixed)
	exp_sequence.set_detunings(detuning_dict={
		"854 SP1": 0, "854 SP2": 0, "397a": 0, "397c": -18, "866": 0, "866 OP": 0,
		"850 RP": 0, "866 RP": 40, "729 Temp1": RESONANCE_DETUNING+secular_f, "729 Temp2": 0
	})
	exp_sequence.set_trapping_parameters(
		trapping_detuning_dict={"397c": -50},
		trapping_amplitude_dict={"397c": 0.4}
	)
	exp_sequence.create_cooling_section(length=500, amplitude_dict={"397c":0.18,"866 RP":0.4, "854 SP1":1})

	# Optical pumping
	exp_sequence.create_section(name="Pump To Ground", duration=2, dds_functions={
		"866 RP": lambda t: 0.7,
	}, pmt_gate_high=True)

	# Probe with variable duration
	if probe_time > 1:
		exp_sequence.create_section(name="Probe", duration=int(probe_time), dds_functions={
			"729 Temp1": lambda t: 1,
		}, pmt_gate_high=True)

	# Measurement
	exp_sequence.create_measurement_section(Length=1000, DDS=["397c", "866 RP"], pmt_gate_high=True, threshold=4)

	# Program hardware
	exp_sequence.build_ram_arrays()
	exp_sequence.flash()

	# Run and compute P_dark
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
	exp_runner.experiment_trap_depth = 0.6
	exp_runner.clear_channels()
	exp_runner.start_experiment(N=3)

	N_bright = exp_runner.bright_counts
	P_bright = N_bright / (exp_runner.N_Valid_Pulses or 1)
	P_dark = 1 - P_bright
	return P_dark

def main():
	probe_times = list(range(0, 31, 2))  # 0..44 inclusive step 2
	P_dark_values = []

	# Initialize live plot
	plt.ion()
	fig, ax = plt.subplots()
	line, = ax.plot([], [], marker='o', label='Data')
	ax.set_xlabel("Probe time (µs)")
	ax.set_ylabel("P_dark")
	ax.set_title("Rabi oscillations at Δ = -23.24 MHz")
	ax.grid(True)
	plt.show(block=False)

	for t in tqdm(probe_times, desc="Rabi scan (varying probe time)"):
		P_dark_values.append(run_rabi_point(t))
		# Live update
		line.set_data(probe_times[:len(P_dark_values)], P_dark_values)
		ax.relim()
		ax.autoscale_view()
		fig.canvas.draw_idle()
		fig.canvas.flush_events()
		plt.pause(0.01)

	# Fit to A * exp(-t/τ) * sin(ω t + φ) + C
	def decaying_sin(t, A, tau, omega, C):
		return A * np.exp(-t / tau) * np.sin(omega * t) + C

	t_data = np.array(probe_times, dtype=float)
	y_data = np.array(P_dark_values, dtype=float)

	A0 = (np.max(y_data) - np.min(y_data)) / 2 or 0.1
	tau0 = 50.0
	omega0 = 2 * np.pi / 40.0
	C0 = np.mean(y_data)

	try:
		popt, pcov = curve_fit(decaying_sin, t_data, y_data, p0=[A0, tau0, omega0, C0], maxfev=5000)
	except Exception:
		popt = [A0, tau0, omega0, C0]

	# Determine next sequential filename: rabi_oscillations_n.csv
	def next_rabi_filename(directory="."):
		pattern = re.compile(r"^rabi_oscillations_(\d+)\.csv$")
		max_n = 0
		for fname in os.listdir(directory):
			m = pattern.match(fname)
			if m:
				max_n = max(max_n, int(m.group(1)))
		return os.path.join(directory, f"rabi_oscillations_{max_n + 1}.csv")

	# Save CSV with header containing fit params
	out_csv = next_rabi_filename(".")
	with open(out_csv, "w", newline="") as f:
		writer = csv.writer(f)
		# include fit parameters in header comments row
		writer.writerow([
			"# fit: A, tau, omega, C = "
			f"{popt[0]:.6g}, {popt[1]:.6g}, {popt[2]:.6g}, {popt[3]:.6g}"
		])
		writer.writerow(["probe_time", "P_dark"])
		for t, p in zip(probe_times, P_dark_values):
			writer.writerow([t, p])
	print(f"Saved: {out_csv}")

	# Plot fitted curve on the same live figure
	t_fit = np.linspace(min(probe_times), max(probe_times), 500)
	ax.plot(t_fit, decaying_sin(t_fit, *popt), '-', label='Fit')
	ax.legend()

	# Finalize plot display
	plt.ioff()
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	main()
