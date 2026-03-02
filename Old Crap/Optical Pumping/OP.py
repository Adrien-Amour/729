from adriq.experiment import *
from adriq.Optomechanics import *
from adriq.tdc_functions import filter_trailing_zeros, compute_time_diffs, filter_runs
from tqdm import tqdm
import matplotlib.pyplot as plt  # added
import numpy as np  # added
from scipy.optimize import curve_fit  # added


dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

pulse_sequencer = Pulse_Sequencer()

exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=0.08*2,  N_Cycles=5000)


exp_sequence.set_detunings(detuning_dict={"854 SP1": 0, "854 SP2": 0, "397b": 0, "397c": -20, "866": 0, "866 OP": 30, "850 RP": 0, "866 RP": 40, "729 Temp1": 0, "729 Temp2": 0})
exp_sequence.set_trapping_parameters(
    trapping_detuning_dict={"397c": -50},
    trapping_amplitude_dict={"397c": 0.4}
)
exp_sequence.load_cooling(length=100)
# Create a Gaussian output function for DDS 397a

# Create op section 
exp_sequence.create_section(name="Wait", duration=10, dds_functions={})
exp_sequence.create_section(name="Pump To Stretch", duration=100, dds_functions={
    "397b": lambda t: 0.4,
    "866 OP": lambda t: 0.5,
    "854 SP1": lambda t: 0.2,
}, pmt_gate_high=True) #First stage of Optical Pumping

exp_sequence.create_section(name="Pump To Ground", duration=5, dds_functions={
    "850 RP":  lambda t: 0.35 if t >= 1.0 else 0.0,
    "854 SP1": lambda t: 0.20 if t >= 1.0 else 0.0,
}, pmt_gate_high=True) #First stage of Optical Pumping


exp_sequence.build_ram_arrays()
exp_sequence.flash()

exp_runner = Experiment_Runner(
    dds_dict,
    pulse_sequencer,
    timeout=100,
    pmt_threshold=2000,
    expected_fluorescence= 8000,
    pulse_expected_fluorescence= 3000,
    sp_threshold=None,
    load_timeout = 100,
    trigger_mode="ram",
    cavity_lock=False,
)
exp_runner.start_experiment(N=30)

# Replace the first histogram with a fitted exponential + offset
first_lower, first_upper, bins = 0, 30, 200
diffs_dict = exp_runner.get_time_diffs(mode="signal-f", lower_cutoff=first_lower, upper_cutoff=first_upper)
all_diffs = np.concatenate(list(diffs_dict.values())) if diffs_dict else np.array([])

if all_diffs.size > 0:
	# Histogram
	counts, edges = np.histogram(all_diffs, bins=bins, range=(first_lower, first_upper))
	centers = 0.5 * (edges[:-1] + edges[1:])
	width = edges[1] - edges[0]

	# Model: A * exp(-t / tau) + C
	def exp_offset(t, A, tau, C):
		return A * np.exp(-t / tau) + C

	# Initial guesses
	A0 = max(float(np.max(counts) - np.min(counts)), 1.0)
	tau0 = max((first_upper - first_lower) / 3.0, 1e-3)
	C0 = float(np.min(counts))

	try:
		popt, _ = curve_fit(
			exp_offset,
			centers,
			counts,
			p0=[A0, tau0, C0],
			bounds=([0.0, 1e-6, 0.0], [np.inf, np.inf, np.inf]),
			maxfev=10000
		)
	except Exception:
		popt = [A0, tau0, C0]

	# Plot histogram and fit
	plt.figure(figsize=(10, 6))
	plt.bar(centers, counts, width=width, alpha=0.6, edgecolor='black', label='Data (0–30 μs)')
	t_fit = np.linspace(first_lower, first_upper, 400)
	plt.plot(t_fit, exp_offset(t_fit, *popt), 'r-', lw=2, label=f'Fit: A={popt[0]:.2g}, τ={popt[1]:.2g} μs, C={popt[2]:.2g}')
	plt.title('PMT counts (signal-f) with exponential fit (first section)')
	plt.xlabel('Time Difference (μs)')
	plt.ylabel('Counts per bin')
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.show()
else:
	print("No signal-f data found in the first window (0–30 μs). Skipping fit.")

# Keep the second histogram as-is
exp_runner.plot_time_diffs_histogram(
	mode="signal-f",
	lower_cutoff=99.5,
	upper_cutoff=105,
	n_bins=200
)