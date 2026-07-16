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


exp_sequence.set_detunings(detuning_dict={"854 SP1": 0, "854 SP2": 0, "397b": 0, "397c": -20, "866S": 0, "866 OP": 30, "850 RP": 0, "866 RP": 40, "729 t1": 0, "729 t2": 0})
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
    "866 OP": lambda t: 0.2,
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
exp_runner.start_experiment(N=80)

# Replace the first histogram with a fitted exponential + offset
first_lower, first_upper, bins = 0, 50, 200
diffs_dict = exp_runner.get_time_diffs(mode="signal-f", lower_cutoff=first_lower, upper_cutoff=first_upper)
all_diffs = np.concatenate(list(diffs_dict.values())) if diffs_dict else np.array([])

if all_diffs.size > 0:
    # Histogram of counts (not density)
    counts, edges = np.histogram(all_diffs, bins=bins, range=(first_lower, first_upper))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]

    # Same model as in Cavity_Ringdown.py, but with an added constant offset
    def delayed_exponential_decay(t, A, tau, delay, offset):
        t = np.asarray(t, dtype=float)
        return np.where(
            t < delay,
            0,
            A * np.exp(-(t - delay) * 2.0 * np.pi * tau) + offset
        )

    # Initial guesses
    offset0 = float(np.min(counts))
    A0 = max(float(np.max(counts) - offset0), 1.0)
    tau0 = 0.5  # MHz
    delay0 = float(centers[int(np.argmax(counts))])

    try:
        popt, pcov = curve_fit(
            delayed_exponential_decay,
            centers,
            counts,
            p0=[A0, tau0, delay0, offset0],
            bounds=(
                [0.0, 0.0, first_lower, 0.0],
                [np.inf, np.inf, first_upper, np.inf],
            ),
            maxfev=20000,
        )
    except Exception as e:
        print("Delayed exponential fit failed:", e)
        popt, pcov = [A0, tau0, delay0, offset0], None

    A_fit, tau_fit, delay_fit, offset_fit = popt
    tau_fit_us = (1.0 / (2.0 * np.pi * tau_fit)) if tau_fit > 0 else float("nan")

    print("\n--- Delayed exponential fit (ringdown model) ---")
    print(f"A      = {A_fit:.6g} counts/bin")
    print(f"tau  = {tau_fit:.6g} MHz")
    print(f"delay  = {delay_fit:.6g} us")
    print(f"offset = {offset_fit:.6g} counts/bin")
    print(f"tau    = {tau_fit_us:.6g} us   (tau = 1/(2π·tau))")

    # Plot data + fit
    plt.figure(figsize=(10, 6))
    plt.bar(centers, counts, width=bin_width, alpha=0.6, edgecolor="black", label="Data")
    t_plot = np.linspace(first_lower, first_upper, 800)
    plt.plot(
        t_plot,
        delayed_exponential_decay(t_plot, *popt),
        "r-",
        lw=2,
        label=f"Fit: tau={tau_fit:.3g} MHz, delay={delay_fit:.3g} us (tau={tau_fit_us:.3g} us)",
    )
    plt.title("PMT counts (signal-f) with delayed exponential fit")
    plt.xlabel("Time Difference (us)")
    plt.ylabel("Counts per bin")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No signal-f data found in the first window (0–30 us). Skipping delayed-exponential fit.")