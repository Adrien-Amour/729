from adriq.experiment import *
import csv
import numpy as np
import matplotlib.pyplot as plt
def build_experiment_sequence(mode, pi_amplitude_plus=0.96, pi_amplitude_minus=0.36, wait_duration=10):

    dds_dict = load_dds_dict(
        "ram",
        r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg",
    )

    pulse_sequencer = Pulse_Sequencer()
    exp_sequence = Experiment_Builder(dds_dict, pulse_sequencer, ram_step=float(0.1), N_Cycles=float(500))

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


    exp_sequence.load_cooling(length=int(2000))
    exp_sequence.load_trapping()


    if mode == "plus":
        exp_sequence.load_section("pump_to_stretch")
        exp_sequence.load_section("pump_to_ground")
        exp_sequence.load_section("wait", duration=wait_duration)
        exp_sequence.load_section("quench_metastables")
        exp_sequence.load_section("wait")
        exp_sequence.load_section("pi_plus", amplitude=pi_amplitude_plus)

    elif mode == "minus":
        exp_sequence.load_section("pump_to_stretch")
        exp_sequence.load_section("pump_to_ground")
        exp_sequence.load_section("quench_metastables")
        exp_sequence.load_section("wait", duration=wait_duration)
        exp_sequence.load_section("pi_minus", amplitude=pi_amplitude_minus)

    elif mode == "SPAM Test":
        exp_sequence.load_section("pump_to_stretch")
        exp_sequence.load_section("pump_to_ground")
        exp_sequence.load_section("quench_metastables")
        exp_sequence.load_section("wait", duration=10)
        exp_sequence.load_section("pi_plus", amplitude=pi_amplitude_plus)
        exp_sequence.load_section("wait", duration=10)
        exp_sequence.load_section("pi_minus", amplitude = pi_amplitude_minus)
        exp_sequence.load_section("wait", duration=10)



    exp_sequence.load_measurement()
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

    return exp_sequence, exp_runner




# #spam test
# spam_cases = [
#     ("No pi pulses", 0.0, 0.0),
#     ("Plus only", 0.96, 0.0),
#     ("Minus only", 0.0, 0.36),
#     ("Both pi pulses", 0.96, 0.36),
# ]

# spam_results = []
# import copy

# spam_histograms = []

# for label, pi_amplitude_plus, pi_amplitude_minus in spam_cases:
#     exp_sequence, exp_runner = build_experiment_sequence(
#         "SPAM Test",
#         pi_amplitude_plus=pi_amplitude_plus,
#         pi_amplitude_minus=pi_amplitude_minus,
#     )

#     exp_runner.measure_expected_fluorescence()
#     exp_runner.start_experiment(N=10)
#     spam_histograms.append({
#     "label": label,
#     "histogram": copy.deepcopy(exp_runner.measurement_hist),
#     })

#     n_valid = int(exp_runner.N_Valid_Pulses or 0)
#     n_bright = int(exp_runner.bright_counts or 0)

#     if n_valid > 0:
#         p_bright = n_bright / n_valid
#         p_dark = 1.0 - p_bright
#         uncorrected_error = np.sqrt(
#             p_dark * (1.0 - p_dark) / n_valid
#         )

#         _, _, corrected_efficiency, corrected_error = (
#             correct_probabilities_with_spam(
#                 p_bright,
#                 uncorrected_error,
#                 p_dark,
#                 uncorrected_error,
#             )
#         )
#     else:
#         p_dark = np.nan
#         uncorrected_error = np.nan
#         corrected_efficiency = np.nan
#         corrected_error = np.nan

#     spam_results.append({
#         "label": label,
#         "pi_amplitude_plus": pi_amplitude_plus,
#         "pi_amplitude_minus": pi_amplitude_minus,
#         "n_valid": n_valid,
#         "uncorrected_efficiency": p_dark,
#         "uncorrected_error": uncorrected_error,
#         "corrected_efficiency": corrected_efficiency,
#         "corrected_error": corrected_error,
#     })

# print("\nSPAM test results:")
# for result in spam_results:
#     print(
#         f"{result['label']}: "
#         f"uncorrected = "
#         f"{result['uncorrected_efficiency']:.4f} "
#         f"+/- {result['uncorrected_error']:.4f}, "
#         f"corrected = "
#         f"{result['corrected_efficiency']:.4f} "
#         f"+/- {result['corrected_error']:.4f}, "
#         f"N = {result['n_valid']}"
#     )

# fig, axes = plt.subplots(
#     len(spam_histograms),
#     1,
#     figsize=(8, 3 * len(spam_histograms)),
#     squeeze=False,
# )

# for axis, result in zip(axes[:, 0], spam_histograms):
#     histogram = result["histogram"]
#     photon_numbers = sorted(histogram)
#     counts = [histogram[n] for n in photon_numbers]

#     axis.bar(photon_numbers, counts, width=0.8)
#     axis.set_title(result["label"])
#     axis.set_xlabel("Detected photon number")
#     axis.set_ylabel("Occurrences")
#     axis.grid(axis="y", alpha=0.3)

# plt.tight_layout()
# plt.show()

wait_durations = np.concatenate(([2], np.arange(10, 61, 10)))
pulse_amplitudes = {"plus": 0.96, "minus": 0.36}
p_dark_by_mode = {"plus": [], "minus": []}
p_dark_errors_by_mode = {"plus": [], "minus": []}

for mode, amplitude in pulse_amplitudes.items():
    for wait_duration in wait_durations:
        print(f"{mode}, wait duration = {wait_duration}")

        exp_sequence, exp_runner = build_experiment_sequence(
            mode,
            pi_amplitude_plus=amplitude if mode == "plus" else 0,
            pi_amplitude_minus=amplitude if mode == "minus" else 0,
            wait_duration=wait_duration,
        )

        exp_runner.measure_expected_fluorescence()
        exp_runner.start_experiment(N=5)

        n_valid = int(exp_runner.N_Valid_Pulses or 0)
        n_bright = int(exp_runner.bright_counts or 0)

        if n_valid:
            p_bright = n_bright / n_valid
            p_dark = 1.0 - p_bright
            p_dark_error = np.sqrt(
                p_dark * (1.0 - p_dark) / n_valid
            )
        else:
            p_dark = np.nan
            p_dark_error = np.nan

        p_dark_by_mode[mode].append(p_dark)
        p_dark_errors_by_mode[mode].append(p_dark_error)

        print(f"P_dark = {p_dark:.4f} +/- {p_dark_error:.4f}")

plt.figure(figsize=(8, 5))

for mode, colour in [("plus", "tab:blue"), ("minus", "tab:orange")]:
    plt.errorbar(
        wait_durations,
        p_dark_by_mode[mode],
        yerr=p_dark_errors_by_mode[mode],
        fmt="o-",
        capsize=3,
        color=colour,
        label=f"{mode} pi pulse",
    )

plt.xlabel("Wait duration (us)")
plt.ylabel("P_dark")
plt.title("Ground-state population versus wait duration")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()