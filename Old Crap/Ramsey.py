from adriq.experiment import *
from adriq.Resonance_Log import *
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from datetime import datetime

# -------------------
# Fixed parameters
# -------------------
t_precess_values = [2,5,10,15,20,30,40,50,60,80]
for t_precess in t_precess_values:
    Power = 1
    resonance = read_resonance(r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\Resonances\Quadrupole_S_1_2_m_-1_2_D_5_2_m+3_2.csv")

    Pulse_Time = 3.2

    # -------------------
    # Hardware setup
    # -------------------
    pulse_sequencer = Pulse_Sequencer()
    dds_dict = load_dds_dict("ram", r"C:\Users\probe\OneDrive - University of Sussex\Desktop\Experiment_Config\dds_config.cfg")

    exp_sequence = Experiment_Builder(
        dds_dict,
        pulse_sequencer,
        ram_step=0.2,
        N_Cycles=500,
        external_trigger=False,
    )

    exp_sequence.set_detunings(detuning_dict={
        "729 Temp1": resonance
    })

    exp_sequence.load_cooling(length=2000)
    exp_sequence.load_trapping()
    exp_sequence.load_section("pump_to_stretch")
    exp_sequence.load_section("pump_to_ground")
    exp_sequence.load_section("quench_metastables") # jut in case

    exp_sequence.create_section(
        name="RamseyBlock",
        duration = math.ceil(2 * Pulse_Time + t_precess),
        dds_functions={
            "729 Temp1": lambda t, a=Power, ton=Pulse_Time: (
                a if float(t) < ton else
                0.0 if float(t) < Pulse_Time + t_precess else
                a if float(t) < Pulse_Time + t_precess + ton else
                0.0
            )
        },
    phase_functions={
        "729 Temp1": lambda t, phase2=0: (
            0.0 if float(t) < Pulse_Time else
            phase2 if float(t) < Pulse_Time + t_precess + Pulse_Time else
            0.0
        )
    },
        pmt_gate_high=True,
    )

    exp_sequence.load_measurement()
    exp_sequence.build_ram_arrays()
    exp_sequence.flash()

    # -------------------
    # Runner
    # -------------------
    exp_runner = Experiment_Runner(
        dds_dict,
        pulse_sequencer,
        timeout=100,
        pmt_threshold=2000,
        expected_fluorescence=8000,
        pulse_expected_fluorescence=4200,
        trigger_mode="ram",
    )

    # -------------------
    # Phase scan
    # -------------------
    phases = np.arange(0, 181, 10)
    p_dark_vals = []
    p_err_vals = []

    for phase in phases:
        print(f"Running phase {phase}°")

        exp_sequence.edit_section(
            "RamseyBlock",
            phase_functions={"729 Temp1": lambda t, p=phase: 0 if float(t) < Pulse_Time + t_precess else p},
        )

        exp_sequence.build_ram_arrays()
        exp_sequence.flash()

        exp_runner.start_experiment(N=5)

        n_valid = int(exp_runner.N_Valid_Pulses or 0)
        n_bright = int(exp_runner.bright_counts or 0)

        if n_valid > 0:
            p_bright = n_bright / n_valid
            p_dark = 1.0 - p_bright
            err = np.sqrt(p_dark * (1.0 - p_dark) / n_valid)
        else:
            p_dark = 0.0
            err = np.nan
        P_dark_corrected, stderr_corrected, _, _ = correct_probabilities_with_spam(p_dark, err, p_bright, err)
        p_dark_vals.append(P_dark_corrected)
        p_err_vals.append(stderr_corrected)

    # Create the directory if it doesn't exist
    os.makedirs("Ramsey_Data", exist_ok=True)

    # Create a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Ramsey_Data/plus_ramsey_tprecess_{t_precess}us_{timestamp}.csv"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase_deg", "p_dark", "stderr"])
        for phase, p_dark, err in zip(phases, p_dark_vals, p_err_vals):
            writer.writerow([phase, p_dark, err])

    print(f"Saved data to {filename}")

    # Plot with error bars
    plt.figure()
    plt.errorbar(phases, p_dark_vals, yerr=p_err_vals, fmt='o-', capsize=4)
    plt.xlabel("Phase (degrees)")
    plt.ylabel("p_dark")
    plt.title("Ramsey Scan")
    plt.grid(True)


plt.show()
