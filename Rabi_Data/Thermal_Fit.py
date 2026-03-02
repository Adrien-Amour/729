import numpy as np
import matplotlib.pyplot as plt
import csv
import ast

# --------------------------
# Config
ratio = 6
estimated_tilt = np.degrees(np.arctan(0.3/2))

CSV_PATH = "rabi_007.csv"  # TODO: set your data file name/path

# --------------------------
# CSV loader (copied from plot.py)
def _binomial_stderr(p, n):
    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=float)
    n = np.maximum(n, 1.0)
    return np.sqrt(p * (1.0 - p) / n)

def read_rabi_csv(path):
    """
    Reads the CSV written by Rabi_Oscillations.py:
      ['# meta', {dict-as-string}]
      probe_time_us,P_dark,P_dark_stderr,N_valid
    Returns: meta(dict), t_us, p, perr, n_valid
    """
    meta = {}
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            if row[0].strip() == "# meta":
                if len(row) >= 2:
                    try:
                        meta = ast.literal_eval(row[1])
                    except Exception:
                        meta = {"raw": row[1]}
                continue
            rows.append(row)

    if not rows:
        raise ValueError(f"No data rows found in {path}")

    header = [h.strip() for h in rows[0]]
    data = rows[1:]

    def col(name):
        try:
            return header.index(name)
        except ValueError as e:
            raise ValueError(f"Missing column {name!r} in {path}. Header={header}") from e

    i_t = col("probe_time_us")
    i_p = col("P_dark")
    i_e = header.index("P_dark_stderr") if "P_dark_stderr" in header else None
    i_n = col("N_valid") if "N_valid" in header else None

    t_us, p, perr, n_valid = [], [], [], []
    for row in data:
        if not row:
            continue
        t_us.append(float(row[i_t]))
        p.append(float(row[i_p]))
        perr.append(float(row[i_e]) if i_e is not None and row[i_e] != "" else float("nan"))
        n_valid.append(int(row[i_n]) if i_n is not None and row[i_n] != "" else 0)

    t_us = np.asarray(t_us, dtype=float)
    p = np.asarray(p, dtype=float)
    n_valid = np.asarray(n_valid, dtype=int)
    perr = np.asarray(perr, dtype=float)

    if np.all(~np.isfinite(perr)) or np.nanmax(np.abs(perr)) == 0.0:
        perr = _binomial_stderr(p, np.maximum(n_valid, 1))

    return meta, t_us, p, perr, n_valid

# --------------------------
# Physical constants
hbar = 1.055e-34
kB = 1.38e-23
mass = 40 * 1.6605e-27  # 40Ca+ mass [kg]

# Laser
lamb = 729e-9  # 729 nm
k = 2 * np.pi / lamb

# --------------------------
# Mode 2 parameters
T2 = 1e-3          # Kelvin
f2 = 2.3e6         # Hz
theta2 = estimated_tilt  # degrees

# Mode 1 parameters
T1 = ratio * T2    # Kelvin
f1 = 2.14e6        # Hz
theta1 = 90 - estimated_tilt  # degrees

# --------------------------
# Functions to calculate Lamb-Dicke parameter and thermal occupation
def calc_eta(frequency, angle_deg):
    omega = 2 * np.pi * frequency
    k_z = k * np.cos(np.deg2rad(angle_deg))
    z_0 = np.sqrt(hbar / (2 * mass * omega))
    return k_z * z_0

def calc_nbar(frequency, temperature):
    omega = 2 * np.pi * frequency
    return 1 / (np.exp(hbar * omega / (kB * temperature)) - 1)

eta1 = calc_eta(f1, theta1)
eta2 = calc_eta(f2, theta2)
nbar1 = calc_nbar(f1, T1)
nbar2 = calc_nbar(f2, T2)

print(f"Mode 1: eta={eta1:.4f}, nbar={nbar1:.2f}")
print(f"Mode 2: eta={eta2:.4f}, nbar={nbar2:.2f}")

# --------------------------
# Rabi flopping parameters
Omega0 = 2 * np.pi * 80e3  # bare Rabi frequency [rad/s]
nmax1 = 500
nmax2 = 200

# Thermal distributions
n1 = np.arange(nmax1)
log_p1 = n1 * np.log(nbar1) - (n1 + 1) * np.log(nbar1 + 1)
p1 = np.exp(log_p1)

n2 = np.arange(nmax2)
log_p2 = n2 * np.log(nbar2) - (n2 + 1) * np.log(nbar2 + 1)
p2 = np.exp(log_p2)

# Time array
t = np.linspace(0, 50e-6, 1000)  # seconds

# --------------------------
# Rabi flopping model
Pe = np.zeros_like(t)
for i in range(nmax1):
    for j in range(nmax2):
        Omega_n = Omega0 * (1 - eta1**2 * i - eta2**2 * j)
        Pe += 0.8 * p1[i] * p2[j] * np.sin(Omega_n * t / 2) ** 2

# --------------------------
# Load experimental data (same as plot.py) and overlay
meta, t_us_data, P_dark_data, P_dark_err, N_valid = read_rabi_csv(CSV_PATH)

plt.errorbar(
    t_us_data, P_dark_data, yerr=P_dark_err,
    fmt="o", capsize=3, label="data (P_dark)"
)
plt.plot(t * 1e6, Pe, label="thermal model (Pe)")

plt.xlabel("Time [µs]")
plt.ylabel("Population")
plt.title("Rabi flopping: data overlay on thermal model")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()