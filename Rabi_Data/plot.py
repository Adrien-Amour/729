import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import csv
import ast  # needed for ast.literal_eval in read_rabi_csv

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
			# meta row written as: ["# meta", "{...}"] or ["# meta", "{'k':1}"]
			if row[0].strip() == "# meta":
				if len(row) >= 2:
					try:
						meta = ast.literal_eval(row[1])
					except Exception:
						meta = {"raw": row[1]}
				continue
			# normal rows (header + data)
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

	# If stderr missing/NaN, recompute from Bernoulli/binomial using N_valid.
	if np.all(~np.isfinite(perr)) or np.nanmax(np.abs(perr)) == 0.0:
		perr = _binomial_stderr(p, np.maximum(n_valid, 1))

	return meta, t_us, p, perr, n_valid
# -------------------------------------------------
# Rabi model: on-resonance with decoherence
# -------------------------------------------------
def rabi_p(t, A0, Gamma, Omega):
    p = A0/2 * (1-np.exp(-Gamma * t) * np.cos( Omega * t))
    return np.clip(p, 1e-9, 1 - 1e-9)

# -------------------------------------------------
# Binomial negative log-likelihood
# -------------------------------------------------
def neg_log_likelihood(params, t, N_dark, N_valid):
    A0, Gamma, Omega = params
    p = rabi_p(t, A0, Gamma, Omega)
    return -np.sum(
        N_dark * np.log(p) +
        (N_valid - N_dark) * np.log(1 - p)
    )

# -------------------------------------------------
# Initial Omega guess from pi-time
# -------------------------------------------------
def omega_guess_from_pi_time(t_us, P_dark):
    idx = np.argmax(P_dark)
    t_pi = t_us[idx] * 1e-6
    return np.pi / t_pi

# -------------------------------------------------
# Fit wrapper using your CSV loader
# -------------------------------------------------
def fit_rabi_from_csv(path):
    meta, t_us, P_dark, _, N_valid = read_rabi_csv(path)

    t = t_us * 1e-6
    N_dark = np.round(P_dark * N_valid).astype(int)

    # ---- initial guesses ----
    A0_guess = P_dark.max()
    Gamma_guess = 5e4       # ~100 µs coherence
    Omega_guess = omega_guess_from_pi_time(t_us, P_dark)

    x0 = [A0_guess, Gamma_guess, Omega_guess]

    bounds = [
        (0.83, 0.85),            # A0
        (0.0, 5e5),            # Gamma (s^-1)
        (0.0, 2*np.pi*50e6),   # Omega (rad/s)
    ]

    res = minimize(
        neg_log_likelihood,
        x0,
        args=(t, N_dark, N_valid),
        bounds=bounds,
        method="L-BFGS-B"
    )

    return meta, t_us, P_dark, N_valid, res

# -------------------------------------------------
# Plot helper
# -------------------------------------------------
def plot_rabi_fit(t_us, P_dark, N_valid, res):
    P_err = np.sqrt(P_dark * (1 - P_dark) / N_valid)

    t_fit = np.linspace(t_us.min(), t_us.max(), 3000)
    p_fit = rabi_p(t_fit * 1e-6, *res.x)

    plt.errorbar(
        t_us, P_dark, yerr=P_err,
        fmt='o', capsize=3, label="data"
    )
    plt.plot(t_fit, p_fit, label="fit")

    plt.xlabel("Probe time (µs)")
    plt.ylabel("P_dark")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# Example call
# -------------------------------------------------
if __name__ == "__main__":

    path = "rabi_009.csv"
    meta, t_us, P_dark, N_valid, res = fit_rabi_from_csv(path)
	plt.plot(t_us, P_dark, marker='o', linestyle='-', label="data")
	plt.axhline(0.98, color='r', linestyle='--', label="state prep efficiency")
	plt.xlabel("Probe time (µs)")
	plt.ylabel("P_dark")
	plt.legend()
	plt.tight_layout()
	plt.show()

    A0, Gamma, Omega = res.x

    print("Fit results:")
    print(f"A0     = {A0:.4f}")
    print(f"Gamma  = {Gamma:.2e} s^-1   (T2* ≈ {1/Gamma*1e6:.1f} µs)")
    print(f"Omega  = {Omega/(2*np.pi):.3f} Hz")
    print(f"π-time = {np.pi/Omega*1e6:.2f} µs")

    plot_rabi_fit(t_us, P_dark, N_valid, res)
