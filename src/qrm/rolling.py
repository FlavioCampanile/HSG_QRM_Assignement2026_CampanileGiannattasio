"""Rolling-window re-estimation of VaR/ES under M1-M3 (last 500 daily returns)  [Q ix].

At every estimation date t the last WINDOW = 500 daily log-returns are turned into
496 overlapping 5-day log-returns; M1-M3 are re-fitted on them, s_k and d_k are
re-calibrated (the PDs pi_k are kept fixed) and the loss distribution is re-simulated.
The same random numbers are used at every date (common random numbers), so that the
time variation of VaR/ES reflects the data and not Monte Carlo noise.

Two readings are computed side by side:
  * "recalibrated" (baseline, the assignment's hint P[Y_k <= d_k] = pi_k): d_k is re-derived
    in every window, so PDs and EL are constant and only the *shape* of the factor law matters;
  * "fixed barrier" (extension): d_k is frozen at its full-sample value, so that a more
    volatile or bearish window raises the implied PDs (a point-in-time view).
"""
import time
import numpy as np
import pandas as pd
from . import config as C, data
from .pipeline import fit_models
from .risk_measures import var_es


def rolling_var_es(daily: pd.DataFrame, port, d_fixed=None, window=C.WINDOW, step=C.ROLL_STEP, n_sim=C.N_ROLL,
                   alpha=C.ALPHA, seed=C.SEED, verbose=False, done=None, max_seconds=None) -> pd.DataFrame:
    rng = np.random.default_rng([seed, 4])
    # common random numbers, drawn once
    eps = rng.standard_normal((n_sim, len(port.lam)), dtype=np.float32)   # idiosyncratic shocks
    z = rng.standard_normal((n_sim, 2))                                    # Gaussian factor shocks (M2, M3)
    u_w = rng.random(n_sim)                                                # uniform -> chi-square mixing variable (M3)
    u_b = rng.random(n_sim)                                                # uniform -> bootstrap index (M1)

    r = daily.to_numpy()
    ends = np.arange(window, len(r) + 1, step)
    if ends[-1] != len(r):
        ends = np.r_[ends, len(r)]                                         # always include the latest window
    # optional checkpointing: `done` holds the rows already computed in a previous (interrupted) run
    rows, start, t0 = [], None, time.time()
    if done is not None and len(done):
        rows = done.reset_index().to_dict("records")
        start = (rows[-1]["rho_t"], rows[-1]["nu"])
    for j, e in enumerate(ends):
        if j < len(rows):
            continue
        if max_seconds is not None and time.time() - t0 > max_seconds:
            break
        win = daily.iloc[e - window:e]
        theta = data.overlapping_log_returns(win).to_numpy()               # 496 overlapping weekly returns
        models = fit_models(theta, start=start)
        m2, m3 = models["M2"], models["M3"]
        start = (m3.rho, m3.nu)                                            # warm start for the next window
        draws = {"M1": models["M1"].simulate_from(u_b), "M2": m2.simulate_from(z), "M3": m3.simulate_from(z, u_w)}
        row = {"date": daily.index[e - 1], "sigma_SPI": m2.sigma[0], "sigma_SPX": m2.sigma[1],
               "rho_gauss": m2.rho, "rho_t": m3.rho, "nu": m3.nu, "tail_dep": m3.tail_dep,
               "kurt_SPI": float(pd.Series(theta[:, 0]).kurt()), "kurt_SPX": float(pd.Series(theta[:, 1]).kurt())}
        for name, model in models.items():
            s = port.std_Y(model.cov)
            d = port.thresholds(model, s, C.QUAD_ROLL)                     # exact (M1, M2) or 2-D quadrature (M3)
            Y = port.systematic(draws[name]) + port.idio_scale(s) * eps
            v, es = var_es(port.loss(Y <= d), alpha)
            row[f"VaR_{name}"], row[f"ES_{name}"] = v, es
            if d_fixed is not None:                                        # fixed-barrier extension
                Lf = port.loss(Y <= d_fixed[name])
                row[f"VaRfix_{name}"], row[f"ESfix_{name}"] = var_es(Lf, alpha)
                row[f"ELfix_{name}"] = float(Lf.mean())
        rows.append(row)
        if verbose and j % 50 == 0:
            print(j, len(ends), row["date"].date(), round(row["VaR_M1"]), round(row["VaR_M2"]), round(row["VaR_M3"]),
                  round(row["nu"], 1), flush=True)
    res = pd.DataFrame(rows).set_index("date")
    res.attrs["complete"] = len(rows) == len(ends)
    return res
