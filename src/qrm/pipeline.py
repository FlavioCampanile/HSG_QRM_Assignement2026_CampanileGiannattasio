"""Shared set-up used by all scripts: load data, fit M1-M3, calibrate s_k and d_k."""
import json
import numpy as np
from . import config as C, data, factor_models as fm
from .credit_model import CreditPortfolio


def fit_models(theta, joint=True, hessian=False, start=None):
    theta = np.asarray(theta, dtype=float)
    return {"M1": fm.M1Empirical().fit(theta),
            "M2": fm.M2Gaussian().fit(theta),
            "M3": fm.M3TCopula().fit(theta, joint=joint, hessian=hessian, start=start)}


def setup(hessian=False):
    """Full-sample set-up: returns prices, weekly returns, fitted models, portfolio, s_k and d_k."""
    px = data.load_prices()
    wk = data.weekly_log_returns(px)
    models = fit_models(wk.to_numpy(), hessian=hessian)
    port = CreditPortfolio(data.load_portfolio())
    s = {m: port.std_Y(models[m].cov) for m in models}
    d = {m: port.thresholds(models[m], s[m]) for m in models}
    return px, wk, models, port, s, d


# ------------------------------------------------------------------ common random numbers
def common_draws(n, n_names, rng):
    """One set of primitive random numbers shared by M1, M2 and M3 (common random numbers).

    z    (n, 2)        independent N(0,1): Gaussian factor shocks of M2 and of M3
    u_w  (n,)          uniforms -> chi-square mixing variable W of M3 (by inversion)
    u_b  (n,)          uniforms -> index of the historical week drawn under M1
    eps  (n, n_names)  idiosyncratic N(0,1) shocks, identical under the three models
    """
    return {"z": rng.standard_normal((n, 2)), "u_w": rng.random(n), "u_b": rng.random(n),
            "eps": rng.standard_normal((n, n_names))}


def factor_draws(models, cr):
    """Theta under each model, built from the same primitive draws.

    M2: Theta = mu + A'Z.  M3: X = sqrt(nu/W) L Z with W = F^{-1}_{chi2}(u_w), then the quantile
    transformation.  M2 and M3 therefore see the same Gaussian shocks and differ only through
    (rho, nu, margins); M1 can share only the idiosyncratic shocks with the other two."""
    return {"M1": models["M1"].simulate_from(cr["u_b"]),
            "M2": models["M2"].simulate_from(cr["z"]),
            "M3": models["M3"].simulate_from(cr["z"], cr["u_w"])}


def asset_returns(models, port, s, cr):
    """(n, 100) matrix of Y_k for each model from one set of common draws; also returns Theta."""
    theta = factor_draws(models, cr)
    Y = {m: port.systematic(theta[m]) + port.idio_scale(s[m]) * cr["eps"] for m in models}
    return Y, theta


def simulate_baseline(models, port, s, n=C.N_SIM):
    """The assignment's baseline experiment (Q v-vii): n joint draws of (Theta, eps) and the
    resulting n x 100 matrix of asset returns Y for each model.

    The three models are simulated from the SAME primitive random numbers.  The 99.9% VaR of
    10,000 scenarios is the 10th largest loss, so with independent streams a difference between
    two models would be dominated by which scenarios each stream happened to draw; with common
    random numbers part of that noise cancels in the difference.  The seed is fixed, so every
    script that calls this function works on exactly the same scenarios."""
    rng = np.random.default_rng([C.SEED, 0])
    return asset_returns(models, port, s, common_draws(n, len(port.lam), rng))[0]


def reference_chunks(models, port, s, n=C.N_REF, chunk=100_000):
    """The reference run (N_REF scenarios, common random numbers) as a stream of chunks.

    Yields (start, Y, theta): the index of the first scenario of the chunk and, for each model, the
    (chunk, 100) matrix of asset returns and the (chunk, 2) factor draws.  The generator is seeded,
    so every script that iterates over it sees exactly the same two million scenarios (script 03 for
    question viii, script 07 for the climate overlay of question xi)."""
    rng = np.random.default_rng([C.SEED, 1000])
    for i in range(0, n, chunk):
        Y, theta = asset_returns(models, port, s, common_draws(chunk, len(port.lam), rng))
        yield i, Y, theta


# ------------------------------------------------------------------ small output helpers
def save_table(name, df, fmt=None, index=True, **kw):
    """Write a DataFrame both as CSV and as a booktabs LaTeX tabular (\\input in the report)."""
    C.TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TABLES / f"{name}.csv", index=index)
    tex = df.to_latex(index=index, escape=False, float_format=fmt, **kw)
    (C.TABLES / f"{name}.tex").write_text(tex)


def _jsonable(o):
    return o.tolist() if isinstance(o, np.ndarray) else float(o)


def save_numbers(name, d):
    """Store scalar results (JSON) so that the report text can be checked against them."""
    C.TABLES.mkdir(parents=True, exist_ok=True)
    (C.TABLES / f"{name}.json").write_text(json.dumps(d, indent=2, default=_jsonable))


def usd(x, digits=0):
    return f"{x:,.{digits}f}"


def sig(x, digits=3, min_dec=2, max_dec=4):
    """Fixed-point string with at least `min_dec` decimals and about `digits` significant digits:
    43.15 -> '43.15', 0.0432 -> '0.0432', 0 -> '0.00'.  Used for changes in basis points, whose
    order of magnitude is not known before the storm catalogue is run."""
    if not np.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    if abs(x) < 0.5 * 10 ** -max_dec:                     # positive but below the printed precision
        return f"$<${10 ** -max_dec:.{max_dec}f}"
    dec = int(min(max(min_dec, digits - 1 - np.floor(np.log10(abs(x)))), max_dec))
    return f"{x:,.{dec}f}"
