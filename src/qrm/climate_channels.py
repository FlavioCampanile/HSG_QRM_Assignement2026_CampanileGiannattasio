"""Physical climate risk through both credit-risk channels, and climate-intensity stresses  [Q xi, extensions].

Links the storm overlay of `qrm.climate` to the model of Pozdyshev, Lobanov and Ilinsky (2025, BIS WP 1274)
discussed in the course.  In their model a climate event (probability q) scales the asset value of the
borrower by exp(-alpha):

    PD channel   P[default] = (1-q) Phi(C) + q Phi(C + alpha~),        alpha~ = alpha / sigma_A,
    LGD channel  LGD_1 = LGD_0 + (1 - e^{-alpha}) (1 - LGD_0),
    IRB          RWA = Multiplier * LGD_0 [CV(Q) - PD] * (1 + q (LGD_1 - LGD_0) / LGD_0).

Our overlay R_k(e) = R_k (1 - phi MDR_k(e)) is the LGD channel with 1 - e^{-alpha} = phi MDR:
    1 - R_k(e) = LGD_0 + phi MDR (1 - LGD_0).
The storm catalogue gives q and alpha directly (the inputs the BIS model takes from outside providers),
and instead of one binary event every counterparty has the exact law of its worst damage ratio of the
year (`climate.worst_mdr_distribution`), i.e. a mixture over several jump sizes.

Everything here is pure numpy/scipy and works on the stored catalogue.
"""
import numpy as np
from scipy.stats import norm
from . import climate, basel_irb as irb


# ------------------------------------------------------------------ hazard stress (no CLIMADA needed)
def scaled_mdr(cat, scale, canton=False):
    """Damage ratios when every gust speed of the catalogue is multiplied by (1 + scale).

    The impact function stored with the catalogue (intensity grid and MDR = MDD x PAA) is interpolated
    linearly, as CLIMADA does; with scale = 0 this reproduces cat.mdr up to float32 rounding (< 1e-8)."""
    x = cat.extra["inten_canton" if canton else "inten"].astype(float) * (1.0 + scale)
    return np.interp(x, cat.extra["impf_intensity"], cat.extra["impf_mdr"])


# ------------------------------------------------------------------ BIS mapping, counterparty by counterparty
def worst_laws(freq, mdr):
    """Exact law of the worst damage ratio of the year for every counterparty: list of (values, probs)."""
    return [climate.worst_mdr_distribution(freq, mdr[:, k]) for k in range(mdr.shape[1])]


def bis_parameters(laws, lgd0, phi):
    """Single-event BIS parameters implied by the catalogue, per counterparty.

    q_k     = P[at least one damaging event at k in a year]
    m_k     = E[worst damage ratio | damaged]
    alpha_k = -log(1 - phi m_k)                 (asset-value jump, log units)
    LGD1_k  = LGD0 + (1 - e^{-alpha}) (1 - LGD0) = LGD0 + phi m_k (1 - LGD0)
    F_k     = 1 + q (LGD1 - LGD0) / LGD0          (LGD add-on factor of the BIS formula)
    q_k (LGD1 - LGD0) = phi E[m_k] (1 - LGD0): the add-on is exactly the expected-loss effect of our overlay."""
    q = np.array([1.0 - p[-1] for _, p in laws])
    Em = np.array([v @ p for v, p in laws])
    mbar = np.divide(Em, q, out=np.zeros_like(Em), where=q > 0)
    alpha = -np.log1p(-phi * mbar)
    lgd1 = lgd0 + (1.0 - np.exp(-alpha)) * (1.0 - lgd0)
    F = 1.0 + q * (lgd1 - lgd0) / lgd0
    return {"q": q, "Em": Em, "mbar": mbar, "alpha": alpha, "lgd1": lgd1, "F": F}


def mixture_pd(pi, laws, phi, sigma_a):
    """Unconditional PD with the PD channel: mixture over the worst damage of the year,
    P_k = sum_j p_j Phi(Phi^{-1}(pi_k) + alpha_j / sigma_A),  alpha_j = -log(1 - phi m_j)  (q-deformed normal).

    sigma_A is the asset volatility over the horizon of pi_k (one year): it converts the jump in log asset
    value into a shift of the standardised default threshold.  Exact under a Gaussian marginal (M2)."""
    out = np.empty(len(pi))
    for k, (v, p) in enumerate(laws):
        a = -np.log1p(-phi * v) / sigma_a
        out[k] = p @ norm.cdf(norm.ppf(pi[k]) + a)
    return out


def expected_loss_both(E, pi, lgd0, laws, phi, sigma_a):
    """Exact expected loss with both channels (storms independent of the credit factors):
    sum_k E_k sum_j p_j Phi(Phi^{-1}(pi_k) + a_j) (LGD0 + phi m_j (1 - LGD0))."""
    tot = 0.0
    for k, (v, p) in enumerate(laws):
        a = -np.log1p(-phi * v) / sigma_a
        pd = norm.cdf(norm.ppf(pi[k]) + a)
        lgd = lgd0[k] + phi * v * (1 - lgd0[k])
        tot += E[k] * (p @ (pd * lgd))
    return tot


def irb_capital(pi, lgd0, laws, phi, sigma_a, M, alpha=0.999, F=None):
    """IRB capital charge per unit of exposure, without and with physical risk (BIS formula).

    base     LGD0 [CV(Q) - PD] MA(PD)                        (Basel IRB, as in Q x)
    lgd      base x F_k                                        (LGD add-on only)
    both     LGD0 [CV_mix(Q) - P_mix] MA(PD) x F_k             (BIS: q-deformed stressed PD and LGD add-on;
             CV_mix = sum_j p_j Phi((Phi^{-1}(pi)+a_j + sqrt(rho) Phi^{-1}(Q)) / sqrt(1-rho)),
             i.e. the climate event independent of the systematic factor)
    naive    standard IRB formula with the climate-adjusted PD P_mix plugged in, times F_k.
    rho and MA are kept at their values for the original PD in 'both', so that the climate effect is
    not confounded with the PD-dependence of the regulatory parameters."""
    rho, MA = irb.asset_correlation(pi), irb.maturity_adjustment(pi, M)
    zq = norm.ppf(alpha)
    base = lgd0 * (irb.stressed_pd(pi, alpha) - pi) * MA
    cv, pm = np.empty(len(pi)), np.empty(len(pi))
    for k, (v, p) in enumerate(laws):
        a = -np.log1p(-phi * v) / sigma_a
        c = norm.ppf(pi[k]) + a
        cv[k] = p @ norm.cdf((c + np.sqrt(rho[k]) * zq) / np.sqrt(1 - rho[k]))
        pm[k] = p @ norm.cdf(c)
    both = lgd0 * (cv - pm) * MA * F
    naive = lgd0 * (irb.stressed_pd(pm, alpha) - pm) * irb.maturity_adjustment(pm, M) * F
    return {"base": base, "lgd": base * F, "both": both, "naive": naive, "pd_mix": pm, "cv_mix": cv}


def threshold_shift(worst, phi, sigma_a, s):
    """Shift of the default threshold of Y_k in a simulated year with worst damage ratios `worst` (n, K).

    Default with the PD channel: Y_k - s_k alpha_k / sigma_A <= d_k, i.e. the standardised asset value
    V_k = Y_k / s_k falls by alpha~ = alpha / sigma_A standard deviations (s_k is the volatility of Y_k,
    so the shift is expressed in the units of Y_k)."""
    return -np.log1p(-phi * np.asarray(worst, float)) / sigma_a * np.asarray(s)[None, :]
