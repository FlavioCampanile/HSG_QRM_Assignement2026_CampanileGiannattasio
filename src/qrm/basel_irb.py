"""Basel II/III IRB capital charge (assignment Section 6.4)  [Q x]."""
import numpy as np
from scipy.stats import norm
from . import config as C


def asset_correlation(pi, avc=C.AVC):
    """Regulatory asset correlation: decreasing in PD, from 24% (PD -> 0) to 12% (PD -> 1)."""
    w = (1 - np.exp(-50 * pi)) / (1 - np.exp(-50))
    return avc * (0.12 * w + 0.24 * (1 - w))


def maturity_slope(pi):
    return (0.11852 - 0.05478 * np.log(pi)) ** 2


def maturity_adjustment(pi, M=C.MATURITY):
    b = maturity_slope(pi)
    return (1 + (M - 2.5) * b) / (1 - 1.5 * b)


def stressed_pd(pi, alpha=C.ALPHA):
    """Vasicek conditional PD given the alpha-quantile of the single systematic factor."""
    rho = asset_correlation(pi)
    return norm.cdf((norm.ppf(pi) + np.sqrt(rho) * norm.ppf(alpha)) / np.sqrt(1 - rho))


def capital_charge(pi, R, M=C.MATURITY, alpha=C.ALPHA):
    """K_k as a fraction of exposure."""
    return (1 - R) * (stressed_pd(pi, alpha) - pi) * maturity_adjustment(pi, M)
