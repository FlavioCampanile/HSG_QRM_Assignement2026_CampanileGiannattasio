"""Basel IRB capital charge (assignment section 6.4)  [Q x]."""
import numpy as np
from scipy.stats import norm
from . import config as C


def asset_correlation(pi, avc=C.AVC):
    w = (1 - np.exp(-50 * pi)) / (1 - np.exp(-50))
    return avc * (0.12 * w + 0.24 * (1 - w))


def maturity_adjustment(pi, M=C.MATURITY):
    b = (0.11852 - 0.05478 * np.log(pi)) ** 2
    return (1 + (M - 2.5) * b) / (1 - 1.5 * b)


def capital_charge(pi, R, M=C.MATURITY, alpha=C.ALPHA):
    """K_k as a fraction of exposure."""
    rho = asset_correlation(pi)
    cond_pd = norm.cdf((norm.ppf(pi) + np.sqrt(rho) * norm.ppf(alpha)) / np.sqrt(1 - rho))
    return (1 - R) * (cond_pd - pi) * maturity_adjustment(pi, M)
