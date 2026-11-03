"""Empirical VaR and Expected Shortfall of a loss sample  [Q viii, ix, xi]."""
import numpy as np


def var(loss, alpha):
    return float(np.quantile(loss, alpha, method="higher"))


def es(loss, alpha):
    """Mean of losses at or beyond VaR_alpha."""
    v = var(loss, alpha)
    return float(loss[loss >= v].mean())
