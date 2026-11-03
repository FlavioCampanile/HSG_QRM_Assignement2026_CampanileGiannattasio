"""Latent-variable credit model  [Q i, ii, iii, v, vii].
Y_k = sqrt(lambda_k) a_k' Theta + sqrt(1-lambda_k) s_k eps_k ;  I_k = 1{Y_k <= d_k}."""
import numpy as np


def std_Yk(a, lam, cov_theta):
    """s_k from Q iii (Var(Y_k) = lambda a' Cov a + (1-lambda) s^2 = s^2 => closed form; derive in report)."""
    raise NotImplementedError


def thresholds(Y, pi):
    """d_k = empirical pi_k-quantile of simulated Y_k, i.e. P[Y_k <= d_k] = pi_k  [Q v]."""
    raise NotImplementedError


def simulate_Y(theta, portfolio, rng):
    """(n_sim, 100) matrix of asset returns given Theta draws."""
    raise NotImplementedError


def portfolio_loss(I, E, R):
    """L = sum_k E_k (1 - R_k) I_k  [Q ii]; I has shape (n_sim, 100)."""
    return (I * (E * (1.0 - R))).sum(axis=1)
