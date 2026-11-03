"""Risk-factor models M1 (empirical), M2 (Gaussian), M3 (Gaussian margins + t-copula)  [Q iv, v, vi].
Each model exposes fit(returns) and simulate(n, rng) -> (n, 2) array of Theta draws."""
import numpy as np


class M1Empirical:
    def fit(self, theta): raise NotImplementedError
    def simulate(self, n, rng): raise NotImplementedError   # bootstrap rows


class M2Gaussian:
    def fit(self, theta): raise NotImplementedError          # MLE: sample mean, 1/n covariance
    def simulate(self, n, rng): raise NotImplementedError    # Cholesky, see assignment 6.3.2


class M3TCopula:
    def fit(self, theta): raise NotImplementedError          # MLE of (nu, rho) + Gaussian margins
    def simulate(self, n, rng): raise NotImplementedError    # chi-square mixture, see 6.3.3
