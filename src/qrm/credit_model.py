"""Latent-variable credit model  [Q i, ii, iii, v, vi, vii].

    Y_k = sqrt(lambda_k) a_k' Theta + sqrt(1-lambda_k) s_k eps_k ,   I_k = 1{Y_k <= d_k},
    L   = sum_k E_k (1-R_k) I_k .
"""
import numpy as np
from scipy.special import ndtr, ndtri
from . import config as C


class CreditPortfolio:
    """Holds the portfolio parameters as arrays (length 100)."""

    def __init__(self, df):
        self.df = df
        self.A = df[["a1", "a2"]].to_numpy()          # (100, 2) factor loadings a_k
        self.lam = df["lam"].to_numpy()
        self.pi = df["pi"].to_numpy()
        self.E = df["E"].to_numpy()
        self.lgd = df["LGD"].to_numpy()
        self.w = self.E * self.lgd                     # loss if k defaults: E_k (1 - R_k)
        self.EL = float(self.w @ self.pi)              # expected loss, identical under M1-M3

    # ---- Q iii -------------------------------------------------------------------------
    def std_Y(self, cov_theta):
        """s_k = sqrt(a_k' Sigma a_k).

        Var(Y_k) = lambda a'Sigma a + (1-lambda) s^2 = s^2  <=>  s^2 = a' Sigma a."""
        return np.sqrt(np.einsum("ki,ij,kj->k", self.A, cov_theta, self.A))

    def systematic(self, theta):
        """sqrt(lambda_k) a_k' Theta for every draw: (n, 100)."""
        return (theta @ self.A.T) * np.sqrt(self.lam)

    def idio_scale(self, s):
        return np.sqrt(1 - self.lam) * s

    # ---- Q i ---------------------------------------------------------------------------
    def conditional_pd(self, theta, d, s):
        """p_k(theta) = Phi( (d_k - sqrt(lambda_k) a_k' theta) / (sqrt(1-lambda_k) s_k) )."""
        return ndtr((d - self.systematic(theta)) / self.idio_scale(s))

    # ---- Q v, vi -----------------------------------------------------------------------
    def simulate_Y(self, theta, s, rng):
        """(n, 100) matrix of asset log-returns given the factor draws."""
        eps = rng.standard_normal((len(theta), len(self.lam)))
        return self.systematic(theta) + self.idio_scale(s) * eps

    def thresholds_quantile(self, Y):
        """d_k as the empirical pi_k-quantile of the simulated Y_k (the literal reading of Q v)."""
        n = len(Y)
        Ys = np.sort(Y, axis=0)
        idx = np.maximum(np.ceil(n * self.pi - 1e-9).astype(int), 1) - 1   # generalised inverse of F_n
        return Ys[idx, np.arange(Y.shape[1])]

    def thresholds_gaussian(self, mean_theta, s):
        """Closed form when Theta is Gaussian (M2): Y_k ~ N(sqrt(lambda) a'mu, s^2)."""
        return np.sqrt(self.lam) * (self.A @ mean_theta) + s * ndtri(self.pi)

    def implied_pd(self, theta_ref, d, s, weights=None, chunk=200_000):
        """P[Y_k <= d_k] = E[p_k(Theta)] for a discrete law of Theta given by points and weights."""
        out = np.zeros(len(self.pi))
        for i in range(0, len(theta_ref), chunk):
            p = self.conditional_pd(theta_ref[i:i + chunk], d, s)
            out += p.sum(axis=0) / len(theta_ref) if weights is None else weights[i:i + chunk] @ p
        return out

    def thresholds_integrated(self, theta_ref, s, weights=None, tol=1e-10, max_iter=200):
        """Solve  E[p_k(Theta; d_k)] = sum_j w_j p_k(theta_j; d_k) = pi_k  for d_k (all k at once).

        The idiosyncratic shock is integrated out analytically, so the only approximation
        left is the one in the law of Theta (none for M1, a Gauss quadrature rule for M3).
        Safeguarded Newton iteration on  h(d) = log P[Y_k <= d] - log pi_k  (h is increasing).
        """
        V = self.systematic(theta_ref)                 # (m, 100)
        wt = np.full(len(V), 1.0 / len(V)) if weights is None else weights
        c = self.idio_scale(s)
        q = ndtri(self.pi)
        lo, hi = V.min(axis=0) + c * q, V.max(axis=0) + c * q   # P <= pi at lo, P >= pi at hi
        d = np.clip(wt @ V + s * q, lo, hi)                     # Gaussian starting value
        for _ in range(max_iter):
            u = (d - V) / c
            P = wt @ ndtr(u)
            dP = (wt @ np.exp(-0.5 * u * u) / np.sqrt(2 * np.pi)) / c
            h = np.log(np.maximum(P, 1e-300)) - np.log(self.pi)
            lo, hi = np.where(h < 0, d, lo), np.where(h >= 0, d, hi)
            step = h * np.maximum(P, 1e-300) / np.maximum(dP, 1e-300)
            new = d - step
            bad = ~np.isfinite(new) | (new <= lo) | (new >= hi)
            new = np.where(bad, 0.5 * (lo + hi), new)           # fall back to bisection
            if np.max(np.abs(new - d) / s) < tol:
                d = new
                break
            d = new
        return d

    def thresholds(self, model, s, grid=None):
        """Reference thresholds for a fitted factor model (closed form or numerical integration)."""
        if model.name == "M2":
            return self.thresholds_gaussian(model.mean, s)
        ref, wt = model.reference_sample() if model.name == "M1" else model.reference_sample(grid)
        return self.thresholds_integrated(ref, s, wt)

    # ---- Q ii, vii ---------------------------------------------------------------------
    def loss(self, I, w=None):
        """L = sum_k E_k (1 - R_k) I_k ; I has shape (n, 100)."""
        return I @ (self.w if w is None else w)

    def simulate_losses(self, model, d, s, n, rng, chunk=100_000, return_defaults=False):
        """Portfolio losses for n joint draws of (Theta, eps); memory-safe for large n."""
        out, nd = np.empty(n), np.empty(n, dtype=np.int32)
        for i in range(0, n, chunk):
            m = min(chunk, n - i)
            I = self.simulate_Y(model.simulate(m, rng), s, rng) <= d
            out[i:i + m] = self.loss(I)
            nd[i:i + m] = I.sum(axis=1)
        return (out, nd) if return_defaults else out

    def large_pool_losses(self, model, d, s, n, rng, chunk=200_000):
        """E[L | Theta] = sum_k E_k (1-R_k) p_k(Theta): the loss of an infinitely granular
        portfolio with the same systematic risk (idiosyncratic risk diversified away)."""
        out = np.empty(n)
        for i in range(0, n, chunk):
            m = min(chunk, n - i)
            out[i:i + m] = self.conditional_pd(model.simulate(m, rng), d, s) @ self.w
        return out
