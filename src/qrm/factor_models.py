"""Risk-factor models for Theta = (SPI, SPX) weekly log-returns  [Q iv, v, vi].

M1  empirical distribution of the historical sample,
M2  bivariate Gaussian N(mu, Sigma),
M3  Gaussian margins N(mu_i, sigma_i^2) coupled by a t-copula (nu, rho).

Every model exposes
    fit(theta)            -> self            (maximum likelihood where applicable)
    simulate(n, rng)      -> (n, 2) array    (draws of Theta)
    mean, cov             -> first two moments of Theta under the model
    reference_sample()    -> (m, 2) array or None; a large/exact sample of Theta used to
                             integrate the conditional PD when calibrating the thresholds d_k.
"""
import numpy as np
from scipy import optimize, special, stats
from scipy.stats import qmc
from . import config as C


# --------------------------------------------------------------------------- M1
class M1Empirical:
    name = "M1"

    def fit(self, theta):
        self.sample = np.asarray(theta, dtype=float)
        self.mean = self.sample.mean(axis=0)
        # covariance of the empirical distribution (divisor n: it is a population moment of F_n)
        self.cov = np.cov(self.sample.T, bias=True)
        return self

    def simulate(self, n, rng):
        """Draw rows (pairs!) with replacement: keeps the empirical dependence intact."""
        return self.sample[rng.integers(0, len(self.sample), size=n)]

    def simulate_from(self, u):
        """Same as simulate(), but from given uniforms (common random numbers)."""
        return self.sample[np.minimum((u * len(self.sample)).astype(int), len(self.sample) - 1)]

    def reference_sample(self):
        return self.sample, None      # the empirical distribution itself (equal weights): integration is exact


# --------------------------------------------------------------------------- M2
class M2Gaussian:
    name = "M2"

    def fit(self, theta):
        x = np.asarray(theta, dtype=float)
        self.n = len(x)
        self.mean = x.mean(axis=0)                 # MLE of mu
        self.cov = np.cov(x.T, bias=True)          # MLE of Sigma (divisor n)
        self.sigma = np.sqrt(np.diag(self.cov))
        self.rho = self.cov[0, 1] / self.sigma.prod()
        z = (x - self.mean) / self.sigma
        q = (z[:, 0] ** 2 - 2 * self.rho * z[:, 0] * z[:, 1] + z[:, 1] ** 2) / (1 - self.rho ** 2)
        self.loglik = float(np.sum(-np.log(2 * np.pi) - np.log(self.sigma.prod())
                                   - 0.5 * np.log(1 - self.rho ** 2) - 0.5 * q))
        return self

    def simulate(self, n, rng):
        """Theta = mu + A'Z with Sigma = A'A (Cholesky, assignment Section 6.3.2)."""
        A = np.linalg.cholesky(self.cov).T         # upper triangular, A'A = Sigma
        return self.mean + rng.standard_normal((n, 2)) @ A

    def simulate_from(self, z_indep):
        """Same as simulate(), but from given N(0, I) draws (common random numbers)."""
        return self.mean + z_indep @ np.linalg.cholesky(self.cov).T

    def reference_sample(self):
        return None                                # thresholds are available in closed form


# --------------------------------------------------------------------------- M3
def _t_scores(z, nu):
    """x = t_nu^{-1}(Phi(z)), evaluated in the lower tail by symmetry for numerical accuracy."""
    return -np.sign(z) * stats.t.ppf(special.ndtr(-np.abs(z)), nu) * 1.0


def t_copula_logdensity(z1, z2, nu, rho):
    """log c(u1,u2; nu, rho) for u_i = Phi(z_i):  c = f_{t,2}(x1,x2) / (f_{t,1}(x1) f_{t,1}(x2))."""
    x1, x2 = _t_scores(z1, nu), _t_scores(z2, nu)
    lg = special.gammaln
    log_f2 = (lg((nu + 2) / 2) - lg(nu / 2) - np.log(np.pi * nu) - 0.5 * np.log(1 - rho ** 2)
              - (nu + 2) / 2 * np.log1p((x1 ** 2 - 2 * rho * x1 * x2 + x2 ** 2) / (nu * (1 - rho ** 2))))
    log_f1 = lambda x: (lg((nu + 1) / 2) - lg(nu / 2) - 0.5 * np.log(np.pi * nu)
                        - (nu + 1) / 2 * np.log1p(x ** 2 / nu))
    return log_f2 - log_f1(x1) - log_f1(x2)


def m3_loglik(params, x):
    """Full log-likelihood of M3; params = (mu1, mu2, sigma1, sigma2, rho, nu)."""
    mu1, mu2, s1, s2, rho, nu = params
    z1, z2 = (x[:, 0] - mu1) / s1, (x[:, 1] - mu2) / s2
    log_margins = stats.norm.logpdf(z1) - np.log(s1) + stats.norm.logpdf(z2) - np.log(s2)
    return float(np.sum(log_margins + t_copula_logdensity(z1, z2, nu, rho)))


class M3TCopula:
    name = "M3"

    def fit(self, theta, joint=True, hessian=False, start=None):
        """Maximum likelihood.

        Step 1 (IFM start): Gaussian margins by MLE, then (rho, nu) by maximising the
                copula likelihood on u_i = Phi((theta_i - mu_i)/sigma_i).
        Step 2 (joint=True): all six parameters are re-optimised jointly on the full
                likelihood, started at the IFM estimate (the full MLE).
        """
        x = np.asarray(theta, dtype=float)
        self.n = len(x)
        mu, sd = x.mean(axis=0), x.std(axis=0)
        z = (x - mu) / sd
        tau = stats.kendalltau(x[:, 0], x[:, 1])[0]
        rho0 = np.sin(np.pi * tau / 2)             # moment-type starting value for rho

        # ---- step 1: copula parameters given the margins (profile over a grid of nu, then polish)
        # (the optimiser works with log(nu), which is much better scaled than nu itself;
        #  `start` = (rho, nu) from a neighbouring window gives a warm start in the rolling analysis)
        cop_nll = lambda p: -np.sum(t_copula_logdensity(z[:, 0], z[:, 1], np.exp(p[1]), p[0]))
        bnds_c = [(-0.995, 0.995), (np.log(2.05), np.log(C.NU_MAX))]
        starts = [(rho0, nu0) for nu0 in (3.0, 6.0, 15.0, 50.0)] if start is None else [start, (rho0, 8.0)]
        best = min((optimize.minimize(cop_nll, [r0, np.log(n0)], method="L-BFGS-B", bounds=bnds_c)
                    for r0, n0 in starts), key=lambda r: r.fun)
        self.ifm = np.r_[mu, sd, best.x[0], np.exp(best.x[1])]
        p_hat, ll = self.ifm.copy(), m3_loglik(self.ifm, x)

        # ---- step 2: joint MLE of the six parameters
        if joint:
            scale = np.r_[sd, sd, 1.0, 1.0]        # rescale so that all parameters are O(1)
            unpack = lambda q: np.r_[q[:4] * scale[:4], q[4], np.exp(q[5])]
            nll = lambda q: -m3_loglik(unpack(q), x)
            bnds = [(None, None)] * 2 + [(0.2, 5.0)] * 2 + bnds_c
            q0 = np.r_[self.ifm[:4] / scale[:4], self.ifm[4], np.log(self.ifm[5])]
            res = optimize.minimize(nll, q0, method="L-BFGS-B", bounds=bnds)
            if np.isfinite(res.fun) and -res.fun >= ll:
                p_hat, ll = unpack(res.x), -res.fun
        self.params, self.loglik = p_hat, ll
        self.mean, self.sigma = p_hat[:2], p_hat[2:4]
        self.rho, self.nu = float(p_hat[4]), float(p_hat[5])
        self._moments()
        self.se = self._std_errors(x) if hessian else None
        return self

    # -- simulation (assignment Sections 6.3.1 and 6.3.3) ---------------------------------
    def _transform(self, z, w):
        """(Z, W) -> Theta:  X = sqrt(nu/W) Z ~ t_nu(rho);  U = t_nu(X);  Theta = mu + sigma Phi^{-1}(U)."""
        x = z * np.sqrt(self.nu / w)[:, None]
        # Phi^{-1}(t_nu(x)) computed in the lower tail by symmetry
        g = -np.sign(x) * special.ndtri(np.maximum(stats.t.cdf(-np.abs(x), self.nu), 1e-300))
        return self.mean + self.sigma * g

    def simulate(self, n, rng):
        L = np.array([[1.0, 0.0], [self.rho, np.sqrt(1 - self.rho ** 2)]])
        z = rng.standard_normal((n, 2)) @ L.T
        w = rng.chisquare(self.nu, size=n)
        return self._transform(z, w)

    def simulate_from(self, z_indep, u_w):
        """Same as simulate(), but from given N(0, I) draws and uniforms (common random numbers)."""
        L = np.array([[1.0, 0.0], [self.rho, np.sqrt(1 - self.rho ** 2)]])
        return self._transform(z_indep @ L.T, stats.chi2.ppf(u_w, self.nu))

    def reference_sample(self, step=None):
        """Quadrature rule (points, weights) for the law of Theta under M3.

        Write Theta_i = mu_i + sigma_i G_i.  By Sklar's theorem (G_1, G_2) has density
            c_{nu,rho}(Phi(g_1), Phi(g_2)) phi(g_1) phi(g_2),
        which is known in closed form, so an expectation E[h(Theta)] is a two-dimensional integral.
        It is computed with the trapezoidal rule on a regular grid over [-GMAX, GMAX]^2 (the
        trapezoidal rule converges geometrically for smooth, rapidly decaying integrands).
        Unlike a (quasi-)random sample, the grid covers the far tails, with tiny weights, which is
        where default probabilities of one basis point are determined.
        `step` is the grid spacing in units of the marginal standard deviation.
        """
        h = C.QUAD_FULL if step is None else step
        g = np.arange(-C.GMAX, C.GMAX + h / 2, h)
        G1, G2 = np.meshgrid(g, g, indexing="ij")
        g1, g2 = G1.ravel(), G2.ravel()
        logw = t_copula_logdensity(g1, g2, self.nu, self.rho) - 0.5 * (g1 ** 2 + g2 ** 2)
        w = np.exp(logw - logw.max())
        keep = w > 1e-18                                         # drop nodes with negligible mass
        w = w[keep]
        # total mass before normalisation (should be 1): a built-in accuracy check of the rule
        self.quad_mass = float(np.exp(logw.max()) * w.sum() * h * h / (2 * np.pi))
        return self.mean + self.sigma * np.column_stack([g1[keep], g2[keep]]), w / w.sum()

    def sobol_sample(self, m_log2=16):
        """Scrambled-Sobol (quasi-Monte Carlo) sample of Theta; kept only to document the accuracy
        gain of the quadrature rule (see scripts/02_simulate_losses.py)."""
        u = qmc.Sobol(d=3, scramble=True, seed=C.SEED).random_base2(m_log2)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        L = np.array([[1.0, 0.0], [self.rho, np.sqrt(1 - self.rho ** 2)]])
        return self._transform(special.ndtri(u[:, :2]) @ L.T, stats.chi2.ppf(u[:, 2], self.nu))

    def _moments(self):
        """Covariance of Theta: variances are sigma_i^2 (Gaussian margins); the Pearson
        correlation implied by the t-copula has no closed form and is integrated numerically."""
        th, wt = self.reference_sample(C.QUAD_ROLL)
        zz = (th - self.mean) / self.sigma
        self.pearson = float(wt @ (zz[:, 0] * zz[:, 1]) / np.sqrt((wt @ zz[:, 0] ** 2) * (wt @ zz[:, 1] ** 2)))
        s1, s2 = self.sigma
        self.cov = np.array([[s1 ** 2, self.pearson * s1 * s2], [self.pearson * s1 * s2, s2 ** 2]])
        # coefficient of (lower = upper) tail dependence of the t-copula
        self.tail_dep = float(2 * stats.t.cdf(-np.sqrt((self.nu + 1) * (1 - self.rho) / (1 + self.rho)), self.nu + 1))

    def _std_errors(self, x):
        """Asymptotic standard errors from the observed information (finite-difference Hessian)."""
        p = self.params
        h = np.maximum(np.abs(p), 1e-3) * 1e-3
        k = len(p)
        H = np.zeros((k, k))
        f = lambda q: m3_loglik(q, x)
        for i in range(k):
            for j in range(i, k):
                ei, ej = np.eye(k)[i] * h[i], np.eye(k)[j] * h[j]
                H[i, j] = H[j, i] = (f(p + ei + ej) - f(p + ei - ej) - f(p - ei + ej) + f(p - ei - ej)) / (4 * h[i] * h[j])
        try:
            return np.sqrt(np.diag(np.linalg.inv(-H)))
        except np.linalg.LinAlgError:
            return np.full(k, np.nan)
