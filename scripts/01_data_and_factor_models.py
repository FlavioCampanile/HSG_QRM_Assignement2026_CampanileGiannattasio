"""Q iv - data, weekly returns, maximum-likelihood estimation of M2 and M3."""
import numpy as np
import pandas as pd
from scipy import stats
from qrm import config as C, pipeline as P
from qrm.plotting import plt, COL, LAB, FIGW, save


def main():
    px, wk, models, port, s, d = P.setup(hessian=True)
    th = wk.to_numpy()
    n = len(th)
    m1, m2, m3 = models["M1"], models["M2"], models["M3"]

    # ---- descriptive statistics of the weekly log-returns --------------------------------
    desc = pd.DataFrame({
        "Mean (\\%)": 100 * wk.mean(), "Std.\\ dev.\\ (\\%)": 100 * wk.std(ddof=0),
        "Skewness": wk.skew(), "Excess kurtosis": wk.kurt(),
        "Min (\\%)": 100 * wk.min(), "Max (\\%)": 100 * wk.max(),
        "Jarque--Bera": [stats.jarque_bera(wk[c])[0] for c in wk]}).T
    P.save_table("tab_descriptive", desc, fmt="%.3f")

    # dependence in calm vs stressed weeks (stress: both indices below their 10% quantile / SPX in worst decile)
    q = wk.quantile(0.10)
    stress = wk["SPX"] <= q["SPX"]
    u = wk.rank() / (n + 1)
    lam_hat = {p: float(((u["SPI"] <= p) & (u["SPX"] <= p)).mean() / p) for p in (0.10, 0.05, 0.01)}
    gauss_joint = {p: float(stats.multivariate_normal(cov=[[1, m2.rho], [m2.rho, 1]]).cdf([stats.norm.ppf(p)] * 2) / p)
                   for p in (0.10, 0.05, 0.01)}

    # ---- parameter table -----------------------------------------------------------------
    se2 = {"mu": m2.sigma / np.sqrt(n), "sigma": m2.sigma / np.sqrt(2 * n), "rho": (1 - m2.rho ** 2) / np.sqrt(n)}
    f = lambda v, e, k=4: f"{v:.{k}f} ({e:.{k}f})"
    par = pd.DataFrame({
        "M2 Gaussian": [f(100 * m2.mean[0], 100 * se2["mu"][0]), f(100 * m2.mean[1], 100 * se2["mu"][1]),
                        f(100 * m2.sigma[0], 100 * se2["sigma"][0]), f(100 * m2.sigma[1], 100 * se2["sigma"][1]),
                        f(m2.rho, se2["rho"]), "--", f"{m2.rho:.4f}", "0",
                        f"{m2.loglik:.2f}", f"{-2 * m2.loglik + 2 * 5:.2f}"],
        "M3 $t$-copula": [f(100 * m3.params[0], 100 * m3.se[0]), f(100 * m3.params[1], 100 * m3.se[1]),
                          f(100 * m3.params[2], 100 * m3.se[2]), f(100 * m3.params[3], 100 * m3.se[3]),
                          f(m3.rho, m3.se[4]), f(m3.nu, m3.se[5], 2), f"{m3.pearson:.4f}", f"{m3.tail_dep:.4f}",
                          f"{m3.loglik:.2f}", f"{-2 * m3.loglik + 2 * 6:.2f}"]},
        index=["$\\mu_1$ (SPI, \\%)", "$\\mu_2$ (SPX, \\%)", "$\\sigma_1$ (SPI, \\%)", "$\\sigma_2$ (SPX, \\%)",
               "$\\rho$", "$\\nu$", "Pearson corr.\\ of $\\Theta$", "Tail dependence $\\lambda_L$",
               "Log-likelihood", "AIC"])
    P.save_table("tab_mle", par)

    lr = 2 * (m3.loglik - m2.loglik)
    P.save_numbers("numbers_q4", {
        "n_weeks": n, "first_week": str(wk.index[0].date()), "last_week": str(wk.index[-1].date()),
        "n_days": len(px), "first_day": str(px.index[0].date()), "last_day": str(px.index[-1].date()),
        "raw_last_SPI": str(pd.to_datetime(pd.read_excel(C.MARKET_FILE, sheet_name="SPI")["Date"]).max().date()),
        "raw_last_SPX": str(pd.to_datetime(pd.read_excel(C.MARKET_FILE, sheet_name="SPX")["Date"]).max().date()),
        "spi_min_week": str(wk["SPI"].idxmin().date()), "spi_max_week": str(wk["SPI"].idxmax().date()),
        "spx_min_week": str(wk["SPX"].idxmin().date()), "spx_min": float(wk["SPX"].min()), "spi_min": float(wk["SPI"].min()),
        "spi_max": float(wk["SPI"].max()),
        "corr_all": float(wk.corr().iloc[0, 1]),
        "corr_calm": float(wk[~stress].corr().iloc[0, 1]), "corr_stress": float(wk[stress].corr().iloc[0, 1]),
        "joint_exceed_emp": lam_hat, "joint_exceed_gauss": gauss_joint,
        "M2": {"mu": m2.mean, "sigma": m2.sigma, "rho": m2.rho, "loglik": m2.loglik},
        "M3": {"params": m3.params, "se": m3.se, "ifm": m3.ifm, "loglik": m3.loglik,
               "pearson": m3.pearson, "tail_dep": m3.tail_dep},
        # LR test of nu = infinity: boundary case, p-value = 0.5 * P[chi2_1 > LR]
        "LR": lr, "LR_pvalue": 0.5 * stats.chi2.sf(lr, 1),
        "kendall_tau": float(stats.kendalltau(th[:, 0], th[:, 1])[0]),
        "worst_weeks": wk.nsmallest(5, "SPX").mul(100).round(2).reset_index().astype(str).values.tolist()})

    # ---- figure: data vs simulated factor models -----------------------------------------
    rng = np.random.default_rng(C.SEED)
    fig, ax = plt.subplots(1, 3, figsize=(FIGW, 1.98), sharex=True, sharey=True)
    for a, (key, pts) in zip(ax, [("M1", th), ("M2", m2.simulate(n, rng)), ("M3", m3.simulate(n, rng))]):
        a.scatter(100 * pts[:, 0], 100 * pts[:, 1], s=3, alpha=0.45, color=COL[key], linewidths=0)
        a.set_title({"M1": "Data (= M1)", "M2": "M2 Gaussian, simulated",
                     "M3": "M3 $t$-copula, simulated"}[key])
        a.set_xlabel("SPI weekly log-return (%)" if key == "M2" else "")
        a.set_xlim(-18, 15); a.set_ylim(-18, 15)
    ax[0].set_ylabel("SPX weekly log-return (%)")
    save(fig, "fig_factors_scatter")

    # ---- figure: normal QQ-plots of the margins -------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(FIGW, 2.13))
    for a, c, mu, sg in zip(ax, wk.columns, m2.mean, m2.sigma):
        z = np.sort((wk[c].to_numpy() - mu) / sg)
        qn = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        a.plot(qn, z, ".", ms=2.5, color=COL["M1"])
        a.plot([-4, 4], [-4, 4], "-", lw=0.8, color=COL["M2"])
        a.set_title(f"{c}: standardised weekly log-returns")
        a.set_xlabel("Gaussian quantile")
    ax[0].set_ylabel("Empirical quantile")
    save(fig, "fig_qq")


if __name__ == "__main__":
    main()
