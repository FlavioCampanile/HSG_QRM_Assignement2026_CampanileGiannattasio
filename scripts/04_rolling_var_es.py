"""Q ix - dynamic VaR/ES: rolling re-estimation on the last 500 daily returns."""
import numpy as np
import pandas as pd
from qrm import config as C, data, pipeline as P
from qrm.credit_model import CreditPortfolio
from qrm.rolling import rolling_var_es
from qrm.plotting import plt, COL, LAB, FIGW, save


def placebo(models, port, n_rep=C.N_PLACEBO, n_sim=C.N_ROLL):
    """How much of a VaR/ES difference between M3 and M2 is pure simulation noise?

    Placebo model: M3 with the parameters of M2 (same means, volatilities and rho) and nu = 200,
    i.e. a t-copula that is Gaussian for all practical purposes.  Its true VaR equals that of M2, so
    any difference measured with n_sim scenarios and common random numbers - exactly the set-up of one
    rolling window - is noise.  Repeated over n_rep independent seeds (full-sample parameters), this
    gives the noise band shown in the figure.  The same experiment with the actual full-sample M3
    gives the sampling distribution of the estimated difference when the true one is known."""
    import copy
    from qrm.risk_measures import var_es
    m2, m3 = models["M2"], models["M3"]
    pl = copy.copy(m3)
    pl.mean, pl.sigma, pl.rho, pl.nu = m2.mean.copy(), m2.sigma.copy(), float(m2.rho), C.NU_MAX
    pl._moments()
    mods = {"M2": m2, "placebo": pl, "M3": m3}
    s = {k: port.std_Y(v.cov) for k, v in mods.items()}
    d = {k: port.thresholds(v, s[k], C.QUAD_ROLL) for k, v in mods.items()}
    rng = np.random.default_rng([C.SEED, 9])
    out = np.empty((n_rep, 3, 2))
    for r in range(n_rep):
        cr = P.common_draws(n_sim, len(port.lam), rng)
        for j, (k, v) in enumerate(mods.items()):
            th = v.simulate_from(cr["z"]) if k == "M2" else v.simulate_from(cr["z"], cr["u_w"])
            out[r, j] = var_es(port.loss(port.systematic(th) + port.idio_scale(s[k]) * cr["eps"] <= d[k]), C.ALPHA)
    rel_pl, rel_m3 = out[:, 1] / out[:, 0] - 1, out[:, 2] / out[:, 0] - 1
    return {"n_rep": n_rep, "n_sim": n_sim,
            "VaR_mean": rel_pl[:, 0].mean(), "VaR_sd": rel_pl[:, 0].std(ddof=1), "VaR_share_neg": (rel_pl[:, 0] < 0).mean(),
            "ES_mean": rel_pl[:, 1].mean(), "ES_sd": rel_pl[:, 1].std(ddof=1), "ES_share_neg": (rel_pl[:, 1] < 0).mean(),
            "M3_VaR_mean": rel_m3[:, 0].mean(), "M3_VaR_sd": rel_m3[:, 0].std(ddof=1), "M3_VaR_share_neg": (rel_m3[:, 0] < 0).mean(),
            "M3_ES_mean": rel_m3[:, 1].mean(), "M3_ES_sd": rel_m3[:, 1].std(ddof=1), "M3_ES_share_neg": (rel_m3[:, 1] < 0).mean(),
            "se_rel_VaR_M2": out[:, 0, 0].std(ddof=1) / out[:, 0, 0].mean()}


def main(recompute=True, budget=None):
    px, wk, models, port, s, d = P.setup()                 # full-sample fit: gives the fixed barriers
    daily = data.daily_log_returns(px)
    out = C.TABLES / "rolling_var_es.csv"
    part = C.TABLES / "rolling_partial.csv"               # checkpoint file (the full run takes ~15 minutes)
    if recompute or not out.exists():
        done = pd.read_csv(part, index_col=0, parse_dates=True) if part.exists() else None
        res = rolling_var_es(daily, port, d_fixed=d, verbose=True, done=done, max_seconds=budget)
        if not res.attrs["complete"]:
            res.to_csv(part)
            print(f"checkpoint: {len(res)} windows done - run the script again to continue")
            return
        res.to_csv(out)
        part.unlink(missing_ok=True)
    res = pd.read_csv(out, index_col=0, parse_dates=True)

    # ---- placebo experiment (cached: about two minutes) ---------------------------------------
    import json
    pfile = C.TABLES / "placebo.json"
    if recompute or not pfile.exists():
        pfile.write_text(json.dumps(placebo(models, port), indent=2, default=float))
    pl = json.loads(pfile.read_text())
    dV, dE = res["VaR_M3"] / res["VaR_M2"] - 1, res["ES_M3"] / res["ES_M2"] - 1
    at_bound = res["nu"] > 0.99 * C.NU_MAX
    band = lambda x, k: (x - pl[f"{k}_mean"]).abs() <= 2 * pl[f"{k}_sd"]

    # ---- summary table -------------------------------------------------------------------
    rows = {}
    for m in ("M1", "M2", "M3"):
        v, e = res[f"VaR_{m}"], res[f"ES_{m}"]
        rows[LAB[m]] = [P.usd(v.mean()), P.usd(v.std()), P.usd(v.min()), str(v.idxmin().date()),
                        P.usd(v.max()), str(v.idxmax().date()), P.usd(v.iloc[-1]), P.usd(e.mean()), P.usd(e.iloc[-1])]
    tab = pd.DataFrame(rows, index=["Mean VaR", "Std.\\ dev.\\ of VaR", "Minimum VaR", "\\quad window ending",
                                    "Maximum VaR", "\\quad window ending", "Latest VaR", "Mean ES", "Latest ES"])
    tab.to_csv(C.TABLES / "tab_rolling.csv")
    (C.TABLES / "tab_rolling.tex").write_text(tab.to_latex(escape=False, column_format="lrrr"))
    P.save_numbers("numbers_q9", {
        "n_windows": len(res), "first": str(res.index[0].date()), "last": str(res.index[-1].date()),
        "mean": res.mean().to_dict(), "std": res.std().to_dict(), "last_row": res.iloc[-1].to_dict(),
        "corr_VaR_M1_M2": float(res["VaR_M1"].corr(res["VaR_M2"])),
        "corr_VaRM2_rho": float(res["VaR_M2"].corr(res["rho_gauss"])),
        "corr_VaRM1_kurt": float(res["VaR_M1"].corr(res[["kurt_SPI", "kurt_SPX"]].mean(axis=1))),
        "share_nu_at_bound": float((res["nu"] > 0.99 * C.NU_MAX).mean()),
        "share_nu_below_10": float((res["nu"] < 10).mean()),
        "ratio_M1_M2_mean": float((res["VaR_M1"] / res["VaR_M2"]).mean()),
        "ratio_M1_M2_max": float((res["VaR_M1"] / res["VaR_M2"]).max()),
        "ratio_M3_M2_mean": float((res["VaR_M3"] / res["VaR_M2"]).mean()),
        "ratio_M3_M2_max": float((res["VaR_M3"] / res["VaR_M2"]).max()),
        "min": res.min().to_dict(), "max": res.max().to_dict(),
        "argmax_VaRfix_M2": str(res["VaRfix_M2"].idxmax().date()), "argmin_VaRfix_M2": str(res["VaRfix_M2"].idxmin().date()),
        "argmax_VaR_M1": str(res["VaR_M1"].idxmax().date()), "argmin_VaR_M1": str(res["VaR_M1"].idxmin().date()),
        "corr_VaRfixM2_sigma": float(res["VaRfix_M2"].corr(res[["sigma_SPI", "sigma_SPX"]].mean(axis=1))),
        "nu_median": float(res["nu"].median()),
        "ratio_M1_M2_min": float((res["VaR_M1"] / res["VaR_M2"]).min()),
        "share_M1_below_M2": float((res["VaR_M1"] < res["VaR_M2"]).mean()),
        "ratio_M3_M2_ES_mean": float((res["ES_M3"] / res["ES_M2"]).mean()),
        "ratio_M3_M2_ES_max": float((res["ES_M3"] / res["ES_M2"]).max()),
        # --- M3 against M2, window by window, and the placebo noise band
        "placebo": pl,
        "dVaR_mean": float(dV.mean()), "dVaR_min": float(dV.min()), "dVaR_max": float(dV.max()), "dVaR_sd": float(dV.std()),
        "dES_mean": float(dE.mean()), "dES_min": float(dE.min()), "dES_max": float(dE.max()), "dES_sd": float(dE.std()),
        "share_VaR_M3_below_M2": float((dV < 0).mean()), "share_ES_M3_below_M2": float((dE < 0).mean()),
        "share_bound_among_below": float(at_bound[dV < 0].mean()) if (dV < 0).any() else 0.0,
        "share_dVaR_in_band": float(band(dV, "VaR").mean()), "share_dES_in_band": float(band(dE, "ES").mean()),
        "share_dVaR_above_band": float((dV > pl["VaR_mean"] + 2 * pl["VaR_sd"]).mean()),
        "share_dES_above_band": float((dE > pl["ES_mean"] + 2 * pl["ES_sd"]).mean()),
        "dVaR_mean_bound": float(dV[at_bound].mean()), "dVaR_sd_bound": float(dV[at_bound].std()),
        "dES_mean_bound": float(dE[at_bound].mean()), "dES_sd_bound": float(dE[at_bound].std()),
        "dVaR_mean_nu_below_10": float(dV[res["nu"] < 10].mean()), "dES_mean_nu_below_10": float(dE[res["nu"] < 10].mean()),
        "dVaR_max_nu_below_10": float(dV[res["nu"] < 10].max()),
        "dVaR_low_minus_bound": float(dV[res["nu"] < 10].mean() - dV[at_bound].mean()),
        "corr_dVaR_drho": float(dV.corr(res["rho_t"] - res["rho_gauss"])),
        "corr_dES_taildep": float(dE.corr(res["tail_dep"])),
        "ratio_std_M1_M2": float(res["VaR_M1"].std() / res["VaR_M2"].std()),
        "sigma_ratio_max_min": float(res["sigma_SPX"].max() / res["sigma_SPX"].min())})

    # ---- figure: M3 - M2 against the placebo noise band -----------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(FIGW, 3.19), sharex=True)
    for a_, x, k, lab in ((ax[0], dV, "VaR", "VaR"), (ax[1], dE, "ES", "ES")):
        a_.axhspan(100 * (pl[f"{k}_mean"] - 2 * pl[f"{k}_sd"]), 100 * (pl[f"{k}_mean"] + 2 * pl[f"{k}_sd"]),
                   color="0.85", label="placebo: mean $\\pm$ 2 s.d.")
        a_.axhline(0, color="k", lw=0.6)
        a_.plot(res.index, 100 * x, color=COL["M3"], lw=0.9, label="M3 relative to M2")
        a_.set_ylabel(f"{lab}: M3/M2$\\,-\\,$1 (%)")
    ax[0].legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax[1].set_xlabel("End date of the 500-day estimation window")
    save(fig, "fig_rolling_diff")

    # ---- figure ----------------------------------------------------------------------------
    fig, ax = plt.subplots(5, 1, figsize=(FIGW, 5.5), sharex=True,
                           gridspec_kw={"height_ratios": [1.4, 1.4, 1.4, 1, 1]})
    for m in ("M1", "M2", "M3"):
        ax[0].plot(res.index, res[f"VaR_{m}"] / 1e6, color=COL[m], lw=0.9, label=LAB[m])
        ax[1].plot(res.index, res[f"ES_{m}"] / 1e6, color=COL[m], lw=0.9, label=LAB[m])
        ax[2].plot(res.index, res[f"VaRfix_{m}"] / 1e6, color=COL[m], lw=0.9, label=LAB[m])
    ax[0].set_ylabel("VaR (USD mn)"); ax[1].set_ylabel("ES (USD mn)")
    ax[2].set_ylabel("VaR, fixed $d_k$ (USD mn)")
    ax[0].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax[3].plot(res.index, res["rho_gauss"], color=COL["M2"], lw=0.9, label="Gaussian $\\rho$ (M2)")
    ax[3].plot(res.index, res["rho_t"], color=COL["M3"], lw=0.9, label="$t$-copula $\\rho$ (M3)")
    ax[3].plot(res.index, res["tail_dep"], color=COL["M3"], lw=0.9, ls="--", label="tail dependence $\\lambda_L$ (M3)")
    ax[3].set_ylabel("Dependence"); ax[3].set_ylim(-0.03, 1.0); ax[3].legend(ncol=3, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 0.97))
    ax[4].plot(res.index, res["nu"], color=COL["M3"], lw=0.9)
    ax[4].set_yscale("log"); ax[4].set_ylabel("$\\hat\\nu$ (log scale)")
    ax[4].set_xlabel("End date of the 500-day estimation window")
    save(fig, "fig_rolling")


if __name__ == "__main__":
    import sys
    # optional: --budget SECONDS stops after that time and writes a checkpoint
    b = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else None
    main(recompute="--plot-only" not in sys.argv, budget=b)
