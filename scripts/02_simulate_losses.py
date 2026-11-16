"""Q v, vi, vii - simulate Y_k, calibrate the thresholds d_k, build the loss distributions."""
import numpy as np
import pandas as pd
from scipy.special import ndtr
from qrm import config as C, pipeline as P
from qrm.plotting import plt, COL, LAB, FIGW, save


def exact_pd(port, model, d, s):
    """P[Y_k <= d_k] under the fitted model, without simulation error (used to judge a threshold)."""
    if model.name == "M2":                                    # Y_k is exactly Gaussian
        return ndtr((d - np.sqrt(port.lam) * (port.A @ model.mean)) / s)
    ref, wt = model.reference_sample() if model.name == "M1" else model.reference_sample(0.05)
    return port.implied_pd(ref, d, s, wt)


def main():
    px, wk, models, port, s, d = P.setup()
    df = port.df
    N = C.N_SIM
    Y_all = P.simulate_baseline(models, port, s)             # 10,000 x 100 draws of Y_k per model
    k_order = np.maximum(np.ceil(N * port.pi - 1e-9).astype(int), 1)   # ceil(N pi_k): rank of the order statistic
    losses, losses_q, ndef, d_q, ratio = {}, {}, {}, {}, {}
    for name in models:
        Y = Y_all[name]
        d_q[name] = port.thresholds_quantile(Y)              # (a) literal answer: empirical pi_k-quantile of Y_k
        Iq, I = Y <= d_q[name], Y <= d[name]                 # (b) reference thresholds (exact / integrated)
        assert (Iq.sum(axis=0) == k_order).all()             # in sample, k defaults exactly ceil(N pi_k) times
        losses[name], ndef[name] = port.loss(I), I.sum(axis=1)
        losses_q[name] = port.loss(Iq)
        ratio[name] = exact_pd(port, models[name], d_q[name], s[name]) / port.pi   # true PD implied by d_hat
    np.savez(C.TABLES / "sim_losses.npz", **{f"L_{m}": losses[m] for m in losses},
             **{f"Lq_{m}": losses_q[m] for m in losses}, **{f"N_{m}": ndef[m] for m in losses})

    # ---- table by rating: order statistic used, reference thresholds, accuracy of the literal ones ----
    order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
    grp = df["rating"]
    tab = pd.DataFrame({"$n$": df.groupby("rating").size(),
                        "$\\pi_k$ (\\%)": (100 * df.groupby("rating")["pi"].first()).map("{:g}".format),
                        "$\\lceil N\\pi_k\\rceil$": pd.Series(k_order, index=grp).groupby(level=0).first().map("{:,}".format)})
    for m in models:                                          # standardised threshold (d_k - E[Y_k]) / s_k
        dstar = (d[m] - np.sqrt(port.lam) * (port.A @ models[m].mean)) / s[m]
        tab[f"$\\bar d^{{\\,*}}_k$ {m}"] = pd.Series(dstar, index=grp).groupby(level=0).mean().map("{:.3f}".format)
    for m in models:
        r = pd.Series(ratio[m], index=grp).groupby(level=0)
        tab[f"PD ratio {m}"] = [f"{a:.2f}--{b:.2f}" for a, b in zip(r.min(), r.max())]
    tab = tab.loc[order]
    tab.index.name = None
    P.save_table("tab_thresholds", tab)

    chk = {m: {"pd_ratio_quantile_min": float(ratio[m].min()), "pd_ratio_quantile_max": float(ratio[m].max()),
               "pd_ratio_quantile_min_BtoCCC": float(ratio[m][df["pi"].to_numpy() >= 0.05].min()),
               "pd_ratio_quantile_max_BtoCCC": float(ratio[m][df["pi"].to_numpy() >= 0.05].max()),
               "pd_ratio_reference_max_abs_err": float(np.abs(exact_pd(port, models[m], d[m], s[m]) / port.pi - 1).max()),
               "EL_true_quantile_thr": float(port.w @ (ratio[m] * port.pi)),
               "EL_sim": float(losses[m].mean()), "EL_sim_quantile": float(losses_q[m].mean())} for m in models}
    P.save_numbers("numbers_q5", {"EL": port.EL, "total_exposure": float(port.E.sum()),
                                  "max_single_loss": float(port.w.max()), "sum_w": float(port.w.sum()),
                                  "EL_share_CCC": float((port.w * port.pi)[df["rating"] == "CCC"].sum() / port.EL),
                                  "EL_share_B_CCC": float((port.w * port.pi)[df["pi"] >= 0.05].sum() / port.EL),
                                  "n_names_k_le_7": int((k_order <= 7).sum()),
                                  "n_cantons": int(df["canton_abbr"].nunique()),
                                  "share_exposure_top3_cantons": float(df.groupby("canton_abbr")["E"].sum().nlargest(3).sum() / port.E.sum()),
                                  "share_exposure_inv_grade": float(port.E[df["pi"].to_numpy() <= 0.002].sum() / port.E.sum()),
                                  "R_mean": float(df["R"].mean()), "R_min": float(df["R"].min()), "R_max": float(df["R"].max()),
                                  "lam_min": float(port.lam.min()), "lam_max": float(port.lam.max()), "lam_mean": float(port.lam.mean()),
                                  "recovery_value": float((port.E * df["R"]).sum()),
                                  "quad_points_M3": int(len(models["M3"].reference_sample()[1])),
                                  "quad_mass_M3": models["M3"].quad_mass,
                                  "checks": chk})

    # ---- figure: simulated-quantile vs reference thresholds -----------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(FIGW, 1.98), sharey=True)
    for a, m in zip(ax, models):
        mean_Y = np.sqrt(port.lam) * (port.A @ models[m].mean)
        x, y = (d[m] - mean_Y) / s[m], (d_q[m] - mean_Y) / s[m]
        a.scatter(x, y, s=10, color=COL[m], alpha=0.7, linewidths=0)
        a.plot([-7, -0.5], [-7, -0.5], "k-", lw=0.6)
        a.set_title(LAB[m]); a.set_xlabel("reference $d_k^*$" if m == "M2" else "")
    ax[0].set_ylabel("empirical quantile $\\hat d_k^{\\,*}$")
    save(fig, "fig_thresholds")

    # ---- figure: histogram of the loss distribution (N = 10,000) -------------------------------
    fig, ax = plt.subplots(figsize=(FIGW, 2.13))
    bins = np.linspace(0, 1.3, 66)                           # USD million
    for m in models:
        ax.hist(losses_q[m] / 1e6, bins=bins, histtype="step", color=COL[m], label=LAB[m], lw=1.1)
    ax.set_yscale("log"); ax.set_xlabel("Portfolio loss $L$ (USD million)"); ax.set_ylabel("Number of scenarios (log scale)")
    ax.axvline(port.EL / 1e6, color="k", ls=":", lw=0.8); ax.text(port.EL / 1e6 * 1.15, 2500, "EL", fontsize=7)
    ax.legend()
    save(fig, "fig_loss_hist")


if __name__ == "__main__":
    main()
