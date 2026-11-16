"""Q viii - VaR and ES at 99.9% under M1, M2, M3; Monte Carlo error; reference run.

Three layers:
  (1) the assignment's experiment, N = 10,000, with the thresholds computed in two ways
      (literal: empirical pi_k-quantile of the simulated Y_k; reference: integrated conditional PD);
  (2) a reference run with N = 2,000,000 (200 blocks of 10,000).  Each block repeats experiment (1),
      so the dispersion across blocks is the Monte Carlo standard error of the N = 10,000 estimators;
  (3) all three models are driven by common random numbers in every block, so the standard error of
      a *difference* between two models (M3 - M2) can be estimated directly from paired blocks.
"""
import numpy as np
import pandas as pd
from qrm import config as C, pipeline as P
from qrm.risk_measures import var_es
from qrm.plotting import plt, COL, LAB, FIGW, save

CHUNK = 100_000                                   # = 10 blocks of N_SIM scenarios


def reference_run(models, port, s, d):
    """N_REF scenarios with common random numbers.  Returns the losses (reference thresholds), the
    number of defaults, the large-pool losses E[L | Theta], and per block of N_SIM scenarios the
    (VaR, ES) obtained with the literal thresholds re-estimated inside that block."""
    a, n_blocks = C.ALPHA, C.N_REF // C.N_SIM
    L = {m: np.empty(C.N_REF) for m in models}
    Linf = {m: np.empty(C.N_REF) for m in models}
    nd = {m: np.empty(C.N_REF, dtype=np.int32) for m in models}
    lit = {m: np.empty((n_blocks, 2)) for m in models}
    for i, Y, theta in P.reference_chunks(models, port, s, chunk=CHUNK):
        for m in models:
            I = Y[m] <= d[m]
            L[m][i:i + CHUNK], nd[m][i:i + CHUNK] = port.loss(I), I.sum(axis=1)
            Linf[m][i:i + CHUNK] = port.conditional_pd(theta[m], d[m], s[m]) @ port.w
            for b in range(CHUNK // C.N_SIM):                 # literal thresholds, block by block
                Yb = Y[m][b * C.N_SIM:(b + 1) * C.N_SIM]
                lit[m][i // C.N_SIM + b] = var_es(port.loss(Yb <= port.thresholds_quantile(Yb)), a)
    return L, Linf, nd, lit


def main():
    px, wk, models, port, s, d = P.setup()
    sim = np.load(C.TABLES / "sim_losses.npz")
    a = C.ALPHA
    L, Linf, nd, lit = reference_run(models, port, s, d)
    blk = {m: np.array([var_es(b, a) for b in L[m].reshape(-1, C.N_SIM)]) for m in models}   # reference thresholds
    n_blocks = len(blk["M2"])

    num = {}
    for m in models:
        v, e = var_es(sim[f"L_{m}"], a)                       # N = 10,000, reference thresholds
        vq, eq = var_es(sim[f"Lq_{m}"], a)                    # N = 10,000, literal thresholds
        V, E = var_es(L[m], a)
        V99, E99 = var_es(L[m], 0.99)
        Vinf, Einf = var_es(Linf[m], a)
        num[m] = {"VaR_10k": v, "ES_10k": e, "VaR_10k_quantile_thr": vq, "ES_10k_quantile_thr": eq,
                  "UL_10k": v - port.EL, "UL_10k_quantile_thr": vq - port.EL,
                  "VaR_ref": V, "ES_ref": E, "VaR99_ref": V99, "ES99_ref": E99,
                  "se_VaR_10k": blk[m][:, 0].std(ddof=1), "se_ES_10k": blk[m][:, 1].std(ddof=1),
                  "se_VaR_10k_q": lit[m][:, 0].std(ddof=1), "se_ES_10k_q": lit[m][:, 1].std(ddof=1),
                  "mean_VaR_10k": blk[m][:, 0].mean(), "mean_ES_10k": blk[m][:, 1].mean(),
                  "mean_VaR_10k_q": lit[m][:, 0].mean(), "mean_ES_10k_q": lit[m][:, 1].mean(),
                  "rmse_VaR_10k": np.sqrt(np.mean((blk[m][:, 0] - V) ** 2)),
                  "rmse_VaR_10k_q": np.sqrt(np.mean((lit[m][:, 0] - V) ** 2)),
                  "VaR_10k_p05": np.quantile(blk[m][:, 0], .05), "VaR_10k_p95": np.quantile(blk[m][:, 0], .95),
                  "se_rel_VaR_10k": blk[m][:, 0].std(ddof=1) / V, "se_rel_VaR_10k_q": lit[m][:, 0].std(ddof=1) / V,
                  "se_VaR_ref": blk[m][:, 0].std(ddof=1) / np.sqrt(n_blocks),
                  "EL_ref": L[m].mean(), "UL_ref": V - port.EL, "ES_over_VaR": E / V,
                  "VaR_largepool": Vinf, "ES_largepool": Einf, "UL_largepool": Vinf - port.EL,
                  "granularity_share": 1 - Vinf / V,
                  "defaults_mean": nd[m].mean(), "defaults_q999": float(np.quantile(nd[m], a)),
                  "defaults_max": int(nd[m].max()), "std_ref": L[m].std(), "VaR_pct_exposure": V / port.E.sum()}
    for m in ("M1", "M3"):
        num[m]["VaR_vs_M2"] = num[m]["VaR_ref"] / num["M2"]["VaR_ref"] - 1
        num[m]["ES_vs_M2"] = num[m]["ES_ref"] / num["M2"]["ES_ref"] - 1

    # ---- the difference M3 - M2: paired blocks (common random numbers) vs independent streams -----
    dif = {}
    for key, src in (("ref_thr", blk), ("lit_thr", lit)):
        dv, de = src["M3"][:, 0] - src["M2"][:, 0], src["M3"][:, 1] - src["M2"][:, 1]
        # the same blocks paired with a one-block shift are independent: what separate seeds would give
        dv_ind = np.roll(src["M3"][:, 0], 1) - src["M2"][:, 0]
        rel, rel_ind = dv / src["M2"][:, 0], dv_ind / src["M2"][:, 0]
        dif[key] = {"sd_dVaR_crn": dv.std(ddof=1), "sd_dES_crn": de.std(ddof=1), "sd_dVaR_indep": dv_ind.std(ddof=1),
                    "mean_dVaR": dv.mean(), "mean_dES": de.mean(),
                    "corr_VaR_M2_M3": float(np.corrcoef(src["M2"][:, 0], src["M3"][:, 0])[0, 1]),
                    "p_M3_below_M2_crn": float((dv < 0).mean()), "p_M3_below_M2_indep": float((dv_ind < 0).mean()),
                    "p_gap_above_10pct_crn": float((rel > 0.10).mean()),
                    "p_gap_above_10pct_indep": float((rel_ind > 0.10).mean()),
                    "p_ES_M3_below_M2_crn": float((de < 0).mean())}
    r = dif["ref_thr"]
    dif.update({"dVaR_10k": num["M3"]["VaR_10k"] - num["M2"]["VaR_10k"], "dES_10k": num["M3"]["ES_10k"] - num["M2"]["ES_10k"],
                "dVaR_10k_q": num["M3"]["VaR_10k_quantile_thr"] - num["M2"]["VaR_10k_quantile_thr"],
                "dES_10k_q": num["M3"]["ES_10k_quantile_thr"] - num["M2"]["ES_10k_quantile_thr"],
                "dVaR_10k_rel": num["M3"]["VaR_10k"] / num["M2"]["VaR_10k"] - 1,
                "dVaR_10k_q_rel": num["M3"]["VaR_10k_quantile_thr"] / num["M2"]["VaR_10k_quantile_thr"] - 1,
                "dVaR_ref": num["M3"]["VaR_ref"] - num["M2"]["VaR_ref"], "dES_ref": num["M3"]["ES_ref"] - num["M2"]["ES_ref"],
                "se_dVaR_ref": r["sd_dVaR_crn"] / np.sqrt(n_blocks), "se_dES_ref": r["sd_dES_crn"] / np.sqrt(n_blocks),
                "t_dVaR_ref": (num["M3"]["VaR_ref"] - num["M2"]["VaR_ref"]) / (r["sd_dVaR_crn"] / np.sqrt(n_blocks)),
                "t_dES_ref": (num["M3"]["ES_ref"] - num["M2"]["ES_ref"]) / (r["sd_dES_crn"] / np.sqrt(n_blocks)),
                "sd_reduction_crn": 1 - r["sd_dVaR_crn"] / r["sd_dVaR_indep"]})
    num["M3_minus_M2"] = dif

    # conditional expected loss E[L | Theta = theta_t] in each observed week (M1): which weeks make the tail?
    cel = port.conditional_pd(wk.to_numpy(), d["M1"], s["M1"]) @ port.w
    top = np.argsort(cel)[::-1][:5]
    num["worst_weeks_M1"] = [{"week": str(wk.index[i].date()), "SPI": float(wk.iloc[i, 0]), "SPX": float(wk.iloc[i, 1]),
                              "cond_EL": float(cel[i])} for i in top]
    P.save_numbers("numbers_q8", num)

    # ---- main tables (headers written by hand; each fits the 5in text block of the report) --------
    u = P.usd
    line = lambda k, cells: f"{k} & " + " & ".join(cells) + " \\\\"
    # (a) the assignment's experiment, N = 10,000
    rows_a = {LAB[m]: [u(num[m]["VaR_10k_quantile_thr"]), u(num[m]["ES_10k_quantile_thr"]),
                       u(num[m]["VaR_10k"]), u(num[m]["ES_10k"]), u(num[m]["se_VaR_10k_q"]), u(num[m]["se_VaR_10k"])]
              for m in models}
    rows_a["M3 $-$ M2"] = [u(dif["dVaR_10k_q"]), u(dif["dES_10k_q"]), u(dif["dVaR_10k"]), u(dif["dES_10k"]),
                           u(dif["lit_thr"]["sd_dVaR_crn"]), u(r["sd_dVaR_crn"])]
    cols_a = ["VaR (empirical quantile)", "ES (empirical quantile)", "VaR (reference)", "ES (reference)",
              "s.e. VaR (empirical quantile)", "s.e. VaR (reference)"]
    pd.DataFrame(rows_a, index=cols_a).T.to_csv(C.TABLES / "tab_var_es.csv")
    keys = list(rows_a)
    (C.TABLES / "tab_var_es.tex").write_text(
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Thresholds $\\hat d_k$} & \\multicolumn{2}{c}{Thresholds $d_k$}"
        " & \\multicolumn{2}{c}{Std.\\ error of VaR} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Model & VaR & ES & VaR & ES & with $\\hat d_k$ & with $d_k$ \\\\\n\\midrule\n"
        + "\n".join(line(k, rows_a[k]) for k in keys[:3]) + "\n\\midrule\n" + line(keys[3], rows_a[keys[3]])
        + "\n\\bottomrule\n\\end{tabular}\n")
    # (b) the reference run, N = 2,000,000
    rows_b = {LAB[m]: [u(num[m]["VaR_ref"]), u(num[m]["ES_ref"]), u(num[m]["UL_ref"]), f"{num[m]['ES_over_VaR']:.3f}",
                       f"{100 * num[m]['VaR_pct_exposure']:.1f}", u(num[m]["VaR99_ref"]), u(num[m]["ES99_ref"])]
              for m in models}
    rows_b["M3 $-$ M2"] = [u(dif["dVaR_ref"]), u(dif["dES_ref"]), u(dif["dVaR_ref"]), "--", "--",
                           u(num["M3"]["VaR99_ref"] - num["M2"]["VaR99_ref"]),
                           u(num["M3"]["ES99_ref"] - num["M2"]["ES99_ref"])]
    rows_b["(std.\\ error)"] = [f"({u(dif['se_dVaR_ref'])})", f"({u(dif['se_dES_ref'])})", f"({u(dif['se_dVaR_ref'])})",
                               "", "", "", ""]
    cols_b = ["VaR 99.9%", "ES 99.9%", "UL 99.9%", "ES/VaR", "VaR (% of exposure)", "VaR 99%", "ES 99%"]
    pd.DataFrame(rows_b, index=cols_b).T.to_csv(C.TABLES / "tab_var_es_ref.csv")
    keys = list(rows_b)
    (C.TABLES / "tab_var_es_ref.tex").write_text(
        "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
        " & \\multicolumn{5}{c}{$\\alpha=99.9\\%$} & \\multicolumn{2}{c}{$\\alpha=99\\%$} \\\\\n"
        "\\cmidrule(lr){2-6}\\cmidrule(lr){7-8}\n"
        "Model & VaR & ES & UL & ES/VaR & \\% exp. & VaR & ES \\\\\n\\midrule\n"
        + "\n".join(line(k, rows_b[k]) for k in keys[:3]) + "\n\\midrule\n"
        + "\n".join(line(k, rows_b[k]) for k in keys[3:]) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- secondary table ------------------------------------------------------------------------------
    t2 = pd.DataFrame({LAB[m]: [u(num[m]["VaR_largepool"]), u(num[m]["ES_largepool"]),
                                f"{num[m]['defaults_mean']:.2f}", f"{num[m]['defaults_q999']:.0f}",
                                u(num[m]["mean_VaR_10k_q"]), u(num[m]["se_VaR_10k_q"]), u(num[m]["rmse_VaR_10k_q"]),
                                u(num[m]["mean_VaR_10k"]), u(num[m]["se_VaR_10k"]), u(num[m]["rmse_VaR_10k"])]
                       for m in models},
                      index=["VaR$_{99.9\\%}$, large pool", "ES$_{99.9\\%}$, large pool",
                             "Mean number of defaults", "99.9\\% quantile of defaults",
                             "Thresholds $\\hat d_k$: mean of 200 VaR estimates", "\\quad standard deviation",
                             "\\quad root mean squared error",
                             "Thresholds $d_k$: mean of 200 VaR estimates", "\\quad standard deviation",
                             "\\quad root mean squared error"])
    P.save_table("tab_var_es_extra", t2)

    # ---- figure: tail of the loss distribution (reference run) --------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(FIGW, 2.2))
    for m in models:
        Ls = np.sort(L[m])[::-1]
        p = np.arange(1, len(Ls) + 1) / len(Ls)
        keep = np.unique(np.geomspace(1, len(Ls), 4000).astype(int)) - 1
        ax[0].plot(Ls[keep] / 1e6, p[keep], color=COL[m], label=LAB[m], lw=1.1)
        ax[0].plot(num[m]["VaR_ref"] / 1e6, 1 - a, "o", color=COL[m], ms=3.5)
    ax[0].axhline(1 - a, color="k", ls=":", lw=0.7)
    ax[0].set_yscale("log"); ax[0].set_ylim(2e-6, 1); ax[0].set_xlim(0, 2.0)
    ax[0].set_xlabel("Loss level $\\ell$ (USD million)"); ax[0].set_ylabel("$P[L > \\ell]$")
    ax[0].set_title("Loss tail, reference run"); ax[0].legend()
    # sampling distribution of the two N = 10,000 VaR estimators
    for j, m in enumerate(models):
        for off, src, hatch in ((-0.2, lit, "///"), (0.2, blk, None)):
            bp = ax[1].boxplot(src[m][:, 0] / 1e6, positions=[j + off], widths=0.32, patch_artist=True, showfliers=False)
            bp["boxes"][0].set(facecolor=COL[m], alpha=0.35, hatch=hatch); bp["medians"][0].set(color="k")
        ax[1].plot([j - 0.42, j + 0.42], [num[m]["VaR_ref"] / 1e6] * 2, "-", color=COL[m], lw=1.2)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(list(models))
    ax[1].set_ylabel("VaR$_{99.9\\%}$ (USD million)")
    ax[1].set_title("VaR estimators, $N=10{,}000$")
    save(fig, "fig_loss_tail")


if __name__ == "__main__":
    main()
