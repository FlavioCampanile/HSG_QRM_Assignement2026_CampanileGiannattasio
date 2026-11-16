"""Q x - Basel IRB capital, RWA, and comparison with the model-based unexpected loss."""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from qrm import config as C, pipeline as P, basel_irb as irb
from qrm.risk_measures import var_es
from qrm.plotting import plt, COL, LAB, FIGW, save


def main():
    px, wk, models, port, s, d = P.setup()
    df = port.df.copy()
    df["rho_irb"] = irb.asset_correlation(df["pi"])
    df["b"] = irb.maturity_slope(df["pi"])
    df["MA"] = irb.maturity_adjustment(df["pi"])
    df["K"] = irb.capital_charge(df["pi"], df["R"])
    df["K_usd"] = df["E"] * df["K"]
    K, RWA = df["K_usd"].sum(), 12.5 * df["K_usd"].sum()

    # model-implied asset correlation of each name with the systematic part: under M2,
    # Corr(Y_j, Y_k) = sqrt(lambda_j lambda_k) Corr(a_j'Theta, a_k'Theta)
    S = models["M2"].cov
    sys = port.A @ S @ port.A.T / np.outer(s["M2"], s["M2"])           # Corr(a_j'Theta, a_k'Theta)
    R_asset = np.sqrt(np.outer(port.lam, port.lam)) * sys
    off = R_asset[~np.eye(100, dtype=bool)]
    df["rho_model"] = (R_asset.sum(axis=1) - port.lam) / 99              # average correlation with the others

    order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
    g = df.groupby("rating").agg(n=("k", "size"), pi=("pi", "first"), rho=("rho_irb", "first"),
                                 b=("b", "first"), MA=("MA", "first"), lgd=("LGD", "mean"), Kbar=("K", "mean"),
                                 E=("E", "sum"), K=("K_usd", "sum"), lam=("lam", "mean")).loc[order]
    tab = pd.DataFrame({"$n$": g["n"], "$\\pi_k$ (\\%)": (100 * g["pi"]).map("{:.2f}".format),
                        "$\\rho_k$": g["rho"].map("{:.4f}".format), "$b(\\pi_k)$": g["b"].map("{:.4f}".format),
                        "MA": g["MA"].map("{:.2f}".format), "$\\overline{\\mathrm{LGD}}$": g["lgd"].map("{:.3f}".format),
                        "$\\bar K_k$ (\\%)": (100 * g["Kbar"]).map("{:.2f}".format),
                        "Exposure": g["E"].map(P.usd), "$\\sum E_k K_k$": g["K"].map(P.usd),
                        "$\\bar\\lambda_k$": g["lam"].map("{:.3f}".format)})
    tab.loc["Total"] = ["100", "", "", "", "", f"{df['LGD'].mean():.3f}", f"{100 * K / df['E'].sum():.2f}",
                        P.usd(df["E"].sum()), P.usd(K), f"{df['lam'].mean():.3f}"]
    tab.index.name = None
    P.save_table("tab_irb", tab)

    # ---- decomposition of the gap  UL(M2) - K  ---------------------------------------------
    q8 = json.loads((C.TABLES / "numbers_q8.json").read_text())
    # (a) IRB formula with each name's own model correlation lambda_k (single factor, infinitely granular)
    cond = norm.cdf((norm.ppf(port.pi) + np.sqrt(port.lam) * norm.ppf(C.ALPHA)) / np.sqrt(1 - port.lam))
    K_lam = float(port.w @ (cond - port.pi))
    steps = {"IRB capital $K$ (regulatory $\\rho_k$)": K,
             "IRB formula with $\\rho_k=\\lambda_k$": K_lam,
             "M2, large pool": q8["M2"]["UL_largepool"],
             "M2, actual portfolio": q8["M2"]["UL_ref"],
             "M3, actual portfolio": q8["M3"]["UL_ref"],
             "M1, actual portfolio": q8["M1"]["UL_ref"]}
    t2 = pd.DataFrame({"UL$_{99.9\\%}$ (USD)": [P.usd(v) for v in steps.values()],
                       "\\% of exposure": [f"{100 * v / port.E.sum():.2f}" for v in steps.values()],
                       "Ratio to $K$": [f"{v / K:.2f}" for v in steps.values()]}, index=list(steps))
    P.save_table("tab_irb_gap", t2)

    P.save_numbers("numbers_q10", {"K": K, "RWA": RWA, "K_pct": K / df["E"].sum(), "K_lambda": K_lam,
                                   "MA_min": df["MA"].min(), "MA_max": df["MA"].max(),
                                   "rho_irb_mean": df["rho_irb"].mean(), "lambda_mean": df["lam"].mean(),
                                   "asset_corr_model_mean": off.mean(), "asset_corr_model_median": np.median(off),
                                   "sys_corr_min": sys.min(), "sys_corr_mean": sys[~np.eye(100, dtype=bool)].mean(),
                                   "K_share_by_rating": (g["K"] / K).to_dict(),
                                   "Kbar_pct": (100 * g["Kbar"]).to_dict(),
                                   "ratio_UL_M2": q8["M2"]["UL_ref"] / K, "ratio_UL_M3": q8["M3"]["UL_ref"] / K,
                                   "ratio_UL_M1": q8["M1"]["UL_ref"] / K,
                                   "granularity_addon_M2": q8["M2"]["UL_ref"] - q8["M2"]["UL_largepool"]})

    # ---- figure ------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(FIGW, 2.2))
    ax[0].scatter(df["rho_irb"], df["lam"], s=12, c=np.log10(df["pi"]), cmap="viridis", linewidths=0)
    ax[0].plot([0.1, 0.25], [0.1, 0.25], "k-", lw=0.6)
    ax[0].set_xlabel("Regulatory asset correlation $\\rho_k$"); ax[0].set_ylabel("Model systematic weight $\\lambda_k$")
    ax[0].set_title("Prescribed vs model correlation"); ax[0].set_xlim(0.11, 0.25); ax[0].set_xticks([0.12, 0.16, 0.20, 0.24])
    vals = [K, K_lam, q8["M2"]["UL_largepool"], q8["M2"]["UL_ref"], q8["M3"]["UL_ref"], q8["M1"]["UL_ref"]]
    labs = ["IRB\n$K$", "IRB\n$\\rho_k\\!=\\!\\lambda_k$", "M2\nlarge\npool", "M2", "M3", "M1"]
    cols = [COL["IRB"], COL["IRB"], COL["M2"], COL["M2"], COL["M3"], COL["M1"]]
    ax[1].bar(range(6), np.array(vals) / 1e6, color=cols, alpha=0.8)
    ax[1].set_xticks(range(6)); ax[1].set_xticklabels(labs, fontsize=6.5)
    ax[1].set_ylabel("UL at 99.9% (USD million)")
    ax[1].set_title("From IRB capital to model UL")
    save(fig, "fig_irb")


if __name__ == "__main__":
    main()
