"""Q xi, extensions (no CLIMADA needed) - physical risk through both credit channels and hazard stresses.

Builds on the storm catalogue of script 06 and the overlay of script 07, and links them to the model of
Pozdyshev, Lobanov and Ilinsky (2025, BIS WP 1274) presented in the course:

  (1) the BIS parameters implied by the catalogue for every counterparty (event probability q_k,
      asset-value jump alpha_k, LGD_1, LGD add-on factor) and the check that the BIS LGD add-on reproduces
      the expected-loss effect of our overlay exactly;
  (2) the climate-adjusted IRB capital of question x (LGD add-on, q-deformed stressed PD, naive plug-in);
  (3) a reference run (the two million credit scenarios of question viii, with independent storm years)
      in which storms act through the LGD only (the assignment's design), through the PD only (a jump in
      the asset value shifts the default threshold) and through both, with two asset volatilities;
  (4) hazard stresses: every gust speed of the catalogue raised by 5%, 10% and 20%, a stylised stand-in for a
      high-physical-risk scenario, propagated through the convex impact function.

Usage
    python scripts/07b_climate_channels.py
"""
import numpy as np
from qrm import config as C, pipeline as P, climate, climate_channels as cc
from qrm.risk_measures import var_es
from qrm.plotting import plt, COL, LAB, FIGW, save

PPM = 1e6
SIGMA_A = 0.25          # annual asset volatility (horizon of pi_k), as in the course example
SIGMA_LOW = 0.10        # low-volatility borrowers: the same damage moves the threshold 2.5 times more
SCALES = (0.05, 0.10, 0.20)
ROWS = {                # reference-run variants: key -> (label, channels, sigma_A, gust scale)
    "lgd":      ("LGD channel (assignment design)", "lgd", SIGMA_A, 0.0),
    "pd":       ("PD channel only", "pd", SIGMA_A, 0.0),
    "both":     ("Both channels", "both", SIGMA_A, 0.0),
    "both_s10": ("Both channels, $\\sigma_A=10\\%$", "both", SIGMA_LOW, 0.0),
    "g05":      ("Both, gusts $+5\\%$", "both", SIGMA_A, 0.05),
    "g10":      ("Both, gusts $+10\\%$", "both", SIGMA_A, 0.10),
    "g20":      ("Both, gusts $+20\\%$", "both", SIGMA_A, 0.20),
}


def exact_dEL(port, laws, channel, sigma):
    """Exact relative increase of the expected loss (storms independent of the credit factors)."""
    R, lgd0, E, pi = port.df["R"].to_numpy(), port.lgd, port.E, port.pi
    if channel == "lgd":
        Em = np.array([v @ p for v, p in laws])
        return C.PHI * float((E * R * pi) @ Em) / port.EL
    if channel == "pd":
        return float((E * lgd0) @ (cc.mixture_pd(pi, laws, C.PHI, sigma) - pi)) / port.EL
    return cc.expected_loss_both(E, pi, lgd0, laws, C.PHI, sigma) / port.EL - 1


def main():
    px, wk, models, port, s, d = P.setup()
    df = port.df
    R, pi, lgd0, E = df["R"].to_numpy(), port.pi, port.lgd, port.E
    ER = E * R
    cat = climate.StormCatalogue.load(C.CATALOGUE_FILE)
    if cat.synthetic:
        raise SystemExit("synthetic catalogue: not reported")
    freq = cat.freq
    num = {"sigma_A": SIGMA_A, "sigma_low": SIGMA_LOW, "phi": C.PHI}

    # ---- hazard stresses: damage ratios with scaled gusts ------------------------------------------
    mdr = {0.0: cat.mdr}
    num["interp_check_maxdiff"] = float(np.abs(cc.scaled_mdr(cat, 0.0) - cat.mdr).max())
    for sc in SCALES:
        mdr[sc] = cc.scaled_mdr(cat, sc)
    laws = {sc: cc.worst_laws(freq, m) for sc, m in mdr.items()}
    num["stress"] = {}
    for sc, m in mdr.items():
        Em = np.array([v @ p for v, p in laws[sc]])
        num["stress"][f"s{round(100*sc):03d}"] = {"mdr_max": float(m.max()), "Em_mean": float(Em.mean()), "Em_max": float(Em.max()),
                                      "Em_ratio_to_base": float(Em.mean() / np.array([v @ p for v, p in laws[0.0]]).mean()),
                                      "n_damaging": int((m.max(axis=1) > 0).sum()),
                                      "p_year_damage": float(1 - np.exp(-freq[m.max(axis=1) > 0].sum()))}

    # ---- (1) BIS parameters implied by the catalogue --------------------------------------------------
    b = cc.bis_parameters(laws[0.0], lgd0, C.PHI)
    w = E * lgd0 * pi                                           # EL weights
    bis = {"q_mean": float(b["q"].mean()), "q_min": float(b["q"].min()), "q_max": float(b["q"].max()),
           "mbar_mean": float(b["mbar"][b["q"] > 0].mean()), "mbar_max": float(b["mbar"].max()),
           "alpha_max": float(b["alpha"].max()), "alpha_mean": float(b["alpha"][b["q"] > 0].mean()),
           "alphatilde_max": float(b["alpha"].max() / SIGMA_A),
           "dlgd_max": float((b["lgd1"] - lgd0).max()), "dlgd_mean": float((b["lgd1"] - lgd0)[b["q"] > 0].mean()),
           "F_max": float(b["F"].max()), "F_ELweighted": float((w @ b["F"]) / w.sum())}
    # BIS LGD add-on in expected-loss terms = exact overlay effect
    bis["dEL_from_F"] = float(w @ (b["F"] - 1)) / port.EL
    bis["dEL_overlay_exact"] = exact_dEL(port, laws[0.0], "lgd", SIGMA_A)
    bis["identity_rel_error"] = abs(bis["dEL_from_F"] / bis["dEL_overlay_exact"] - 1)
    num["bis"] = bis

    # ---- (2) climate-adjusted IRB capital (question x) -------------------------------------------------
    irb = {}
    for key, sig in (("s25", SIGMA_A), ("s10", SIGMA_LOW)):
        k = cc.irb_capital(pi, lgd0, laws[0.0], C.PHI, sig, C.MATURITY, C.ALPHA, b["F"])
        K0 = float(E @ k["base"])
        irb[key] = {v: float(E @ k[v]) for v in ("base", "lgd", "both", "naive")}
        irb[key].update({f"d_{v}": irb[key][v] / K0 - 1 for v in ("lgd", "both", "naive")})
        irb[key].update({f"dRWA_{v}_usd": 12.5 * (irb[key][v] - K0) for v in ("lgd", "both", "naive")})
        irb[key]["dPD_ELweighted"] = float(((E * lgd0) @ (k["pd_mix"] - pi)) / ((E * lgd0) @ pi))
    num["irb"] = irb

    # ---- (3)+(4) reference run ----------------------------------------------------------------------
    active = np.flatnonzero(mdr[SCALES[-1]].max(axis=1) > 0)    # the largest stress has the most damaging events
    rng = np.random.default_rng([C.SEED, 71])                   # storm stream, independent of credit and of script 07
    L0 = {m: np.empty(C.N_REF) for m in models}
    L1 = {r: {m: np.empty(C.N_REF) for m in models} for r in ROWS}
    # Expected add-on of every scenario given its factor draw and storm year:
    #   dA = sum_k E_k [ I_k (LGD_k(e) - LGD_0k)  +  (p_k(Theta; d_k + shift_k) - p_k(Theta; d_k)) LGD_k(e) ].
    # The LGD part uses the simulated defaults; the PD part (the few defaults that the jump adds) is replaced by
    # its conditional expectation given Theta.  A threshold shift of 1e-4 standard deviations flips a default in
    # the 2,000 tail scenarios only a handful of times, so the simulated change of ES is pure discreteness noise;
    # the conditional add-on gives the first-order effect, as in eq. (firstorder) of the report.
    dA = {r: {m: np.empty(C.N_REF) for m in models} for r in ROWS}
    chunk = 100_000
    for i, Y, theta in P.reference_chunks(models, port, s, chunk=chunk):
        sl = slice(i, i + chunk)
        ev = climate.draw_event_years(freq, active, chunk, rng)  # the same storm years for every stress
        worst = {sc: climate.worst_from_events(m_, *ev, chunk, dtype=np.float64) for sc, m_ in mdr.items()}
        for m in models:
            I0 = Y[m] <= d[m]
            L0[m][sl] = port.loss(I0)
            p0 = port.conditional_pd(theta[m], d[m], s[m])
            for r, (_, ch, sig, sc) in ROWS.items():
                wk_ = worst[sc]
                lgd_e = climate.adjusted_lgd(R, wk_, C.PHI) if ch != "pd" else np.broadcast_to(lgd0, wk_.shape)
                lgd_part = (I0 * (lgd_e - lgd0)) @ E
                if ch == "lgd":
                    L1[r][m][sl] = L0[m][sl] + lgd_part
                    dA[r][m][sl] = lgd_part
                    continue
                shift = cc.threshold_shift(wk_, C.PHI, sig, s[m])
                I1 = Y[m] <= d[m] + shift
                L1[r][m][sl] = (I1 * lgd_e) @ E
                p1 = port.conditional_pd(theta[m], d[m] + shift, s[m])
                dA[r][m][sl] = lgd_part + ((p1 - p0) * lgd_e) @ E

    num["ref"] = {}
    k_tail = int(round(C.N_REF * (1 - C.ALPHA)))
    for r, (_, ch, sig, sc) in ROWS.items():
        num["ref"][r] = {"dEL_exact": exact_dEL(port, laws[sc], ch, sig)}
        for m in models:
            v0, e0 = var_es(L0[m], C.ALPHA)
            v1, e1 = var_es(L1[r][m], C.ALPHA)
            order = np.argsort(-L0[m], kind="stable")
            fo_es = float(dA[r][m][order[:k_tail]].mean())
            num["ref"][r][m] = {"dVaR_sim": v1 / v0 - 1, "dES_sim": e1 / e0 - 1, "dES_sim_usd": e1 - e0,
                                "dES": fo_es / e0, "dES_usd": fo_es,
                                "dEL_sim": float(L1[r][m].mean() / L0[m].mean() - 1),
                                "dEL_cond": float(dA[r][m].mean() / L0[m].mean())}
        num["ref"][r]["tail_over_mean_M2"] = num["ref"][r]["M2"]["dES"] / num["ref"][r]["dEL_exact"]
    num["pd_share_EL"] = num["ref"]["pd"]["dEL_exact"] / num["ref"]["both"]["dEL_exact"]
    num["pd_share_ES_M2"] = num["ref"]["pd"]["M2"]["dES"] / num["ref"]["both"]["M2"]["dES"]
    num["g20_ES_mult_M2"] = num["ref"]["g20"]["M2"]["dES"] / num["ref"]["both"]["M2"]["dES"]
    num["g20_EL_mult"] = num["ref"]["g20"]["dEL_exact"] / num["ref"]["both"]["dEL_exact"]
    P.save_numbers("numbers_q11x", num)

    # ---- tables ------------------------------------------------------------------------------------------
    pp = lambda x: P.sig(PPM * x)
    line = lambda k, cells: f"{k} & " + " & ".join(cells) + " \\\\"
    ml = list(models)
    body = []
    for r, (lab, ch, sig, sc) in ROWS.items():
        body.append(line(lab, [pp(num["ref"][r]["dEL_exact"])] + [pp(num["ref"][r][m]["dES"]) for m in ml]
                         + [f"{num['ref'][r]['M2']['dES_usd']:,.2f}"]))
        if r in ("both_s10",):
            body.append("\\midrule")
    (C.TABLES / "tab_climate_channels.tex").write_text(
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        " & & \\multicolumn{3}{c}{$\\Delta$ES$_{99.9\\%}$ (ppm)} & \\\\\n\\cmidrule(lr){3-5}\n"
        "Variant & $\\Delta$EL (ppm) & M1 & M2 & M3 & $\\Delta$ES M2 (USD) \\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")

    k25, k10 = irb["s25"], irb["s10"]
    rows_irb = [("Basel IRB (question x)", k25["base"], None, None),
                ("LGD add-on $F_k$ only", k25["lgd"], k25["d_lgd"], k10["d_lgd"]),
                ("Both channels (BIS formula)", k25["both"], k25["d_both"], k10["d_both"]),
                ("Naive: $P_k$ plugged into IRB, $\\times F_k$", k25["naive"], k25["d_naive"], k10["d_naive"])]
    (C.TABLES / "tab_climate_irb.tex").write_text(
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        " & $K$ (USD), $\\sigma_A=25\\%$ & \\multicolumn{2}{c}{$\\Delta K$ (ppm)} \\\\\n\\cmidrule(lr){3-4}\n"
        " & & $\\sigma_A=25\\%$ & $\\sigma_A=10\\%$ \\\\\n\\midrule\n"
        + "\n".join(line(lab, [P.usd(v, 2) if dd is not None else P.usd(v), "--" if dd is None else pp(dd),
                               "--" if d2 is None else pp(d2)]) for lab, v, dd, d2 in rows_irb)
        + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- figure: convexity of the hazard stress --------------------------------------------------------
    sc_all = [0.0, *SCALES]
    keys = {0.0: "both", 0.05: "g05", 0.10: "g10", 0.20: "g20"}
    fig, ax = plt.subplots(1, 2, figsize=(FIGW, 2.0))
    g = np.linspace(15, 60, 400)
    yi = np.interp(g, cat.extra["impf_intensity"], cat.extra["impf_mdr"])
    ax[0].semilogy(g[yi > 0], 100 * yi[yi > 0], color="k", lw=1.0)
    gi = np.asarray(cat.extra["inten"], float)
    for sc, ls in zip(sc_all, ("-", "--", "-.", ":")):
        ax[0].axvline(gi.max() * (1 + sc), color=COL["M2"], ls=ls, lw=0.8)
    ax[0].set_xlabel("Gust speed (m/s)"); ax[0].set_ylabel("MDR (%, log scale)")
    ax[0].set_title("Impact function; largest gust, +0/5/10/20%")
    x = 100 * np.array(sc_all)
    ax[1].plot(x, [PPM * num["ref"][keys[sc]]["dEL_exact"] for sc in sc_all], "o-", color="k", ms=3, lw=1, label="$\\Delta$EL (exact)")
    for m in ml:
        ax[1].plot(x, [PPM * num["ref"][keys[sc]][m]["dES"] for sc in sc_all], "s--", color=COL[m], ms=3, lw=0.8,
                   label=f"$\\Delta$ES {m}")
    ax[1].set_yscale("log"); ax[1].set_xlabel("Increase of all gust speeds (%)"); ax[1].set_ylabel("ppm (log scale)")
    ax[1].set_title("Both channels: effect of the gust stress"); ax[1].set_ylim(top=ax[1].get_ylim()[1] * 6); ax[1].legend(loc="upper left", ncol=2)
    save(fig, "fig_climate_stress")

    print(open(C.TABLES / "tab_climate_channels.tex").read())
    print(open(C.TABLES / "tab_climate_irb.tex").read())
    import json
    print(json.dumps({k: num[k] for k in ("bis", "stress", "irb")}, indent=1, default=float))
    print({k: num[k] for k in ("pd_share_EL", "pd_share_ES_M2", "g20_ES_mult_M2", "g20_EL_mult", "interp_check_maxdiff")})


if __name__ == "__main__":
    main()
