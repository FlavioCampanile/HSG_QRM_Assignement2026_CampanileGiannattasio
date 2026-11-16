"""Q xi, step 2 (no CLIMADA needed) - storm overlay on recovery rates.

Reads the storm catalogue written by script 06 and produces

  (1) the assignment's experiment: 10,000 storm years drawn event by event from N_e ~ Poisson(f_e),
      laid over the same 10,000 credit scenarios as in questions v-viii;
  (2) the sampling error of (1): the same credit scenarios with N_STORM_REP independent re-draws of
      the storm years;
  (3) a reference run on the two million credit scenarios of question viii, for the baseline and
      for five variants that answer the discussion points of the question (canton-level damage,
      phi = 1, weekly frequencies, a perfectly diversified portfolio, dependence between storm and
      credit years);
  (4) a sensitivity table that needs no catalogue (uniform damage ratio m for all names).

Usage
    python scripts/07_climate_overlay.py           # catalogue from results/climate/storm_catalogue.npz
    python scripts/07_climate_overlay.py --demo    # SYNTHETIC catalogue: tests the code, never reported
"""
import argparse
import numpy as np
import pandas as pd
from qrm import config as C, data, pipeline as P, climate
from qrm.risk_measures import var_es
from qrm.plotting import plt, COL, LAB, FIGW, save

PPM = 1e6                                                   # relative changes are reported in parts per million (1e-6)
VARIANTS = {                                                # reference-run variants: key -> row label
    "base": "Baseline",
    "canton": "MDR at canton centroid",
    "phi1": "$\\phi_k=1$",
    "weekly": "Weekly frequencies $f_e/52$",
    "diversified": "Independent storm years",
    "comonotone": "Comonotone storm/credit years",
}


# ------------------------------------------------------------------ (4) sensitivity, no catalogue
def sensitivity(models, port, I, grid=(0.001, 0.01, 0.05, 0.10, 0.25)):
    """Every counterparty suffers the same damage ratio m in every scenario: an upper bound for any
    catalogue whose damage ratios do not exceed m."""
    R = port.df["R"].to_numpy()
    rows, num = {}, {}
    for m in models:
        L0 = port.loss(I[m])
        v0, e0 = var_es(L0, C.ALPHA)
        num[m], r = {}, []
        for g in grid:
            v1, e1 = var_es(climate.overlay_losses(I[m], port.E, R, np.full((1, len(R)), g), C.PHI), C.ALPHA)
            num[m][f"dVaR_{g}"], num[m][f"dES_{g}"] = v1 / v0 - 1, e1 / e0 - 1
            r += [f"{100 * (v1 / v0 - 1):.2f}", f"{100 * (e1 / e0 - 1):.2f}"]
        rows[LAB[m]] = r
        # recovery value at stake relative to the loss, in the ten scenarios that make up the tail
        tail = L0 >= v0
        num[m]["tail_recovery_ratio"] = float((I[m][tail] @ (port.E * R)).sum() / L0[tail].sum())
    cols = pd.MultiIndex.from_product([[f"$m={100 * g:g}\\%$" for g in grid], ["VaR", "ES"]])
    tab = pd.DataFrame(rows, index=cols).T
    P.save_table("tab_climate_sensitivity", tab, multicolumn_format="c")
    num["recovery_ratio_portfolio"] = float((port.E * R).sum() / port.w.sum())
    P.save_numbers("numbers_q11s", num)


# ------------------------------------------------------------------ description of the catalogue
def hazard_summary(cat, port, df):
    freq, mdr = cat.freq, cat.mdr
    dmg = mdr.max(axis=1) > 0
    Em, Em_c = climate.expected_worst_mdr(freq, mdr), climate.expected_worst_mdr(freq, cat.mdr_canton)
    inten = cat.extra["inten"]
    by_canton = pd.Series(Em, index=df["canton_abbr"].to_numpy()).groupby(level=0).mean().sort_values()
    w = port.E * df["R"].to_numpy()                              # recovery value E_k R_k at stake
    ev_loss = mdr @ w / w.sum()                                  # share of recovery value destroyed by event e
    num = {"n_events": int(len(freq)), "n_damaging": int(dmg.sum()),
           "sum_freq": float(freq.sum()), "sum_freq_damaging": float(freq[dmg].sum()),
           "p_year_event": float(-np.expm1(-freq.sum())), "p_year_damage": float(-np.expm1(-freq[dmg].sum())),
           "freq_min": float(freq.min()), "freq_max": float(freq.max()),
           "return_period_rarest": float(1 / freq.min()),
           "gust_max": float(inten.max()), "gust_p99_damaging": float(np.quantile(inten[dmg], 0.99)) if dmg.any() else 0.0,
           "mdr_max": float(mdr.max()), "mdr_mean_positive": float(mdr[mdr > 0].mean()) if (mdr > 0).any() else 0.0,
           "share_pairs_positive": float((mdr > 0).mean()),
           "worst_event_recovery_share": float(ev_loss.max()),
           "share_names_hit_by_top_event": float((mdr[int(np.argmax(ev_loss))] > 0).mean()),
           "exp_worst_mean": float(Em.mean()), "exp_worst_min": float(Em.min()), "exp_worst_max": float(Em.max()),
           "exp_worst_weighted": float(Em @ w / w.sum()), "exp_worst_canton_mean": float(Em_c.mean()),
           "exp_worst_point_vs_canton_maxdiff": float(np.abs(Em - Em_c).max()),
           "corr_point_canton": float(np.corrcoef(mdr.ravel(), cat.mdr_canton.ravel())[0, 1]) if mdr.std() > 0 else 1.0,
           "canton_low": str(by_canton.index[0]), "canton_low_value": float(by_canton.iloc[0]),
           "canton_high": str(by_canton.index[-1]), "canton_high_value": float(by_canton.iloc[-1]),
           "nearest_km_max": float(cat.extra["nearest_km"].max()), "nearest_km_mean": float(cat.extra["nearest_km"].mean()),
           "meta": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in cat.meta.items()}}
    # pairwise dependence of the damage ratios across names: average correlation over events (frequency-weighted)
    if dmg.sum() > 2:
        Z = mdr[dmg]
        cw = np.cov(Z.T, aweights=freq[dmg])
        sd = np.sqrt(np.diag(cw))
        ok = sd > 0
        corr = cw[np.ix_(ok, ok)] / np.outer(sd[ok], sd[ok])
        num["mdr_pairwise_corr_mean"] = float(corr[~np.eye(ok.sum(), dtype=bool)].mean())
    return num, Em


def hazard_table(h, tag):
    f = lambda x, d=2: f"{x:,.{d}f}"
    rows = [("Events in the catalogue", f"{h['n_events']:,}"),
            ("\\quad with damage at one or more counterparties", f"{h['n_damaging']:,}"),
            ("Sum of annual frequencies $\\sum_e f_e$", f(h["sum_freq"], 3)),
            ("\\quad damaging events only", f(h["sum_freq_damaging"], 3)),
            ("Probability of a year with a damaging event (\\%)", f(100 * h["p_year_damage"], 1)),
            ("Largest gust speed at a counterparty (m/s)", f(h["gust_max"], 1)),
            ("Largest damage ratio $\\max_{e,k}\\mathrm{MDR}_k(e)$ (\\%)", f(100 * h["mdr_max"], 3)),
            ("Mean damage ratio when positive (ppm)", P.sig(1e6 * h["mdr_mean_positive"])),
            ("Expected worst damage ratio per year $\\mathrm{E}[m_k]$, mean over $k$ (ppm)", P.sig(1e6 * h["exp_worst_mean"])),
            ("\\quad smallest and largest across $k$ (ppm)", f"{P.sig(1e6 * h['exp_worst_min'])} -- {P.sig(1e6 * h['exp_worst_max'])}"),
            ("Recovery value destroyed by the worst event (ppm)", P.sig(1e6 * h["worst_event_recovery_share"]))]
    body = "\n".join(f"{a} & {b} \\\\" for a, b in rows)
    (C.TABLES / f"tab_climate_hazard{tag}.tex").write_text(
        "\\begin{tabular}{lr}\n\\toprule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")
    pd.DataFrame(rows, columns=["quantity", "value"]).to_csv(C.TABLES / f"tab_climate_hazard{tag}.csv", index=False)


# ------------------------------------------------------------------ (3) reference run
def reference_overlay(cat, models, port, s, d, ER, pi, Em):
    """Baseline loss and overlay loss for every variant on the N_REF scenarios of question viii.

    Also returns A[m] = sum_k E_k R_k E[m_k] I_k, the expected add-on of every credit scenario given
    its defaults (storms are independent of the credit scenario).  It gives the first-order effect of
    the overlay without any storm simulation (derivative of VaR and ES in the direction of the add-on):
        dVaR ~ phi E[A | L = VaR],        dES ~ phi E[A | L >= VaR]."""
    freq, mdr, mdr_c = cat.freq, cat.mdr, cat.mdr_canton
    active = np.flatnonzero((mdr.max(axis=1) > 0) | (mdr_c.max(axis=1) > 0))
    rng = np.random.default_rng([C.SEED, 62])                 # storm stream, independent of the credit stream
    L0 = {m: np.empty(C.N_REF) for m in models}
    L1 = {v: {m: np.empty(C.N_REF) for m in models} for v in VARIANTS}
    A = {m: np.empty(C.N_REF) for m in models}
    chunk = 100_000
    for i, Y, _ in P.reference_chunks(models, port, s, chunk=chunk):
        sl = slice(i, i + chunk)
        if active.size:
            ev = climate.draw_event_years(freq, active, chunk, rng)
            w_point = climate.worst_from_events(mdr, *ev, chunk)
            w_canton = climate.worst_from_events(mdr_c, *ev, chunk)      # same storm years, canton-level damage
            w_week = climate.worst_from_events(mdr, *climate.draw_event_years(freq / C.WEEKS_PER_YEAR, active, chunk, rng), chunk)
        else:
            w_point = w_canton = w_week = np.zeros((chunk, mdr.shape[1]), dtype=np.float32)
        w_div = climate.sample_worst_independent(freq, mdr, chunk, rng)
        severity = w_point @ (ER * pi).astype(np.float32)     # expected add-on of the storm year
        for m in models:
            I = Y[m] <= d[m]
            base = port.loss(I)
            L0[m][sl] = base
            A[m][sl] = I @ (ER * Em)
            a_point = climate.addon(I, ER, w_point)
            L1["base"][m][sl] = base + C.PHI * a_point
            L1["phi1"][m][sl] = base + 1.0 * a_point
            L1["canton"][m][sl] = base + C.PHI * climate.addon(I, ER, w_canton)
            L1["weekly"][m][sl] = base + C.PHI * climate.addon(I, ER, w_week)
            L1["diversified"][m][sl] = base + C.PHI * climate.addon(I, ER, w_div)
            # comonotone coupling within the chunk: worst storm years paired with largest credit losses
            L1["comonotone"][m][sl] = base + C.PHI * climate.addon(I, ER, w_point[climate.comonotone_order(severity, base)])
    return L0, L1, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic catalogue (code test only)")
    ap.add_argument("--catalogue", default=str(C.CATALOGUE_FILE))
    args = ap.parse_args()
    tag = "_DEMO" if args.demo else ""

    px, wk, models, port, s, d = P.setup()
    df = port.df
    R, pi = df["R"].to_numpy(), port.pi
    ER = port.E * R                                           # recovery value at stake, E_k R_k
    K = len(R)

    # the 10,000 credit scenarios of questions v-viii, with the empirical-quantile thresholds
    Y = P.simulate_baseline(models, port, s)
    I = {m: Y[m] <= port.thresholds_quantile(Y[m]) for m in models}
    L0 = {m: port.loss(I[m]) for m in models}
    sensitivity(models, port, I)                              # needs no catalogue: always produced

    if args.demo:
        cat = climate.demo_catalogue(df["latitude"], df["longitude"], df["canton_abbr"], np.random.default_rng(0))
    else:
        from pathlib import Path
        if not Path(args.catalogue).exists():
            print(f"No storm catalogue at {args.catalogue}: only the sensitivity table was produced.\n"
                  "Run scripts/06_storm_catalogue.py first (it needs CLIMADA).")
            return
        cat = climate.StormCatalogue.load(args.catalogue)
        if cat.synthetic:
            raise SystemExit("The catalogue file is synthetic; use --demo for tests.")
    assert cat.mdr.shape[1] == K, "catalogue and portfolio have a different number of counterparties"
    freq, mdr = cat.freq, cat.mdr

    haz, Em = hazard_summary(cat, port, df)
    # Annual damage to the portfolio's recovery value, D = sum_k E_k R_k m_k / sum_k E_k R_k, over 200,000
    # simulated storm years: its distribution is extremely skewed (most years cause almost no damage),
    # which is why the add-on at a single order statistic (the VaR) is usually far below its mean.
    wy = climate.sample_worst_fast(freq, mdr, 200_000, np.random.default_rng([C.SEED, 63]), dtype=np.float64)
    D = wy @ ER / ER.sum()
    Ds = np.sort(D)[::-1]
    haz.update({"D_mean": float(D.mean()), "D_median": float(np.median(D)), "D_q99": float(np.quantile(D, 0.99)),
                "D_max": float(D.max()), "D_share_zero": float((D == 0).mean()),
                "D_share_of_mean_top1pct": float(Ds[: len(Ds) // 100].sum() / Ds.sum()) if Ds.sum() > 0 else 0.0,
                "D_median_over_mean": float(np.median(D) / D.mean()) if D.mean() > 0 else 0.0})
    # recovery value relative to the loss given default, EL-weighted (sum E R pi / sum E (1-R) pi)
    haz["recovery_ratio_EL_weighted"] = float((ER * pi).sum() / (port.w * pi).sum())
    hazard_table(haz, tag)
    num = {"hazard": haz, "phi": C.PHI}
    # expected loss: storms and defaults are independent, so E[L_clim] = EL + phi sum_k E_k R_k pi_k E[m_k]
    num["EL_base"] = port.EL
    num["EL_clim_exact"] = port.EL + C.PHI * float(ER * pi @ Em)
    num["dEL_exact"] = num["EL_clim_exact"] / port.EL - 1
    Em_week = climate.expected_worst_mdr(freq / C.WEEKS_PER_YEAR, mdr)
    num["dEL_exact_weekly"] = C.PHI * float(ER * pi @ Em_week) / port.EL
    num["dEL_exact_phi1"] = float(ER * pi @ Em) / port.EL

    # ---- (1) the assignment's experiment ---------------------------------------------------------
    rng = np.random.default_rng([C.SEED, 6])
    worst, n_ev = climate.sample_worst_literal(freq, mdr, C.N_SIM, rng)
    num["assign"] = {"share_years_with_damage": float((n_ev > 0).mean()),
                     "max_events_in_a_year": int(n_ev.max()),
                     "share_name_years_hit": float((worst > 0).mean()),
                     "mean_worst_mdr_if_hit": float(worst[worst > 0].mean()) if (worst > 0).any() else 0.0,
                     "mean_lgd_base": float(port.lgd.mean()),
                     "mean_lgd_clim": float(climate.adjusted_lgd(R, worst, C.PHI).mean())}
    L1 = {}
    for m in models:
        L1[m] = climate.overlay_losses(I[m], port.E, R, worst, C.PHI)
        (v0, e0), (v1, e1) = var_es(L0[m], C.ALPHA), var_es(L1[m], C.ALPHA)
        tail = L0[m] >= v0
        num["assign"][m] = {"VaR_base": v0, "ES_base": e0, "VaR_clim": v1, "ES_clim": e1,
                            "dVaR": v1 / v0 - 1, "dES": e1 / e0 - 1, "dVaR_usd": v1 - v0, "dES_usd": e1 - e0,
                            "mean_base": float(L0[m].mean()), "mean_clim": float(L1[m].mean()),
                            "dEL": float(L1[m].mean() / L0[m].mean() - 1),
                            "n_tail": int(tail.sum()),
                            "n_tail_with_addon": int((L1[m][tail] > L0[m][tail]).sum()),
                            "max_rel_addon": float(((L1[m] - L0[m]) / np.maximum(L0[m], 1e-12))[L0[m] > 0].max())}

    # ---- (2) sampling error of the storm draw: same credit scenarios, new storm years ------------
    rng = np.random.default_rng([C.SEED, 61])
    rep = {m: np.empty((C.N_STORM_REP, 2)) for m in models}
    base = {m: var_es(L0[m], C.ALPHA) for m in models}
    for r in range(C.N_STORM_REP):
        w = climate.sample_worst_fast(freq, mdr, C.N_SIM, rng)
        for m in models:
            v1, e1 = var_es(L0[m] + C.PHI * climate.addon(I[m], ER, w), C.ALPHA)
            rep[m][r] = v1 / base[m][0] - 1, e1 / base[m][1] - 1
    num["rep"] = {"n_rep": C.N_STORM_REP}
    for m in models:
        dv, de = rep[m][:, 0], rep[m][:, 1]
        num["rep"][m] = {"dVaR_mean": dv.mean(), "dVaR_sd": dv.std(ddof=1), "dVaR_q05": np.quantile(dv, .05),
                         "dVaR_q95": np.quantile(dv, .95), "dVaR_max": dv.max(), "dVaR_share_zero": float((dv <= 1e-12).mean()),
                         "dES_mean": de.mean(), "dES_sd": de.std(ddof=1), "dES_q05": np.quantile(de, .05),
                         "dES_q95": np.quantile(de, .95), "dES_max": de.max(), "dES_share_zero": float((de <= 1e-12).mean())}

    # ---- (3) reference run and variants ------------------------------------------------------------
    R0, R1, A = reference_overlay(cat, models, port, s, d, ER, pi, Em)
    num["ref"] = {v: {} for v in VARIANTS}
    num["first_order"] = {}
    for m in models:
        v0, e0 = var_es(R0[m], C.ALPHA)
        # first-order effect: average expected add-on of the scenarios at (VaR) and beyond (ES) the quantile.
        # "At the quantile" = losses ranked between k/2 and 2k from the top, k = N(1-alpha) = 2,000.
        order = np.argsort(-R0[m], kind="stable")
        k = int(round(C.N_REF * (1 - C.ALPHA)))
        num["first_order"][m] = {"dVaR": C.PHI * float(A[m][order[k // 2: 2 * k]].mean()) / v0,
                                 "dES": C.PHI * float(A[m][order[:k]].mean()) / e0,
                                 "dEL": C.PHI * float(A[m].mean()) / float(R0[m].mean()),
                                 "mean_addon_usd": C.PHI * float(A[m].mean()),
                                 "addon_tail_usd": C.PHI * float(A[m][order[:k]].mean()),
                                 "dVaR_usd": C.PHI * float(A[m][order[k // 2: 2 * k]].mean()),
                                 "mean_tail_addon_usd": C.PHI * float(A[m][order[:k]].mean())}
        for v in VARIANTS:
            v1, e1 = var_es(R1[v][m], C.ALPHA)
            num["ref"][v][m] = {"VaR_base": v0, "ES_base": e0, "VaR_clim": v1, "ES_clim": e1,
                                "dVaR": v1 / v0 - 1, "dES": e1 / e0 - 1, "dVaR_usd": v1 - v0, "dES_usd": e1 - e0,
                                "dEL": float(R1[v][m].mean() / R0[m].mean() - 1)}
    b = num["ref"]["base"]
    for v in VARIANTS:                                       # each variant relative to the baseline overlay
        for m in models:
            num["ref"][v][m]["dVaR_vs_base"] = (num["ref"][v][m]["dVaR"] / b[m]["dVaR"]) if b[m]["dVaR"] > 0 else float("nan")
            num["ref"][v][m]["dES_vs_base"] = (num["ref"][v][m]["dES"] / b[m]["dES"]) if b[m]["dES"] > 0 else float("nan")
            num["ref"][v][m]["tail_multiplier"] = (num["ref"][v][m]["dVaR"] / num["ref"][v][m]["dEL"]) if num["ref"][v][m]["dEL"] > 0 else float("nan")
    P.save_numbers(f"numbers_q11{tag}", num)

    # ---- tables --------------------------------------------------------------------------------------
    u, pp = P.usd, lambda x: P.sig(PPM * x)
    c2 = lambda x: f"{x:,.2f}"                               # USD with cents: the add-ons are a few dollars
    line = lambda k, cells: f"{k} & " + " & ".join(cells) + " \\\\"
    blocks = []
    for title, src in (("Assignment run: $N=10{,}000$, thresholds $\\hat d_k$", num["assign"]),
                       ("Reference run: $N=2{,}000{,}000$, thresholds $d_k$", num["ref"]["base"])):
        blocks.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}} \\\\\n" + "\n".join(
            line(LAB[m], [u(src[m]["VaR_base"]), c2(src[m]["dVaR_usd"]), pp(src[m]["dVaR"]),
                          u(src[m]["ES_base"]), c2(src[m]["dES_usd"]), pp(src[m]["dES"])]) for m in models))
    (C.TABLES / f"tab_climate{tag}.tex").write_text(
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        " & \\multicolumn{3}{c}{VaR$_{99.9\\%}$} & \\multicolumn{3}{c}{ES$_{99.9\\%}$} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "Model & no overlay & $\\Delta$ USD & $\\Delta$ ppm & no overlay & $\\Delta$ USD & $\\Delta$ ppm \\\\\n\\midrule\n"
        + "\n\\midrule\n".join(blocks) + "\n\\bottomrule\n\\end{tabular}\n")
    ml = list(models)
    body = "\n".join(line(lab, [pp(num["ref"][v]["M2"]["dEL"])] + [pp(num["ref"][v][m]["dVaR"]) for m in ml]
                          + [pp(num["ref"][v][m]["dES"]) for m in ml]) for v, lab in VARIANTS.items())
    fo = num["first_order"]
    body = (line("First-order approximation", [pp(num["dEL_exact"])] + [pp(fo[m]["dVaR"]) for m in ml]
                 + [pp(fo[m]["dES"]) for m in ml]) + "\n\\midrule\n" + body)
    (C.TABLES / f"tab_climate_variants{tag}.tex").write_text(
        "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
        " & & \\multicolumn{3}{c}{$\\Delta$VaR$_{99.9\\%}$ (ppm)} & \\multicolumn{3}{c}{$\\Delta$ES$_{99.9\\%}$ (ppm)} \\\\\n"
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
        "Variant & $\\Delta$EL (ppm) & M1 & M2 & M3 & M1 & M2 & M3 \\\\\n\\midrule\n" + body
        + "\n\\bottomrule\n\\end{tabular}\n")
    pd.DataFrame({v: {f"{q}_{m}": num["ref"][v][m][q] for m in ml for q in ("dEL", "dVaR", "dES")} for v in VARIANTS}).T \
        .to_csv(C.TABLES / f"tab_climate_variants{tag}.csv")

    # ---- figure: hazard, exposure and vulnerability ------------------------------------------------
    x = cat.extra
    fig, ax = plt.subplots(1, 2, figsize=(FIGW, 2.25), gridspec_kw={"width_ratios": [1.25, 1]})
    foot = np.asarray(x["footprint_top"], float)
    keep = np.flatnonzero(foot > 0)
    if len(keep) > 25_000:                                   # thin the centroids: the map is only illustrative
        keep = keep[:: len(keep) // 25_000 + 1]
    sc = ax[0].scatter(x["cent_lon"][keep], x["cent_lat"][keep], c=foot[keep], s=2, cmap="YlOrRd", linewidths=0, rasterized=True)
    ax[0].scatter(df["longitude"], df["latitude"], s=4 + 40 * port.E / port.E.max(), facecolors="none", edgecolors="k", linewidths=0.5)
    lo_, la_ = df["longitude"], df["latitude"]
    ax[0].set_xlim(lo_.min() - 0.5, lo_.max() + 0.5); ax[0].set_ylim(la_.min() - 0.35, la_.max() + 0.35)
    ax[0].set_aspect(1 / np.cos(np.deg2rad(46.8))); ax[0].grid(False)
    ax[0].set_xlabel("Longitude"); ax[0].set_ylabel("Latitude")
    ax[0].set_title("Most damaging event and counterparties")
    fig.colorbar(sc, ax=ax[0], shrink=0.8, pad=0.02).set_label("Gust (m/s)")
    gi = np.asarray(x["inten"], float)
    gmax = max(60.0, float(gi.max()) + 3)
    xi = np.asarray(x["impf_intensity"], float); yi = np.asarray(x["impf_mdr"], float)
    sel = xi <= gmax
    ax[1].semilogy(xi[sel & (yi > 0)], 100 * yi[sel & (yi > 0)], color=COL["M2"], lw=1.2)
    ax[1].set_xlabel("Gust speed (m/s)"); ax[1].set_ylabel("MDR (%, log scale)", color=COL["M2"])
    ax[1].set_xlim(0, gmax); ax[1].set_title("Impact function and gusts")
    a2 = ax[1].twinx()
    a2.hist(gi[gi > 0], bins=np.arange(0, gmax + 2, 2), weights=np.broadcast_to(freq[:, None], gi.shape)[gi > 0] / K,
            color=COL["M1"], alpha=0.35)
    a2.set_ylabel("Events per year and site", color=COL["M1"]); a2.grid(False); a2.spines["right"].set_visible(True)
    save(fig, f"fig_climate_hazard{tag}")

    # ---- figure: size of the effect ------------------------------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(FIGW, 3.9), gridspec_kw={"height_ratios": [1, 1.25]})
    for j, m in enumerate(models):
        for k_, (col, off) in enumerate(((0, -0.2), (1, 0.2))):
            bx = ax[0].boxplot(PPM * rep[m][:, col], positions=[j + off], widths=0.32, patch_artist=True, showfliers=False,
                               whis=(5, 95))
            bx["boxes"][0].set(facecolor=COL[m], alpha=0.35, hatch=None if col == 0 else "///"); bx["medians"][0].set(color="k")
            ax[0].plot(j + off, PPM * num["assign"][m]["dVaR" if col == 0 else "dES"], "o", color="k", ms=3, zorder=5)
            # reference: first-order value, i.e. the expected increase over storm draws (formula in the report)
            ax[0].plot([j + off - 0.18, j + off + 0.18], [PPM * num["first_order"][m]["dVaR" if col == 0 else "dES"]] * 2,
                       "-", color=COL[m], lw=1.6, zorder=4)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels([LAB[m] for m in models])
    ax[0].set_ylabel("Increase (ppm)")
    ax[0].set_title("Sampling error at $N=10{,}000$: VaR (plain), ES (hatched)")
    vk = list(VARIANTS)
    short = {"base": "Baseline", "canton": "Canton MDR", "phi1": "$\\phi=1$", "weekly": "$f_e/52$",
             "diversified": "Diversified", "comonotone": "Comonotone"}
    vals = np.array([[PPM * num["ref"][v][m]["dES"] for m in models] for v in vk])     # ES: the stable measure
    floor = max(vals[vals > 0].min() / 3, 1e-6) if (vals > 0).any() else 1e-6
    for j, m in enumerate(models):
        ax[1].bar(np.arange(len(vk)) + (j - 1) * 0.27, np.maximum(vals[:, j], floor), width=0.25, color=COL[m], alpha=0.85, label=LAB[m])
    ax[1].set_yscale("log"); ax[1].set_xticks(range(len(vk))); ax[1].set_xticklabels([short[v] for v in vk])
    ax[1].set_ylabel("$\\Delta$ES$_{99.9\\%}$ (ppm, log scale)"); ax[1].set_title("Reference run: baseline overlay and variants")
    ax[1].legend(ncol=3, loc="upper left")
    save(fig, f"fig_climate_effect{tag}")

    print(open(C.TABLES / f"tab_climate{tag}.tex").read())
    print(open(C.TABLES / f"tab_climate_variants{tag}.tex").read())
    print({k: v for k, v in haz.items() if k != "meta"})


if __name__ == "__main__":
    main()
