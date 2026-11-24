"""Connect the results to the LaTeX report.

The report (report/main.tex and report/sections/*.tex) is ordinary, hand-written LaTeX.  Every number
quoted in the text is written as  \val{file.key.path|format} , for example \val{q8.M2.VaR_ref|usd}.
This script

  1. scans the .tex sources for these keys, looks each one up in the JSON files written by scripts
     01-07 (results/tables/numbers_*.json) and writes report/generated/numbers.tex, which defines them;
  2. copies the tables (.tex) and figures (.pdf) from results/ to report/tables and report/figures.

So the text can be edited freely (also on Overleaf), and when results change only numbers.tex, the
tables and the figures have to be replaced.  Keys whose results do not exist yet (question xi before
the CLIMADA run) are left undefined and show up in the PDF as a bold [key], never as a wrong number.
"""
import datetime as dt
import json
import re
import shutil
from qrm import config as C, pipeline as P

REPORT = C.REPORT
MINUS = "\\ensuremath{-}"                       # typographic minus that works in text and in math mode


def _grp(s):
    """1,234,567 -> 1{,}234{,}567 (no spurious space after the comma in math mode) and a proper minus."""
    return s.replace(",", "{,}").replace("-", MINUS)


def _signed(x, d):
    return (f"{x:+.{d}f}").replace("-", MINUS).replace("+", "\\ensuremath{+}")


FMT = {
    "usd": lambda x: _grp(f"{x:,.0f}"),
    "int": lambda x: _grp(f"{int(round(x)):,}"),
    "mn2": lambda x: _grp(f"{x / 1e6:.2f}"),
    "pct0": lambda x: _grp(f"{100 * x:.0f}"), "pct1": lambda x: _grp(f"{100 * x:.1f}"),
    "pct2": lambda x: _grp(f"{100 * x:.2f}"), "pct3": lambda x: _grp(f"{100 * x:.3f}"),
    "pct4": lambda x: _grp(f"{100 * x:.4f}"),
    "spct1": lambda x: _signed(100 * x, 1), "spct2": lambda x: _signed(100 * x, 2),
    "ratio_pct0": lambda x: f"{100 * x:.0f}", "inv_pct0": lambda x: f"{100 / x:.0f}",
    "excess_pct0": lambda x: f"{100 * (x - 1):.0f}",                       # a ratio r shown as (r - 1) in %
    "f0": lambda x: _grp(f"{x:,.0f}"), "f1": lambda x: _grp(f"{x:.1f}"), "f2": lambda x: _grp(f"{x:.2f}"),
    "f3": lambda x: _grp(f"{x:.3f}"), "f4": lambda x: _grp(f"{x:.4f}"),
    "bp": lambda x: _grp(P.sig(1e4 * x)),                                  # relative change in basis points
    "ppm": lambda x: _grp(P.sig(1e6 * x)),                                 # relative change in parts per million
    "usd2": lambda x: _grp(f"{x:,.2f}"),                                   # USD with cents
    "sig": lambda x: _grp(P.sig(x)),
    "excess_ppm": lambda x: _grp(P.sig(1e6 * (x - 1))),                    # a factor F shown as (F - 1) in ppm
    "usd0": lambda x: _grp(f"{x:,.0f}"),
    "sci": lambda x: (lambda m, e: f"\\ensuremath{{{m}\\times10^{{{e}}}}}")(*f"{x:.0e}".split("e")[0:1], int(f"{x:.0e}".split("e")[1])),
    "date": lambda s: f"{dt.date.fromisoformat(s).day} {dt.date.fromisoformat(s):%B %Y}",
    "str": lambda s: str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&"),
}


def load_numbers():
    num = {}
    for f in sorted(C.TABLES.glob("numbers_*.json")):
        if "DEMO" not in f.stem:                 # results from the synthetic storm catalogue are never reported
            num[f.stem.replace("numbers_", "")] = json.loads(f.read_text())
    q4, q8, q10 = num["q4"], num["q8"], num["q10"]
    ww = q8["worst_weeks_M1"]
    num["derived"] = {
        "M1_vs_M2_99": q8["M1"]["VaR99_ref"] / q8["M2"]["VaR99_ref"] - 1,
        "one_over_n": 1 / q4["n_weeks"],
        "cel1": ww[0]["cond_EL"], "cel2": ww[1]["cond_EL"], "cel3": ww[2]["cond_EL"],
        "cel1_week": ww[0]["week"], "cel2_week": ww[1]["week"], "cel3_week": ww[2]["week"],
        "cel1_spi": -ww[0]["SPI"], "cel1_spx": -ww[0]["SPX"],
        "K_share_B_CCC": q10["K_share_by_rating"]["B"] + q10["K_share_by_rating"]["CCC"],
        "K_lambda_ratio": q10["K_lambda"] / q10["K"],
        "max_thr_gap": max(abs(q8[m]["VaR_10k_quantile_thr"] / q8[m]["VaR_10k"] - 1) for m in ("M1", "M2", "M3")),
        "gran_share_UL_M2": q10["granularity_addon_M2"] / q8["M2"]["UL_ref"],
        "gran_share_VaR_max": max(q8[m]["granularity_share"] for m in ("M1", "M2", "M3")),
        "UL_M3_vs_M2": q8["M3"]["UL_ref"] / q8["M2"]["UL_ref"] - 1,
        "UL_M1_vs_M2": q8["M1"]["UL_ref"] / q8["M2"]["UL_ref"] - 1,
        "ES_over_VaR_min": min(q8[m]["ES_over_VaR"] for m in ("M1", "M2", "M3")) - 1,
        "ES_over_VaR_max": max(q8[m]["ES_over_VaR"] for m in ("M1", "M2", "M3")) - 1,
        "se_rel_min": min(q8[m][k] for m in ("M1", "M2", "M3") for k in ("se_rel_VaR_10k", "se_rel_VaR_10k_q")),
        "se_rel_max": max(q8[m][k] for m in ("M1", "M2", "M3") for k in ("se_rel_VaR_10k", "se_rel_VaR_10k_q")),
        "bias_min": min(q8[m][k] / q8[m]["VaR_ref"] - 1 for m in ("M1", "M2", "M3") for k in ("mean_VaR_10k", "mean_VaR_10k_q")),
        "bias_max": max(q8[m][k] / q8[m]["VaR_ref"] - 1 for m in ("M1", "M2", "M3") for k in ("mean_VaR_10k", "mean_VaR_10k_q")),
        "noise_ratio": q8["M3_minus_M2"]["ref_thr"]["sd_dVaR_crn"] / q8["M3_minus_M2"]["dVaR_ref"],
    }
    return num


def lookup(obj, parts):
    """Follow a dotted path; a key may itself contain dots (e.g. '0.01'), so try the longest match first."""
    if not parts:
        return obj
    if isinstance(obj, list):
        return lookup(obj[int(parts[0])], parts[1:])
    for n in range(len(parts), 0, -1):
        key = ".".join(parts[:n])
        if isinstance(obj, dict) and key in obj:
            try:
                return lookup(obj[key], parts[n:])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
    raise KeyError(".".join(parts))


def tex_minus(s):
    """In the tables written by pandas a negative number starts with a hyphen: use a math minus."""
    return re.sub(r"(?<=[\s&(])-(?=[\d.])", "$-$", s)


def main():
    num = load_numbers()
    sources = [REPORT / "main.tex"] + sorted((REPORT / "sections").rglob("*.tex"))
    uncommented = lambda t: "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in t.split("\n"))
    keys = sorted({k for f in sources for k in re.findall(r"\\val\{([^}]+)\}", uncommented(f.read_text(encoding="utf-8")))})
    lines, missing = [], []
    for k in keys:
        path, _, fmt = k.partition("|")
        try:
            v = lookup(num, path.split("."))
            lines.append(f"\\qrmdef{{{k}}}{{{FMT[fmt](v) if fmt else FMT['str'](v)}}}")
        except (KeyError, IndexError, TypeError, ValueError):
            missing.append(k)
    (REPORT / "generated").mkdir(exist_ok=True)
    (REPORT / "generated" / "numbers.tex").write_text(
        "% GENERATED by scripts/08_build_report.py - do not edit by hand.\n"
        "% One line per \\val{...} used in the report: \\qrmdef{key|format}{value}.\n" + "\n".join(lines) + "\n",
        encoding="utf-8")

    (REPORT / "tables").mkdir(exist_ok=True)
    (REPORT / "figures").mkdir(exist_ok=True)
    for f in sorted(C.TABLES.glob("*.tex")):
        if "DEMO" not in f.name:
            (REPORT / "tables" / f.name).write_text(tex_minus(f.read_text()))
    for f in sorted(C.FIGURES.glob("*.pdf")):
        if "DEMO" not in f.name:
            shutil.copy(f, REPORT / "figures" / f.name)

    print(f"report/generated/numbers.tex: {len(lines)} values defined, {len(missing)} pending")
    other = [k for k in missing if not k.startswith(("q11.",))]
    if missing and not other:
        print("  all pending values belong to question xi: run scripts 06 and 07, then this script again")
    for k in other:
        print("  UNKNOWN KEY (check the spelling):", k)


if __name__ == "__main__":
    main()
