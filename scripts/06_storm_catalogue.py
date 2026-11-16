"""Q xi, step 1 (the only script that needs CLIMADA) - build the storm catalogue.

Downloads the probabilistic winter-storm hazard for Switzerland through the CLIMADA data API (the
same call as in the ETH "mandatory exercise 1"), evaluates the Welker et al. (2021) impact function
at the location of every counterparty, and stores frequencies f_e and damage ratios MDR_k(e) in
results/climate/storm_catalogue.npz.  Script 07 then runs the overlay without CLIMADA.

Usage (inside the conda environment that has CLIMADA, e.g. `conda activate climada_env`)
    python scripts/06_storm_catalogue.py                  # hazard from the CLIMADA data API
    python scripts/06_storm_catalogue.py --hazard FILE    # hazard from a local .hdf5 file
"""
import argparse
import numpy as np
from qrm import config as C, data, climada_io, climate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hazard", default=None, help="local CLIMADA hazard file (.hdf5); default: data API")
    ap.add_argument("--no-crosscheck", action="store_true", help="skip the comparison with ImpactCalc")
    args = ap.parse_args()

    df, cantons = data.load_portfolio(), data.load_cantons()
    hazard, info = climada_io.load_hazard(args.hazard)
    impf = climada_io.welker_impact_function()
    cat = climada_io.build_catalogue(hazard, info, df, cantons, impf)

    if not args.no_crosscheck:
        # MDR_k(e) recomputed by CLIMADA's impact engine (exposure of value 1 at every counterparty)
        try:
            diff, aai = climada_io.cross_check_impactcalc(hazard, df, cat.mdr, impf)
            cat.meta.update(crosscheck_max_abs_diff=diff, crosscheck_aai_unit_exposure=aai)
            print(f"cross-check with ImpactCalc: max |MDR difference| = {diff:.3e}")
        except Exception as err:                       # version differences in the Exposures API: not fatal
            cat.meta["crosscheck_error"] = repr(err)
            print("cross-check with ImpactCalc skipped:", repr(err))

    C.CLIMATE.mkdir(parents=True, exist_ok=True)
    cat.save(C.CATALOGUE_FILE)

    # ---- a short log, so that problems with the hazard are visible immediately -----------------
    dmg = cat.mdr.max(axis=1) > 0
    Em = climate.expected_worst_mdr(cat.freq, cat.mdr)
    print("\n=== storm catalogue ===")
    for k, v in cat.meta.items():
        print(f"{k:28s} {v}")
    print(f"{'events':28s} {len(cat.freq)}   (damaging at >= 1 counterparty: {int(dmg.sum())})")
    print(f"{'sum of frequencies':28s} {cat.freq.sum():.4f} per year   (damaging: {cat.freq[dmg].sum():.4f})")
    print(f"{'frequency range':28s} {cat.freq.min():.3e} .. {cat.freq.max():.3e}")
    print(f"{'max gust at counterparties':28s} {cat.extra['inten'].max():.1f} {cat.meta['intensity_unit']}")
    print(f"{'max MDR_k(e)':28s} {cat.mdr.max():.3e}")
    print(f"{'E[worst MDR per year]':28s} mean {Em.mean():.3e}, min {Em.min():.3e}, max {Em.max():.3e}")
    print(f"{'distance to centroid (km)':28s} max {cat.extra['nearest_km'].max():.2f}")
    print(f"\nsaved {C.CATALOGUE_FILE}\nnext: python scripts/07_climate_overlay.py")
    if cat.extra["nearest_km"].max() > 25:
        print("WARNING: some counterparties are more than 25 km from the nearest hazard centroid.")
    if not np.isfinite(cat.mdr).all() or cat.mdr.max() > 1 or cat.mdr.min() < 0:
        print("WARNING: damage ratios outside [0, 1] - check the impact function.")


if __name__ == "__main__":
    main()
