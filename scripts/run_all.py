"""Run the whole analysis in the order of the assignment questions and rebuild the report inputs.

    python scripts/run_all.py                  # everything; the rolling analysis is reused if present
    python scripts/run_all.py --force-rolling  # recompute the rolling analysis as well (about 15 minutes)
    python scripts/run_all.py --skip-climada   # do not call CLIMADA (script 06); use the stored catalogue

Script 06 is the only one that needs CLIMADA.  If CLIMADA is not installed in the active environment it
is skipped automatically and script 07 uses results/climate/storm_catalogue.npz if that file exists.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(script, *args):
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(HERE / script), *args], check=True)


if __name__ == "__main__":
    run("01_data_and_factor_models.py")
    run("02_simulate_losses.py")
    run("03_var_es.py")
    done = (ROOT / "results" / "tables" / "rolling_var_es.csv").exists()
    run("04_rolling_var_es.py", *([] if "--force-rolling" in sys.argv or not done else ["--plot-only"]))
    run("05_basel_irb.py")
    if "--skip-climada" not in sys.argv and importlib.util.find_spec("climada") is not None:
        run("06_storm_catalogue.py")
    else:
        print("\n=== 06_storm_catalogue.py skipped (CLIMADA not available or --skip-climada)")
    run("07_climate_overlay.py")
    run("07b_climate_channels.py")
    run("08_build_report.py")
