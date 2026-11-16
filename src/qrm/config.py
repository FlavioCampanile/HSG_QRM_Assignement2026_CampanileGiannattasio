"""Central constants, so that every question uses identical settings (reproducibility)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
CLIMATE = ROOT / "results" / "climate"          # storm catalogue exported from CLIMADA (Q xi)
REPORT = ROOT / "report"

SEED = 2026
N_SIM = 10_000            # simulations required by the assignment (Q v, vi, xi)
N_REF = 2_000_000         # large "reference" run used to quantify Monte Carlo error
N_REP = 200               # independent replications of the N_SIM experiment (MC standard errors)
ALPHA = 0.999             # confidence level (Q viii, x)
SAMPLE_START = "2000-01-01"   # assignment: data from 1 January 2000
WINDOW = 500              # rolling window of daily returns (Q ix)
ROLL_STEP = 5             # re-estimate every 5 trading days (one week)
N_ROLL = 50_000           # simulations per rolling window (common random numbers)
HORIZON_DAYS = 5          # T = 1 week = 5 trading days
MATURITY = 1.0            # IRB effective maturity (Q x)
AVC = 1.0                 # Basel III asset-value-correlation multiplier (Q x)
PHI = 0.5                 # collateral sensitivity (Q xi)
N_STORM_REP = 500         # independent re-draws of the 10,000 storm years (Q xi: sampling error of the overlay)
WEEKS_PER_YEAR = 52       # horizon-consistent variant of the overlay: weekly frequencies f_e / 52
NU_MAX = 200.0            # upper bound for the t-copula degrees of freedom (Gaussian limit)

MARKET_FILE = DATA_RAW / "SPI_SPX_daily_2000_2026.xlsx"
PORTFOLIO_FILE = DATA_RAW / "credit_portfolio_data.xlsx"
CATALOGUE_FILE = CLIMATE / "storm_catalogue.npz"

# quadrature grid (trapezoidal rule on the two standardised factors) used to integrate
# the conditional default probability under M3 when calibrating the thresholds d_k
GMAX = 8.5                # grid covers +/- 8.5 standard deviations of each factor
QUAD_FULL = 0.10          # grid step, full-sample calibration
QUAD_ROLL = 0.20          # grid step, rolling windows (re-done 1,213 times)
N_PLACEBO = 200           # replications of the placebo experiment (Q ix)
