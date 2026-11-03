"""Central constants, so every question uses identical settings (reproducibility)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

SEED = 2026
N_SIM = 10_000          # simulations (Q v, vi, xi)
ALPHA = 0.999           # confidence level (Q viii, x)
WINDOW = 500            # rolling window of daily returns (Q ix)
HORIZON_DAYS = 5        # T = 1 week
MATURITY = 1.0          # IRB effective maturity (Q x)
AVC = 1.0               # Basel III asset value correlation multiplier (Q x)
PHI = 0.5               # collateral sensitivity (Q xi)

MARKET_FILE = DATA_RAW / "SPI_SPX_daily_2000_2026.xlsx"
PORTFOLIO_FILE = DATA_RAW / "credit_portfolio_data.xlsx"
