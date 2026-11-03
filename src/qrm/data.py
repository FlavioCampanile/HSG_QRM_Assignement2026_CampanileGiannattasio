"""Loading data and building weekly log-returns of (SPI, SPX)  [feeds Q iv]."""
import numpy as np
import pandas as pd
from . import config as C


def load_prices() -> pd.DataFrame:
    """Daily SPI and SPX index levels, indexed by date."""
    raise NotImplementedError


def log_returns(prices: pd.DataFrame, horizon_days: int = C.HORIZON_DAYS) -> pd.DataFrame:
    """Log-returns over `horizon_days` trading days (weekly Theta).
    Decide and justify: overlapping vs non-overlapping weekly returns."""
    raise NotImplementedError


def load_portfolio() -> pd.DataFrame:
    """100 counterparties: rating, E_k, R_k, pi_k, a_k1, a_k2, lambda_k (+ climate columns)."""
    raise NotImplementedError
