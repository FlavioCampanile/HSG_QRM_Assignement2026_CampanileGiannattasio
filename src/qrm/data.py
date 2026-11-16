"""Loading the data and building the weekly log-returns Theta = (SPI, SPX)  [feeds Q iv, ix]."""
import numpy as np
import pandas as pd
from . import config as C


def load_prices(start: str | None = C.SAMPLE_START) -> pd.DataFrame:
    """Daily SPI and SPX index levels on the *common* trading calendar.

    Cleaning steps (documented in the report, Section 2):
      * SPI dates are stored as text and are parsed as ISO dates (YYYY-MM-DD);
      * the SPI sheet opens with twelve monthly values for 1987 (1987-01-01, ..., 1987-12-01)
        before the daily series starts on 1988-01-04; they are not daily observations and
        are dropped (irrelevant for the 2000+ sample, but it keeps the loader correct for
        any start date);
      * only days on which both indices have a quote are kept (inner join), so that
        no return mixes a stale price with a fresh one;
      * the sample starts on `start` (1 January 2000, as stated in the assignment).
    """
    spi = pd.read_excel(C.MARKET_FILE, sheet_name="SPI")
    spx = pd.read_excel(C.MARKET_FILE, sheet_name="SPX")
    spi["Date"] = pd.to_datetime(spi["Date"], format="%Y-%m-%d")
    spx["Date"] = pd.to_datetime(spx["Date"])
    spi = spi[spi["Date"] >= "1988-01-04"].set_index("Date")["SPI"]
    spx = spx.set_index("Date")["Close"].rename("SPX")
    px = pd.concat([spi, spx], axis=1, join="inner").sort_index().dropna()
    if start is not None:
        px = px[px.index >= start]
    return px


def daily_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log-returns on the common calendar."""
    return np.log(prices).diff().dropna()


def weekly_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Non-overlapping weekly log-returns, Wednesday close to Wednesday close.

    Wednesday is used because it is the weekday least affected by holidays and by the
    weekend effect; if a Wednesday is not a common trading day the last available
    quote of that week (Thursday-to-Wednesday window) is used.  Non-overlapping
    returns are (approximately) serially independent, which is what the i.i.d.
    likelihoods of M2 and M3 require.
    """
    wk = np.log(prices).resample("W-WED").last().dropna()
    if prices.index[-1] < wk.index[-1]:          # last week still incomplete: drop it
        wk = wk.iloc[:-1]
    return wk.diff().dropna()


def overlapping_log_returns(daily: pd.DataFrame, h: int = C.HORIZON_DAYS) -> pd.DataFrame:
    """Overlapping h-day log-returns (sum of h consecutive daily log-returns).

    Used in the rolling analysis (Q ix): a window of 500 daily returns contains only
    100 non-overlapping weeks, but 496 overlapping 5-day returns.
    """
    return daily.rolling(h).sum().dropna()


def load_portfolio() -> pd.DataFrame:
    """100 counterparties: rating, E_k, R_k, pi_k, lambda_k, a_k1, a_k2, canton, coordinates."""
    df = pd.read_excel(C.PORTFOLIO_FILE, sheet_name="Counterparties")
    df = df.rename(columns={"Counterparty k": "k", "Exposure USD": "E", "R_k": "R", "pi_k": "pi",
                            "lambda_k": "lam", "a_k1": "a1", "a_k2": "a2"})
    df["LGD"] = 1.0 - df["R"]
    return df


def load_cantons() -> pd.DataFrame:
    """The 26 Swiss cantons: abbreviation, population, centroid coordinates  [Q xi]."""
    return pd.read_excel(C.PORTFOLIO_FILE, sheet_name="Swiss Cantons")
