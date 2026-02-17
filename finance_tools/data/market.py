"""
Shared market data utilities.

Provides functions for fetching market-wide data (risk-free rates, etc.)
that are used across multiple scripts.

Usage:
    from finance_tools.data.market import fetch_risk_free_rate, fetch_risk_free_history
"""

from datetime import date

import pandas as pd
import yfinance as yf


def fetch_risk_free_rate() -> float:
    """Fetch the current 3-month T-bill yield from yfinance (^IRX).

    Returns the annualized rate as a decimal (e.g. 0.045 for 4.5%).
    Falls back to 0.0 on any error.
    """
    try:
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        if len(hist) > 0:
            # ^IRX reports yield in percentage points (e.g. 4.25 = 4.25%)
            rate = float(hist["Close"].iloc[-1]) / 100.0
            if 0 < rate < 0.20:  # sanity check: 0-20%
                return rate
    except Exception:
        pass
    return 0.0


def fetch_risk_free_history(start: str | date,
                            end: str | date | None = None) -> pd.Series:
    """Fetch daily 3-month T-bill yield history from yfinance (^IRX).

    Parameters
    ----------
    start : start date (inclusive)
    end : end date (inclusive). Defaults to today.

    Returns
    -------
    pd.Series indexed by tz-naive normalized dates, with values being the
    annualized rate as a decimal (e.g. 0.045 for 4.5%).  Forward-filled to
    cover weekends/holidays.

    Falls back to a constant 0.0 Series on any error.
    """
    start_str = str(start)
    end_str = str(end) if end is not None else str(date.today())
    try:
        irx = yf.Ticker("^IRX")
        hist = irx.history(start=start_str, end=end_str)
        if len(hist) == 0:
            return pd.Series(dtype=float)

        # Normalize index: strip tz, dedupe
        if hasattr(hist.index, "tz") and hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        hist.index = pd.to_datetime(hist.index).normalize()
        hist = hist[~hist.index.duplicated(keep="first")]

        # Convert from percentage points to decimal
        rf = hist["Close"] / 100.0

        # Sanity filter: clamp to [0, 0.20]
        rf = rf.clip(0.0, 0.20)

        return rf
    except Exception:
        return pd.Series(dtype=float)
