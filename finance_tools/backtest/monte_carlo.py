"""
Monte Carlo stress-test engine for portfolio backtests.

Fetches historical data once for a wide date range, then slices into
random windows and runs all strategies in each window.  Collects results
into a DataFrame for statistical comparison.

Usage:
    from finance_tools.backtest.monte_carlo import (
        fetch_full_history, generate_windows, run_monte_carlo,
        mc_results_to_dataframe, summarize_mc_results, compute_win_rates,
    )
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class TimeWindow:
    """A single backtest time window."""
    start: date
    end: date
    iteration: int


@dataclass
class MCIterationResult:
    """Results from one strategy in one window."""
    iteration: int
    window_start: date
    window_end: date
    strategy: str
    sharpe: float
    ann_return: float
    ann_vol: float
    max_dd: float
    final_value: float
    total_return: float
    n_trades: int
    n_tickers_active: int


# =====================================================================
# Data fetching & slicing
# =====================================================================

def fetch_full_history(tickers: list[str],
                       start: str | date,
                       end: str | date) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for all tickers spanning the full date range.

    Returns {ticker: DataFrame}.  Tickers with no data are silently skipped.
    """
    hist_dict: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        hist = yf.Ticker(ticker).history(start=str(start), end=str(end))
        if hist.empty:
            continue
        hist_dict[ticker] = hist
    return hist_dict


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone and normalize to midnight timestamps."""
    df = df.copy()
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="first")]
    return df


def slice_history(hist_dict: dict[str, pd.DataFrame],
                  start: date,
                  end: date) -> dict[str, pd.DataFrame]:
    """Slice each ticker's DataFrame to [start, end].

    Tickers with no data in the window are excluded from the result.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    sliced: dict[str, pd.DataFrame] = {}
    for ticker, df in hist_dict.items():
        ndf = _normalize_index(df)
        subset = ndf.loc[(ndf.index >= start_ts) & (ndf.index <= end_ts)]
        if len(subset) > 0:
            sliced[ticker] = subset
    return sliced


# =====================================================================
# Window generation
# =====================================================================

def generate_windows(hist_dict: dict[str, pd.DataFrame],
                     window_years: float = 3.0,
                     n_iterations: int = 10,
                     seed: int = 603) -> list[TimeWindow]:
    """Pick random start dates within the valid range of the data.

    The valid range is determined by the earliest and latest dates across
    all tickers.  Start dates are chosen uniformly at random; each window
    spans ``window_years`` calendar years from its start.

    Returns a sorted list of TimeWindow objects.
    """
    # Find global date bounds
    all_min = None
    all_max = None
    for df in hist_dict.values():
        ndf = _normalize_index(df)
        if all_min is None or ndf.index.min() < all_min:
            all_min = ndf.index.min()
        if all_max is None or ndf.index.max() > all_max:
            all_max = ndf.index.max()

    if all_min is None or all_max is None:
        return []

    window_days = int(window_years * 365.25)
    latest_start = all_max - timedelta(days=window_days)
    if latest_start <= all_min:
        # Not enough data for even one full window
        return []

    rng = np.random.default_rng(seed)
    total_range = (latest_start - all_min).days
    offsets = rng.integers(0, total_range + 1, size=n_iterations)

    windows = []
    for i, offset in enumerate(offsets):
        start_date = (all_min + timedelta(days=int(offset))).date()
        end_date = start_date + timedelta(days=window_days)
        windows.append(TimeWindow(start=start_date, end=end_date,
                                  iteration=i))

    # Sort by start date
    windows.sort(key=lambda w: w.start)
    return windows


# =====================================================================
# MC runner
# =====================================================================

def run_monte_carlo(hist_dict: dict[str, pd.DataFrame],
                    strategies,
                    windows: list[TimeWindow],
                    initial_cash: float = 5_000.0,
                    cash_reserve_pct: float = 0.05,
                    rf_rate: float = 0.0,
                    rf_history: pd.Series | None = None) -> list[MCIterationResult]:
    """Run all strategies across all windows.

    Parameters
    ----------
    hist_dict : full-range historical data (fetch once, slice per window)
    strategies : list of PortfolioStrategy instances
    windows : list of TimeWindow from generate_windows()
    initial_cash : starting cash per window
    cash_reserve_pct : cash reserve fraction
    rf_rate : risk-free rate for Sharpe calculation (constant fallback)
    rf_history : daily annualized rf rate Series (sliced per window).
                 When provided, each window gets its own rf_daily slice.

    Returns
    -------
    List of MCIterationResult (one per strategy per window).
    """
    from finance_tools.backtest.portfolio import PortfolioBacktester

    results: list[MCIterationResult] = []

    for window in windows:
        sliced = slice_history(hist_dict, window.start, window.end)
        if len(sliced) < 2:
            continue

        # Slice rf_history to this window
        rf_daily_window = None
        if rf_history is not None and len(rf_history) > 0:
            start_ts = pd.Timestamp(window.start)
            end_ts = pd.Timestamp(window.end)
            rf_daily_window = rf_history.loc[
                (rf_history.index >= start_ts) & (rf_history.index <= end_ts)
            ]

        for strat in strategies:
            # Use strategy-level cash reserve if available, else global default
            strat_cash_pct = getattr(strat, "cash_reserve_pct", cash_reserve_pct)
            bt = PortfolioBacktester(
                sliced, strat,
                initial_cash=initial_cash,
                cash_reserve_pct=strat_cash_pct,
            )
            res = bt.run()
            res.rf_rate = rf_rate
            if rf_daily_window is not None:
                res.rf_daily = rf_daily_window

            results.append(MCIterationResult(
                iteration=window.iteration,
                window_start=window.start,
                window_end=window.end,
                strategy=res.strategy_name,
                sharpe=res.sharpe_ratio,
                ann_return=res.annualized_return,
                ann_vol=res.annualized_volatility,
                max_dd=res.max_drawdown,
                final_value=res.final_value,
                total_return=res.total_return,
                n_trades=res.n_trades,
                n_tickers_active=len(sliced),
            ))

    return results


# =====================================================================
# Aggregation
# =====================================================================

def mc_results_to_dataframe(results: list[MCIterationResult]) -> pd.DataFrame:
    """Convert MC results to a pandas DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "iteration": r.iteration,
            "window_start": r.window_start,
            "window_end": r.window_end,
            "strategy": r.strategy,
            "sharpe": r.sharpe,
            "ann_return": r.ann_return,
            "ann_vol": r.ann_vol,
            "max_dd": r.max_dd,
            "final_value": r.final_value,
            "total_return": r.total_return,
            "n_trades": r.n_trades,
            "n_tickers_active": r.n_tickers_active,
        })
    return pd.DataFrame(rows)


def summarize_mc_results(df: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy summary statistics.

    Returns a DataFrame indexed by strategy with columns for mean/median/std
    of Sharpe, annualized return, volatility, max drawdown, and win rate
    (fraction of iterations with positive Sharpe).
    """
    metrics = ["sharpe", "ann_return", "ann_vol", "max_dd"]
    agg_funcs = {m: ["mean", "median", "std"] for m in metrics}
    summary = df.groupby("strategy").agg(agg_funcs)

    # Flatten multi-level columns
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]

    # Win rate: fraction of iterations with positive Sharpe
    win = df.groupby("strategy")["sharpe"].apply(
        lambda s: (s > 0).mean()
    )
    summary["win_rate"] = win

    return summary


def compute_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise head-to-head Sharpe win-rate matrix.

    Returns a square DataFrame where entry [A, B] is the fraction of
    iterations where strategy A had a higher Sharpe than strategy B.
    Diagonal entries are NaN.
    """
    strategies = sorted(df["strategy"].unique())
    n = len(strategies)
    matrix = pd.DataFrame(np.nan, index=strategies, columns=strategies)

    # Group by iteration for pairwise comparison
    for i, s_a in enumerate(strategies):
        for j, s_b in enumerate(strategies):
            if i == j:
                continue
            a_sharpes = df[df["strategy"] == s_a].set_index("iteration")["sharpe"]
            b_sharpes = df[df["strategy"] == s_b].set_index("iteration")["sharpe"]
            common = a_sharpes.index.intersection(b_sharpes.index)
            if len(common) == 0:
                continue
            wins = (a_sharpes.loc[common] > b_sharpes.loc[common]).mean()
            matrix.loc[s_a, s_b] = wins

    return matrix
