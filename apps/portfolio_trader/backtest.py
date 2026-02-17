"""
Backtest the Portfolio Trader's whole-share inv-vol EW algorithm.

Simulates 2 years of daily trading starting with $5,000 and $650
monthly contributions, using the exact same compute_target_trades()
logic from app.py.

Usage:
    python backtest.py
"""

import os

import numpy as np
import pandas as pd
import yfinance as yf

from finance_tools.strategies.equal_weight import compute_target_trades, CASH_RESERVE_PCT
from finance_tools.backtest.portfolio import PortfolioState
from finance_tools.data.market import fetch_risk_free_rate, fetch_risk_free_history
from finance_tools.data.universe import TRADING_ASSISTANT_10
from finance_tools.utils.plotting import (
    setup_style, COLORS, PALETTE, FIGSIZE, savefig,
)

from app import DEFAULT_TICKERS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TICKERS = DEFAULT_TICKERS
INITIAL_CASH = 5_000.0
MONTHLY_CONTRIBUTION = 650.0
PERIOD = "2y"

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_all(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Download 2 years of daily OHLCV for each ticker."""
    print("Fetching 2 years of historical data...")
    data = {}
    for t in tickers:
        ticker = yf.Ticker(t)
        hist = ticker.history(period=PERIOD)
        if len(hist) == 0:
            print(f"  Warning: no data for {t}")
            continue
        # Normalize index to tz-naive dates
        if hasattr(hist.index, "tz") and hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        hist.index = pd.to_datetime(hist.index).normalize()
        hist = hist[~hist.index.duplicated(keep="first")]
        data[t] = hist
        print(f"  {t}: {len(hist)} days, ${hist['Close'].iloc[-1]:.2f}")
    print()
    return data


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_backtest(data: dict[str, pd.DataFrame]):
    """Simulate the trading assistant algorithm over historical data."""
    tickers = sorted(data.keys())

    # Union of all trading dates
    all_dates = sorted(set().union(*(set(df.index) for df in data.values())))
    print(f"Simulation: {all_dates[0].date()} to {all_dates[-1].date()} "
          f"({len(all_dates)} trading days, {len(tickers)} tickers)\n")

    # State
    cash = INITIAL_CASH
    positions = {t: 0 for t in tickers}
    total_contributed = INITIAL_CASH
    prev_month = None

    # Tracking
    dates = []
    daily_values = []
    daily_cash = []
    daily_contributions = []
    trade_count = 0
    trade_log = []

    for i, date in enumerate(all_dates):
        contribution = 0.0
        current_month = (date.year, date.month)

        # Monthly contribution on first trading day of new month
        if prev_month is not None and current_month != prev_month:
            contribution = MONTHLY_CONTRIBUTION
            cash += contribution
            total_contributed += contribution
        prev_month = current_month

        # Get today's prices (active tickers only)
        prices = {}
        for t in tickers:
            if t in data and date in data[t].index:
                close = data[t].loc[date, "Close"]
                if pd.notna(close):
                    prices[t] = float(close)

        if not prices:
            dates.append(date)
            daily_values.append(cash + sum(
                positions[t] * 0 for t in tickers))
            daily_cash.append(cash)
            daily_contributions.append(contribution)
            continue

        # Build history up to today for each active ticker
        history = {}
        for t in prices:
            hist_slice = data[t].loc[:date].dropna(subset=["Close"])
            if len(hist_slice) > 0:
                history[t] = hist_slice

        # Build portfolio state
        portfolio = PortfolioState(
            cash=cash,
            positions={t: positions[t] for t in prices},
            prices=prices,
        )

        # Compute target trades using exact app algorithm
        trades = compute_target_trades(portfolio, history)

        # Execute trades: sells first, then buys
        for trade in trades:
            t = trade["ticker"]
            if trade["action"] == "SELL":
                positions[t] -= trade["shares"]
                cash += trade["amount"]
                trade_count += 1
                trade_log.append((date, "SELL", t, trade["shares"], trade["price"]))
            elif trade["action"] == "BUY":
                # Enforce cash reserve on each buy
                total_eq = sum(positions[tk] * prices.get(tk, 0)
                               for tk in prices)
                tv = cash + total_eq
                min_cash = tv * CASH_RESERVE_PCT
                available = max(cash - min_cash, 0)
                actual_shares = min(trade["shares"],
                                    int(available // trade["price"]))
                if actual_shares >= 1:
                    spend = actual_shares * trade["price"]
                    positions[t] += actual_shares
                    cash -= spend
                    trade_count += 1
                    trade_log.append((date, "BUY", t, actual_shares, trade["price"]))

        # Record daily state
        total_eq = sum(positions[t] * prices.get(t, 0) for t in prices)
        dates.append(date)
        daily_values.append(cash + total_eq)
        daily_cash.append(cash)
        daily_contributions.append(contribution)

    # Build results
    values = pd.Series(daily_values, index=dates)
    cash_series = pd.Series(daily_cash, index=dates)
    contrib_series = pd.Series(daily_contributions, index=dates)

    return {
        "values": values,
        "cash": cash_series,
        "contributions": contrib_series,
        "total_contributed": total_contributed,
        "positions": positions,
        "prices": prices,
        "trade_count": trade_count,
        "trade_log": trade_log,
        "tickers": tickers,
    }


# ---------------------------------------------------------------------------
# SPY benchmark (same contributions, whole shares)
# ---------------------------------------------------------------------------

def run_spy_benchmark(spy_data: pd.DataFrame) -> dict:
    """Simulate buying SPY with the same $5K + $650/mo schedule."""
    dates_all = spy_data.index.tolist()

    cash = INITIAL_CASH
    shares = 0
    total_contributed = INITIAL_CASH
    prev_month = None
    trade_count = 0

    dates = []
    daily_values = []
    daily_contributions = []

    for date in dates_all:
        contribution = 0.0
        current_month = (date.year, date.month)
        if prev_month is not None and current_month != prev_month:
            contribution = MONTHLY_CONTRIBUTION
            cash += contribution
            total_contributed += contribution
        prev_month = current_month

        price = float(spy_data.loc[date, "Close"])

        # Buy as many whole shares as possible (keep 5% reserve)
        tv = cash + shares * price
        min_cash = tv * CASH_RESERVE_PCT
        available = max(cash - min_cash, 0)
        can_buy = int(available // price)
        if can_buy >= 1:
            spend = can_buy * price
            shares += can_buy
            cash -= spend
            trade_count += 1

        dates.append(date)
        daily_values.append(cash + shares * price)
        daily_contributions.append(contribution)

    values = pd.Series(daily_values, index=dates)
    contrib_series = pd.Series(daily_contributions, index=dates)

    return {
        "values": values,
        "contributions": contrib_series,
        "total_contributed": total_contributed,
        "trade_count": trade_count,
        "final_shares": shares,
        "final_cash": cash,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _compute_stats(values: pd.Series, contributions: pd.Series,
                   total_contributed: float, rf_rate: float = 0.0,
                   rf_daily: pd.Series | None = None) -> dict:
    """Compute performance stats for a value series."""
    final = values.iloc[-1]
    n_days = len(values)
    total_ret = final / total_contributed - 1
    ann_ret = (1 + total_ret) ** (252 / n_days) - 1

    market_only = values - contributions
    prev_total = values.shift(1)
    daily_rets = (market_only / prev_total - 1).iloc[1:]
    ann_vol = daily_rets.std() * np.sqrt(252)

    # Daily excess returns for Sharpe
    if rf_daily is not None and len(rf_daily) > 0:
        rf_aligned = rf_daily.reindex(daily_rets.index).ffill().bfill().fillna(0.0)
        daily_excess = daily_rets - rf_aligned / 252
        avg_rf = float(rf_aligned.mean())
    else:
        daily_excess = daily_rets - rf_rate / 252
        avg_rf = rf_rate

    if len(daily_excess) >= 2 and daily_excess.std() > 0:
        sharpe = float(daily_excess.mean() / daily_excess.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    cummax = values.cummax()
    dd = ((values - cummax) / cummax).min()

    return {
        "final": final,
        "net": final - total_contributed,
        "total_return": total_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "rf_rate": avg_rf,
        "max_dd": dd,
    }


def print_report(result: dict, spy_result: dict, rf_rate: float = 0.0,
                 rf_daily: pd.Series | None = None):
    """Print side-by-side summary statistics."""
    values = result["values"]
    n_days = len(values)

    ew = _compute_stats(result["values"], result["contributions"],
                        result["total_contributed"], rf_rate=rf_rate,
                        rf_daily=rf_daily)
    sp = _compute_stats(spy_result["values"], spy_result["contributions"],
                        spy_result["total_contributed"], rf_rate=rf_rate,
                        rf_daily=rf_daily)

    print("=" * 62)
    print(f"  {'':30s} {'Inv-Vol EW':>14s} {'SPY':>14s}")
    print("=" * 62)
    print(f"  {'Period':<30s} {str(values.index[0].date()) + ' to ' + str(values.index[-1].date()):>30s}")
    print(f"  {'Trading days':<30s} {n_days:>30d}")
    print(f"  {'Total invested':<30s} {'$' + f'{result['total_contributed']:,.0f}':>14s} "
          f"{'$' + f'{spy_result['total_contributed']:,.0f}':>14s}")
    print(f"  {'Final value':<30s} {'$' + f'{ew['final']:,.2f}':>14s} "
          f"{'$' + f'{sp['final']:,.2f}':>14s}")
    print(f"  {'Net gain':<30s} {'$' + f'{ew['net']:,.2f}':>14s} "
          f"{'$' + f'{sp['net']:,.2f}':>14s}")
    print(f"  {'Total return':<30s} {ew['total_return']:>14.1%} {sp['total_return']:>14.1%}")
    print(f"  {'Ann. return':<30s} {ew['ann_return']:>14.1%} {sp['ann_return']:>14.1%}")
    print(f"  {'Ann. volatility':<30s} {ew['ann_vol']:>14.1%} {sp['ann_vol']:>14.1%}")
    rf_label = f"Sharpe ratio (Rf={ew['rf_rate']:.1%})"
    print(f"  {rf_label:<30s} {ew['sharpe']:>14.2f} {sp['sharpe']:>14.2f}")
    print(f"  {'Max drawdown':<30s} {ew['max_dd']:>14.1%} {sp['max_dd']:>14.1%}")
    print(f"  {'Trades':<30s} {result['trade_count']:>14d} {spy_result['trade_count']:>14d}")
    print("=" * 62)
    print()

    # Final positions
    positions = result["positions"]
    prices = result["prices"]
    print(f"  {' FINAL POSITIONS ':─^40s}")
    print(f"  {'Ticker':<8} {'Shares':>8} {'Value':>12} {'Alloc':>8}")
    print(f"  {'─' * 40}")
    total_eq = sum(positions[t] * prices.get(t, 0) for t in result["tickers"])
    final_cash = result["cash"].iloc[-1]
    tv = final_cash + total_eq
    for t in result["tickers"]:
        sh = positions[t]
        val = sh * prices.get(t, 0)
        alloc = val / tv if tv > 0 else 0
        print(f"  {t:<8} {sh:>8.0f} {'$' + f'{val:,.2f}':>12} {alloc:>8.1%}")
    cash_alloc = final_cash / tv if tv > 0 else 0
    print(f"  {'Cash':<8} {'':>8} {'$' + f'{final_cash:,.2f}':>12} {cash_alloc:>8.1%}")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(result: dict, spy_result: dict):
    """Generate backtest figures with SPY comparison."""
    import matplotlib.pyplot as plt

    setup_style()
    os.makedirs(os.path.join(SCRIPT_DIR, "figures"), exist_ok=True)

    ew = result["values"]
    spy = spy_result["values"]
    contrib_cumulative = result["contributions"].cumsum()

    # Align to common dates
    common = ew.index.intersection(spy.index)
    ew_c = ew.loc[common]
    spy_c = spy.loc[common]

    # --- Figure 1: Both portfolios vs contributions ---
    fig, ax = plt.subplots(figsize=FIGSIZE["double"])
    ax.plot(ew_c.index, ew_c.values / 1000,
            color=PALETTE["blue"], linewidth=1.5, label="Inv-Vol EW")
    ax.plot(spy_c.index, spy_c.values / 1000,
            color=PALETTE["red"], linewidth=1.5, label="SPY Only")
    contrib_c = contrib_cumulative.loc[common]
    ax.fill_between(common, 0, contrib_c.values / 1000,
                    color=PALETTE["grey"], alpha=0.2, label="Total Invested")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value (\\$K)")
    ax.set_title("Inv-Vol EW vs SPY: \\$5K + \\$650/mo (Whole Shares)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    savefig(fig, os.path.join(SCRIPT_DIR, "figures", "backtest_vs_spy.png"))
    plt.close(fig)

    # --- Figure 2: Relative performance (EW / SPY) ---
    ratio = ew_c / spy_c
    fig, ax = plt.subplots(figsize=FIGSIZE["double"])
    ax.plot(ratio.index, ratio.values,
            color=PALETTE["blue"], linewidth=1.5)
    ax.axhline(1.0, color=PALETTE["grey"], linestyle="--", linewidth=0.8)
    ax.fill_between(ratio.index, 1.0, ratio.values,
                    where=ratio.values >= 1.0,
                    color=PALETTE["blue"], alpha=0.15)
    ax.fill_between(ratio.index, 1.0, ratio.values,
                    where=ratio.values < 1.0,
                    color=PALETTE["red"], alpha=0.15)
    ax.set_xlabel("Date")
    ax.set_ylabel("EW / SPY Ratio")
    ax.set_title("Relative Performance (above 1.0 = EW outperforms)")
    ax.grid(True, alpha=0.3)
    savefig(fig, os.path.join(SCRIPT_DIR, "figures", "backtest_relative.png"))
    plt.close(fig)

    # --- Figure 3: Drawdown comparison ---
    ew_cummax = ew_c.cummax()
    spy_cummax = spy_c.cummax()
    ew_dd = (ew_c - ew_cummax) / ew_cummax * 100
    spy_dd = (spy_c - spy_cummax) / spy_cummax * 100
    fig, ax = plt.subplots(figsize=FIGSIZE["double"])
    ax.fill_between(ew_dd.index, ew_dd.values, 0,
                    color=PALETTE["blue"], alpha=0.3, label="Inv-Vol EW")
    ax.fill_between(spy_dd.index, spy_dd.values, 0,
                    color=PALETTE["red"], alpha=0.3, label="SPY")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (\\%)")
    ax.set_title("Drawdown Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, os.path.join(SCRIPT_DIR, "figures", "backtest_drawdown.png"))
    plt.close(fig)

    # --- Figure 4: Monthly return difference ---
    ew_monthly = ew_c.resample("ME").last()
    spy_monthly = spy_c.resample("ME").last()
    ew_mret = ew_monthly.pct_change().dropna() * 100
    spy_mret = spy_monthly.pct_change().dropna() * 100
    common_months = ew_mret.index.intersection(spy_mret.index)
    diff = ew_mret.loc[common_months] - spy_mret.loc[common_months]
    fig, ax = plt.subplots(figsize=FIGSIZE["double"])
    colors = [PALETTE["blue"] if d >= 0 else PALETTE["red"] for d in diff.values]
    ax.bar(diff.index, diff.values, width=20, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Excess Return (\\%)")
    ax.set_title("EW minus SPY: Monthly Return Difference")
    ax.grid(True, alpha=0.3, axis="y")
    savefig(fig, os.path.join(SCRIPT_DIR, "figures", "backtest_excess.png"))
    plt.close(fig)

    print(f"Figures saved to {os.path.join(SCRIPT_DIR, 'figures')}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Fetch portfolio tickers + SPY
    all_tickers = TICKERS + ["SPY"]
    data = fetch_all(all_tickers)

    spy_data = data.pop("SPY")

    result = run_backtest(data)
    spy_result = run_spy_benchmark(spy_data)

    rf_rate = fetch_risk_free_rate()
    # Fetch historical rf for accurate Sharpe ratio
    start_date = result["values"].index[0]
    rf_daily = fetch_risk_free_history(start_date)
    if len(rf_daily) > 0:
        avg_rf = float(rf_daily.mean())
        print(f"Risk-free rate: avg {avg_rf:.2%} over period "
              f"(spot {rf_rate:.2%})\n")
    else:
        rf_daily = None
        print(f"Risk-free rate (3-mo T-bill): {rf_rate:.2%}\n")
    print_report(result, spy_result, rf_rate=rf_rate, rf_daily=rf_daily)
    plot_results(result, spy_result)


if __name__ == "__main__":
    main()
