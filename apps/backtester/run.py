"""
Run and compare multi-stock portfolio strategies.

Usage:
    python run.py [--tickers T1 T2 ...] [--start DATE] [--end DATE]

Examples:
    python run.py                           # 50 stocks, 10y from today
    python run.py --tickers F GM TSLA       # Custom tickers
    python run.py --start 2018-01-01 --end 2023-01-01 --cash 50000
"""

import os
import sys
import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from finance_tools.utils.plotting import (
    setup_style, savefig, PALETTE, FIGSIZE,
)
from finance_tools.backtest.portfolio import PortfolioBacktester, PortfolioBacktestResult
from finance_tools.strategies.portfolio import (
    EqualWeightRebalance, InverseVolatilityWeight,
    IndependentMeanReversion, RelativeStrength,
)
from finance_tools.data.market import fetch_risk_free_rate, fetch_risk_free_history
from finance_tools.data.universe import ALL_TICKERS
from finance_tools.data.results import ResultsStore

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
FIG_COMP = os.path.join(FIG_DIR, "comparisons")
FIG_ALLOC = os.path.join(FIG_DIR, "allocations")
FIG_DIAG = os.path.join(FIG_DIR, "diagnostics")

STRATEGY_COLORS = [
    PALETTE["blue"], PALETTE["red"], PALETTE["green"],
    PALETTE["yellow"], PALETTE["purple"], PALETTE["cyan"],
]


def plot_portfolio_comparison(results: list[PortfolioBacktestResult],
                              label: str):
    """Portfolio value over time for all strategies."""
    import matplotlib.pyplot as plt

    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE["double"])

    for i, res in enumerate(results):
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        ax.plot(res.daily_values.index, res.daily_values.values,
                color=color, lw=1.2, label=res.strategy_name)

    ax.set_ylabel("Portfolio Value (\\$)")
    ax.set_title(f"{label} --- Portfolio Strategy Comparison")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    savefig(fig, os.path.join(FIG_COMP, "portfolio_comparison.png"))


def plot_allocation_over_time(result: PortfolioBacktestResult, label: str):
    """Stacked area chart showing allocation weights over time."""
    import matplotlib.pyplot as plt

    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE["double"])

    dates = result.daily_values.index

    # Compute daily allocations using actual prices
    alloc_data = {}
    total_vals = result.daily_values.values
    cash_vals = result.daily_cash.values

    for t in result.tickers:
        equity_t = (result.daily_positions[t].values
                    * result.daily_prices[t].values)
        alloc_data[t] = np.where(total_vals > 0, equity_t / total_vals, 0.0)

    # Cash allocation
    cash_alloc = cash_vals / np.where(total_vals > 0, total_vals, 1.0)

    # Stack: tickers + cash
    bottoms = np.zeros(len(dates))
    colors = STRATEGY_COLORS[:len(result.tickers)] + [PALETTE["grey"]]

    for i, t in enumerate(result.tickers):
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        ax.fill_between(dates, bottoms, bottoms + alloc_data[t],
                        color=color, alpha=0.6, label=t)
        bottoms += alloc_data[t]

    ax.fill_between(dates, bottoms, bottoms + cash_alloc,
                    color=PALETTE["grey"], alpha=0.4, label="Cash")

    ax.set_ylabel("Allocation")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{result.strategy_name} --- Allocation Over Time")
    ax.legend(fontsize=6, loc="upper left", ncol=3)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    safe_name = result.strategy_name.replace(" ", "_").replace("/", "_")
    savefig(fig, os.path.join(FIG_ALLOC, f"allocation_{safe_name}.png"))


def plot_ticker_contribution(results: list[PortfolioBacktestResult],
                             label: str):
    """Bar chart of per-ticker return contribution for each strategy."""
    import matplotlib.pyplot as plt

    setup_style()

    n_strategies = len(results)
    fig, axes = plt.subplots(1, n_strategies, figsize=(3.5 * n_strategies, 4),
                             sharey=True)
    if n_strategies == 1:
        axes = [axes]

    for idx, (res, ax) in enumerate(zip(results, axes)):
        contributions = res.ticker_contribution()
        tickers = sorted(contributions.keys())
        vals = [contributions[t] for t in tickers]
        colors_bar = [STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
                      for i in range(len(tickers))]

        ax.bar(tickers, vals, color=colors_bar, alpha=0.8)
        ax.axhline(0, color="black", lw=0.5, ls="--")
        ax.set_title(res.strategy_name, fontsize=8)
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    axes[0].set_ylabel("P\\&L (\\$)")
    fig.suptitle(f"{label} --- Per-Ticker Contribution", fontsize=10)
    fig.tight_layout()
    savefig(fig, os.path.join(FIG_DIAG, "ticker_contribution.png"))


def plot_normalized_to_treasuries(results: list[PortfolioBacktestResult],
                                  rf_rate: float,
                                  rf_daily: pd.Series | None = None):
    """Plot each strategy's value normalized to Treasury growth."""
    import matplotlib.pyplot as plt

    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE["double"])

    for i, res in enumerate(results):
        dates = res.daily_values.index
        values = res.daily_values.values
        initial = values[0]

        if rf_daily is not None and len(rf_daily) > 0:
            # Cumulative treasury growth from daily historical rates
            rf_aligned = rf_daily.reindex(dates).ffill().bfill().fillna(0.0)
            daily_growth = 1 + rf_aligned / 252
            treasury = initial * daily_growth.cumprod().values
        else:
            # Fallback: constant rate
            day0 = dates[0]
            days_elapsed = np.array([(d - day0).days for d in dates])
            treasury = initial * (1 + rf_rate) ** (days_elapsed / 365.25)

        normalized = values / treasury
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        ax.plot(dates, normalized, color=color, lw=1.2,
                label=res.strategy_name)

    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("Value / Treasury Growth")
    ax.set_title("Portfolio Value Normalized to Treasuries")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    savefig(fig, os.path.join(FIG_COMP, "normalized_vs_treasuries.png"))


def plot_normalized_to_spy(results: list[PortfolioBacktestResult],
                           spy_hist, initial_cash: float):
    """Plot each strategy's value normalized to SPY buy-and-hold."""
    import matplotlib.pyplot as plt

    setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE["double"])

    # SPY buy-and-hold growth curve
    spy_close = spy_hist["Close"].values
    spy_dates = spy_hist.index
    spy_value = initial_cash * (spy_close / spy_close[0])

    for i, res in enumerate(results):
        dates = res.daily_values.index
        values = res.daily_values.values

        # Align SPY to strategy dates via nearest-date lookup
        spy_aligned = np.interp(
            [d.timestamp() for d in dates],
            [d.timestamp() for d in spy_dates],
            spy_value,
        )

        normalized = values / spy_aligned
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        ax.plot(dates, normalized, color=color, lw=1.2,
                label=res.strategy_name)

    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("Value / SPY Buy-and-Hold")
    ax.set_title("Portfolio Value Normalized to SPY")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    savefig(fig, os.path.join(FIG_COMP, "normalized_vs_spy.png"))


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-stock portfolio backtest comparison")
    parser.add_argument("--tickers", nargs="+",
                        default=ALL_TICKERS,
                        help="Ticker symbols (default: 100-stock S&P 500 universe)")
    today = date.today().isoformat()
    ten_years_ago = (date.today() - timedelta(days=3653)).isoformat()
    parser.add_argument("--start", default=ten_years_ago,
                        help=f"Start date (default: {ten_years_ago})")
    parser.add_argument("--end", default=today,
                        help=f"End date (default: {today})")
    parser.add_argument("--cash", type=float, default=1_000_000,
                        help="Initial cash (default: $1,000,000)")
    parser.add_argument("--reserve", type=float, default=0.05,
                        help="Cash reserve fraction (default: 0.05)")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]
    for d in [FIG_COMP, FIG_ALLOC, FIG_DIAG]:
        os.makedirs(d, exist_ok=True)

    # Fetch data for all tickers
    print(f"Fetching data for {len(tickers)} tickers ({args.start} to {args.end})...")
    hist_dict = {}
    for ticker in tickers:
        hist = yf.Ticker(ticker).history(start=args.start, end=args.end)
        if hist.empty:
            print(f"  WARNING: No data for '{ticker}', skipping.")
            continue
        hist_dict[ticker] = hist
        print(f"  {ticker}: {len(hist)} trading days")

    if len(hist_dict) < 2:
        print("Need at least 2 tickers with data.")
        sys.exit(1)

    # Fetch SPY for benchmark comparison (not in strategy universe)
    print("  Fetching SPY benchmark...")
    spy_hist = yf.Ticker("SPY").history(start=args.start, end=args.end)
    print(f"  SPY: {len(spy_hist)} trading days")

    # Define strategies
    strategies = [
        EqualWeightRebalance(threshold=0.05),
        InverseVolatilityWeight(threshold=0.05),
        IndependentMeanReversion(20, 2.0),
        RelativeStrength(20),
    ]

    # Fetch risk-free rate for Sharpe calculation
    rf_rate = fetch_risk_free_rate()
    rf_daily = fetch_risk_free_history(args.start, args.end)
    if len(rf_daily) > 0:
        avg_rf = float(rf_daily.mean())
        print(f"\nRisk-free rate: avg {avg_rf:.2%} over period (spot {rf_rate:.2%})")
    else:
        rf_daily = None
        print(f"\nRisk-free rate (3-mo T-bill): {rf_rate:.2%}")

    # Run backtests
    results = []
    print(f"\n{'='*60}")
    for strat in strategies:
        bt = PortfolioBacktester(
            hist_dict, strat,
            initial_cash=args.cash,
            cash_reserve_pct=args.reserve,
        )
        res = bt.run()
        res.rf_rate = rf_rate
        if rf_daily is not None:
            res.rf_daily = rf_daily
        results.append(res)
        print(res.summary())
        print(f"{'='*60}")

    # Save results (clear old runs from this script first)
    store = ResultsStore()
    cleared = store.clear_by_script("run.py")
    if cleared:
        print(f"  Cleared {cleared} old run(s)")
    for res in results:
        run_id = store.save(res, runner_script="run.py")
        print(f"  Saved {res.strategy_name} as run #{run_id}")
    store.close()

    # Figures
    label = "+".join(sorted(hist_dict.keys()))
    print("\nGenerating figures...")
    plot_portfolio_comparison(results, label)

    # Allocation plots for each strategy
    for res in results:
        plot_allocation_over_time(res, label)

    plot_ticker_contribution(results, label)
    plot_normalized_to_treasuries(results, rf_rate, rf_daily=rf_daily)
    plot_normalized_to_spy(results, spy_hist, args.cash)

    print("Done!")


if __name__ == "__main__":
    main()
